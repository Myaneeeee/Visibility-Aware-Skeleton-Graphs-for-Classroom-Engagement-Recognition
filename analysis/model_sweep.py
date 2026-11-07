"""Why logistic regression? Test whether the classifier choice matters at all.

Same folds, same features. LR was only ever a PROBE -- chosen in
diagnose_representation.py so feature blocks are comparable to each other, not
because it is the right final model. This checks whether that choice cost us
anything.
"""
import os, sys, warnings
import numpy as np, torch
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "steps"))
import _bootstrap  # noqa: F401
import common, cv_utils, paths, metrics as M

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NJ = 17
BONES = [(0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,7),(7,9),(6,8),(8,10),
         (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),(5,6)]

d = common.load_labeled_data(paths.SAV_WINDOWS_DIR, paths.ANNOTATIONS_FILE)
X, y = torch.stack(d["tensors"]), np.array(d["labels"])
folds = cv_utils.source_video_folds(d["video_names"], k=5)
N = X.shape[0]

marg = np.zeros((N, 85), np.float32); bone = np.zeros((N, len(BONES)*8), np.float32)
for i in range(N):
    xy = X[i, :2].numpy(); vel = np.diff(xy, axis=1)
    marg[i] = np.concatenate([xy.mean(1).ravel(), xy.std(1).ravel(),
                              np.sqrt((vel**2).sum(0)).mean(0)])
    bv = np.stack([xy[:, :, b] - xy[:, :, a] for a, b in BONES], -1)
    bl = np.sqrt((bv**2).sum(0)); ba = np.arctan2(bv[1], bv[0])
    bone[i] = np.concatenate([bv.mean(1).ravel(), bv.std(1).ravel(), bl.mean(0), bl.std(0),
                              np.cos(ba).mean(0), np.sin(ba).mean(0)])
bone = np.nan_to_num(bone)

MODELS = {
    "logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
    "LR (balanced weights)": lambda: make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=3000, class_weight="balanced")),
    "linear SVM": lambda: make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0)),
    "RBF SVM": lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0)),
    "k-NN (k=15)": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
    "random forest (500)": lambda: RandomForestClassifier(500, n_jobs=-1, random_state=0),
    "extra trees (500)": lambda: ExtraTreesClassifier(500, n_jobs=-1, random_state=0),
    "hist gradient boosting": lambda: HistGradientBoostingClassifier(random_state=0),
    "HGB (balanced)": lambda: HistGradientBoostingClassifier(random_state=0, class_weight="balanced"),
    "sklearn MLP (128,64)": lambda: make_pipeline(StandardScaler(),
                                MLPClassifier((128, 64), max_iter=600, random_state=0)),
}

def evaluate(mk, feats):
    f1 = []
    for tr, te in folds:
        m = mk(); m.fit(feats[tr], y[tr])
        f1.append(M.compute_metrics(y[te], m.predict(feats[te]), 2)["macro_f1"])
    return np.array(f1)

print(f"N={N}, {y.mean():.1%} Distracted, GroupKFold-5 by source video.\n")
print(f"{'classifier':<26}{'on 85 marginals':>18}{'on 144 bone feats':>20}")
print("-" * 64)
best = {}
for name, mk in MODELS.items():
    a, b = evaluate(mk, marg), evaluate(mk, bone)
    best[name] = (a.mean(), b.mean())
    print(f"{name:<26}{a.mean():>12.4f} ±{a.std(ddof=1):<4.3f}{b.mean():>14.4f} ±{b.std(ddof=1):<4.3f}")
print("-" * 64)
from e_ladder_table import arm_stats  # noqa: E402
from paired_ci import ARM  # noqa: E402
print("saved runs, 10 seeds:")
for name, arm in [("  MLP", "MLP"), ("  visible-joint ST-GCN + MC", "E1+mc"),
                  ("  MLP + visible-joint ST-GCN + MC", "E1+mc & MLP")]:
    print(f"{name:<34}{arm_stats(ARM[arm])['macro_f1'][0]:>8.4f}")
bm = max(best.items(), key=lambda kv: kv[1][1])
print(f"\nbest overall: {bm[0]} on bone features = {bm[1][1]:.4f}")

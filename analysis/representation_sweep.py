"""Systematic sweep of skeleton representations, plus the confound checks.

Every block goes through the SAME linear probe, scaler, folds and metric code as
diagnose_representation.py, so every number is comparable to the feature-group
table in outputs/reports/05_representation.md.

Two things this is designed to catch, not just to rank representations:

  * prenormalize_2d parks low-confidence joints at exactly (0, 0). Any distance or
    angle feature involving such a joint therefore encodes WHICH joints were
    unobserved rather than where the body was. The "upper-body only" and
    "confidence/occlusion only" blocks separate geometry from occlusion pattern.
  * bbox position and size describe where a student sits and how big they appear.
    If those alone predict the label, the task has a seating confound.
"""
import os, sys, warnings
import numpy as np, torch
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "steps"))
import _bootstrap  # noqa: F401
import common, cv_utils, paths
from diagnose_representation import linear_probe

NJ = 17
OBSERVED = [i for i in range(NJ) if i not in (13, 14, 15, 16)]   # drop knees/ankles
BONES = [(0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,7),(7,9),(6,8),(8,10),
         (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),(5,6)]
IU_ALL = np.triu_indices(NJ, k=1)                 # 136 pairs
IU_OBS = np.triu_indices(len(OBSERVED), k=1)      # 78 pairs over observed joints

d = common.load_labeled_data(paths.SAV_WINDOWS_DIR, paths.ANNOTATIONS_FILE)
X, y = torch.stack(d["tensors"]), np.array(d["labels"])
folds = cv_utils.source_video_folds(d["video_names"], k=5)
N = X.shape[0]

# raw bboxes, in loader order -- for the seating-confound check
bbox_by_id = {}
import glob
for p in sorted(glob.glob(os.path.join(paths.SAV_WINDOWS_DIR, "*_windows.pt"))):
    for w in torch.load(p, weights_only=False):
        bbox_by_id[common.make_window_id(w)] = np.asarray(w["bboxes"], dtype=np.float64)
BB = [bbox_by_id[i] for i in d["window_ids"]]

F = {}
def put(name, arr):
    F[name] = np.nan_to_num(np.asarray(arr, np.float32))

# ------------------------------------------------------------------ build
marg = np.zeros((N, 85)); dist = np.zeros((N, 272)); dist_obs = np.zeros((N, 156))
bone = np.zeros((N, len(BONES) * 8)); accel = np.zeros((N, 34)); freq = np.zeros((N, 17 * 4))
rom = np.zeros((N, 17 * 3)); half = np.zeros((N, 68)); confF = np.zeros((N, 17 * 2 + 2))
headt = np.zeros((N, 24)); bboxF = np.zeros((N, 8)); margobs = np.zeros((N, 65))

for i in range(N):
    xy, cf = X[i, :2].numpy(), X[i, 2].numpy()            # (2,90,17), (90,17)
    vel = np.diff(xy, axis=1)                             # (2,89,17)
    spd = np.sqrt((vel ** 2).sum(0))                      # (89,17)
    acc = np.diff(vel, axis=1)

    marg[i] = np.concatenate([xy.mean(1).ravel(), xy.std(1).ravel(), spd.mean(0)])
    o = np.array(OBSERVED)
    margobs[i] = np.concatenate([xy[:, :, o].mean(1).ravel(), xy[:, :, o].std(1).ravel(),
                                 spd[:, o].mean(0)])

    dm = np.sqrt(((xy[:, :, :, None] - xy[:, :, None, :]) ** 2).sum(0))   # (90,17,17)
    dist[i] = np.concatenate([dm.mean(0)[IU_ALL], dm.std(0)[IU_ALL]])
    dmo = dm[:, o][:, :, o]
    dist_obs[i] = np.concatenate([dmo.mean(0)[IU_OBS], dmo.std(0)[IU_OBS]])

    bv = np.stack([xy[:, :, b] - xy[:, :, a] for a, b in BONES], -1)       # (2,90,B)
    blen = np.sqrt((bv ** 2).sum(0)); bang = np.arctan2(bv[1], bv[0])
    bone[i] = np.concatenate([bv.mean(1).ravel(), bv.std(1).ravel(),
                              blen.mean(0), blen.std(0),
                              np.cos(bang).mean(0), np.sin(bang).mean(0)])

    accel[i] = np.concatenate([np.sqrt((acc ** 2).sum(0)).mean(0),
                               np.sqrt((acc ** 2).sum(0)).std(0)])
    P = np.abs(np.fft.rfft(spd, axis=0)) ** 2
    nb = P.shape[0]
    freq[i] = np.concatenate([P[1:nb//8].sum(0), P[nb//8:nb//4].sum(0),
                              P[nb//4:nb//2].sum(0), P[nb//2:].sum(0)])
    rom[i] = np.concatenate([np.ptp(xy[0], 0), np.ptp(xy[1], 0),
                             np.percentile(spd, 90, axis=0)])
    h = xy.shape[1] // 2
    half[i] = np.concatenate([(xy[:, h:].mean(1) - xy[:, :h].mean(1)).ravel(),
                              (xy[:, h:].std(1) - xy[:, :h].std(1)).ravel()])
    confF[i] = np.concatenate([cf.mean(0), (cf > 0.5).mean(0),
                               [cf.mean(), (cf > 0.5).mean()]])

    head, torso = [0, 1, 2, 3, 4], [5, 6, 11, 12]
    hc, tc = xy[:, :, head].mean(2), xy[:, :, torso].mean(2)               # (2,90)
    rel = hc - tc                                                          # head offset
    ang = np.arctan2(rel[1], rel[0])
    mag = np.sqrt((rel ** 2).sum(0))
    headt[i] = np.concatenate([
        rel.mean(1), rel.std(1), rel.min(1), rel.max(1), np.ptp(rel, axis=1),   # 10
        [mag.mean(), mag.std(), mag.min(), mag.max()],                          # 4
        [np.cos(ang).mean(), np.sin(ang).mean(), np.cos(ang).std(), np.sin(ang).std()],  # 4
        [np.sqrt((np.diff(rel, axis=1) ** 2).sum(0)).mean()],                   # 1
        [np.sqrt((np.diff(rel, axis=1) ** 2).sum(0)).std()],                    # 1
        xy[:, :, head].std(1).mean(1) - xy[:, :, torso].std(1).mean(1),         # 2
        [xy[:, :, head].std(1).mean(), xy[:, :, torso].std(1).mean()],          # 2
    ])

    b = BB[i]
    cx, cy, w_, h_ = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    bboxF[i] = [cx.mean(), cy.mean(), w_.mean(), h_.mean(),
                cx.std(), cy.std(), (w_ * h_).mean(), (w_ / np.clip(h_, 1, None)).mean()]

put("A. per-joint marginals (the 85 baseline)", marg)
put("B. pairwise distance, all 17 joints", dist)
put("C. pairwise distance, 13 observed joints only", dist_obs)
put("D. bone vectors + lengths + angles", bone)
put("E. head-relative-to-torso geometry", headt)
put("F. acceleration", accel)
put("G. motion frequency bands", freq)
put("H. range of motion / p90 speed", rom)
put("I. first-half vs second-half change", half)
put("J. marginals, 13 observed joints only", margobs)
put("CONFOUND: detector confidence only", confF)
put("CONFOUND: bbox position + size only", bboxF)

def run(mat):
    fm = linear_probe(np.asarray(mat, np.float32), y, folds, 2)
    f1 = np.array([m["macro_f1"] for m in fm])
    return f1, np.mean([m["mcc"] for m in fm])

print(f"N={N} windows, {y.mean():.1%} Distracted, GroupKFold-5 by source video, "
      f"logistic-regression probe.\n")
print(f"{'representation':<46}{'n':>5}{'macroF1':>9}{'sd(fold)':>10}{'MCC':>8}")
print("-" * 78)
res = {}
for k in F:
    f1, mcc = run(F[k]); res[k] = f1
    print(f"{k:<46}{F[k].shape[1]:>5}{f1.mean():>9.4f}{f1.std(ddof=1):>10.4f}{mcc:>8.4f}")

from e_ladder_table import arm_stats  # noqa: E402
from paired_ci import ARM  # noqa: E402
print(f"\n{'reference points (saved runs, 10 seeds)':<46}{'':>5}{'':>9}")
for name, arm, n_feats in [("  MLP", "MLP", 85), ("  visible-joint ST-GCN + MC", "E1+mc", ""),
                           ("  MLP + visible-joint ST-GCN + MC", "E1+mc & MLP", "")]:
    print(f"{name:<46}{n_feats:>5}{arm_stats(ARM[arm])['macro_f1'][0]:>9.4f}")

base = res["A. per-joint marginals (the 85 baseline)"]
print("\nPaired vs the 85 marginals, over the same 5 folds:")
print(f"{'representation':<46}{'delta':>9}{'wins':>7}")
print("-" * 62)
for k, v in sorted(res.items(), key=lambda kv: -kv[1].mean()):
    if k.startswith("A."): continue
    print(f"{k:<46}{v.mean()-base.mean():>+9.4f}{int((v>base).sum()):>5}/5")

# ---------------------------------------------------------------- combinations
print("\n" + "=" * 78)
print("Do the relational blocks combine, and do they survive without the weak joints?")
print("=" * 78)
combos = {
    "marginals (85)": F["A. per-joint marginals (the 85 baseline)"],
    "bone (144)": F["D. bone vectors + lengths + angles"],
    "marginals + bone": np.hstack([F["A. per-joint marginals (the 85 baseline)"],
                                   F["D. bone vectors + lengths + angles"]]),
    "bone + distance(obs)": np.hstack([F["D. bone vectors + lengths + angles"],
                                       F["C. pairwise distance, 13 observed joints only"]]),
    "marginals + bone + distance(obs)": np.hstack([
        F["A. per-joint marginals (the 85 baseline)"],
        F["D. bone vectors + lengths + angles"],
        F["C. pairwise distance, 13 observed joints only"]]),
    "everything above + bbox": np.hstack([
        F["A. per-joint marginals (the 85 baseline)"],
        F["D. bone vectors + lengths + angles"],
        F["C. pairwise distance, 13 observed joints only"],
        F["CONFOUND: bbox position + size only"]]),
}
print(f"{'combination':<40}{'n':>6}{'macroF1':>9}{'sd(fold)':>10}{'vs 85':>9}{'wins':>7}")
print("-" * 81)
for k, m in combos.items():
    f1, _ = run(m)
    print(f"{k:<40}{m.shape[1]:>6}{f1.mean():>9.4f}{f1.std(ddof=1):>10.4f}"
          f"{f1.mean()-base.mean():>+9.4f}{int((f1>base).sum()):>5}/5")

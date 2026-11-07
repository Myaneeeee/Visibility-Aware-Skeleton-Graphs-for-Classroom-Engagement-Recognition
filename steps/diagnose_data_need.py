"""Fact-gathering diagnostics: is the bottleneck data, or the model?

Two questions the ablation table cannot answer on its own.

1. LOGISTIC-REGRESSION CONTROL.
   The MLP baseline beats every ST-GCN rung. Two explanations are consistent
   with that: (a) the 85 hand-made pose statistics carry nearly all the signal,
   so any classifier on top of them wins, or (b) the MLP's nonlinearity is
   doing real work. A linear model on the identical features, over the
   identical folds, separates the two. If the linear model is also ~0.73, the
   finding is about the FEATURES, not the model class.

2. TWO LEARNING CURVES, PLOTTED AGAINST THE SAME X-AXIS.
   Both curves shrink the TRAINING fold to the same number of windows; the
   test fold is always kept whole, so every point is measured on the same
   evaluation data.
     - "rooms"   : whole source videos are dropped (the key keeps its old
                   name so existing JSON still loads). Simulates having downloaded
                   fewer source videos.
     - "windows" : all videos are kept, random windows dropped (stratified, so
                   the class ratio stays put and only the QUANTITY changes).
                   Simulates having labeled less per video.
   Reading the result: if "windows" is still climbing at 100%, labeling the
   more of the extracted-but-unlabeled windows should help. If "rooms" sits
   clearly below "windows" at equal window counts, video diversity is the
   scarce thing and more source videos are what is needed.

Runs the MLP only -- it is the current best model, and it is cheap enough on
CPU to leave the GPU free for the ablation sweep.

Writes outputs/diagnostics/data_need.json.
"""
import argparse
import json
import os

import numpy as np
import torch

import _bootstrap  # noqa: F401  -- puts ../lib on sys.path
import baselines as bl
import common
import cv_utils
import metrics as metrics_mod
import paths
import evaluate as ev

FRACTIONS = [0.25, 0.5, 0.75, 1.0]
SEEDS = [51, 52, 53]
K = 5
EPOCHS = 40
DEVICE = "cpu"


def fold_mean(fold_metrics, key):
    return float(np.mean([m[key] for m in fold_metrics]))


def fold_sd(fold_metrics, key):
    return float(np.std([m[key] for m in fold_metrics], ddof=1))


def logistic_control(feats, labels, folds, num_classes):
    """Linear model on the same features, same folds, same metric code."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    fold_metrics = []
    for train_idx, test_idx in folds:
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000))
        clf.fit(feats[train_idx], labels[train_idx])
        pred = clf.predict(feats[test_idx])
        fold_metrics.append(
            metrics_mod.compute_metrics(labels[test_idx], pred, num_classes))
    return fold_metrics


def subsample_rooms(train_idx, train_rooms, target_n, rng):
    """Keep whole source videos until target_n windows are reached."""
    rooms = list(dict.fromkeys(train_rooms))
    rng.shuffle(rooms)
    keep, n = set(), 0
    for r in rooms:
        keep.add(r)
        n = int(np.sum([g in keep for g in train_rooms]))
        if n >= target_n:
            break
    return train_idx[np.array([g in keep for g in train_rooms])]


def subsample_windows(train_idx, train_y, target_n, rng):
    """Keep a class-stratified random subset, so only the quantity changes."""
    keep = []
    for cls in np.unique(train_y):
        pos = np.where(train_y == cls)[0]
        # Round up so the rarer class is never wiped out by flooring.
        take = max(1, int(round(len(pos) * target_n / len(train_y))))
        keep.extend(rng.choice(pos, size=min(take, len(pos)), replace=False))
    return train_idx[np.array(sorted(keep))]


def learning_curve(X, labels, groups, folds, num_classes, model_kind,
                   device, fixed_steps, backbone_path=None):
    """Macro-F1 vs training-set size, under both shrink modes.

    fixed_steps decouples "more data" from "more optimization". At a constant
    epoch count a bigger subset also gets proportionally more gradient updates,
    so a flat curve could just as well mean the full set is overfitting at 40
    epochs. Scaling epochs by 1/frac holds the update budget constant, and the
    conclusion only stands if both variants agree.
    """
    y = torch.tensor(labels, dtype=torch.long)
    rows = []
    for mode in ["rooms", "windows"]:
        for frac in FRACTIONS:
            for seed in SEEDS:
                ev.set_determinism(seed)
                rng = np.random.default_rng(seed)
                fold_metrics, sizes, room_counts = [], [], []
                epochs = int(round(EPOCHS / frac)) if fixed_steps else EPOCHS
                for train_idx, test_idx in folds:
                    target_n = int(round(len(train_idx) * frac))
                    if frac >= 1.0:
                        sub = train_idx
                    elif mode == "rooms":
                        sub = subsample_rooms(train_idx, groups[train_idx],
                                              target_n, rng)
                    else:
                        sub = subsample_windows(train_idx, labels[train_idx],
                                                target_n, rng)
                    sizes.append(len(sub))
                    room_counts.append(len(set(groups[sub])))
                    model = ev._make_model(model_kind, num_classes,
                                           backbone_path=backbone_path)
                    model = ev._train_fold(model, X[sub], y[sub], epochs, device)
                    model.eval()
                    with torch.no_grad():
                        pred = model(X[test_idx].to(device)).argmax(dim=1).cpu().numpy()
                    fold_metrics.append(
                        metrics_mod.compute_metrics(labels[test_idx], pred,
                                                    num_classes))
                rows.append({
                    "mode": mode,
                    "frac": frac,
                    "seed": seed,
                    "epochs": epochs,
                    "n_train_mean": float(np.mean(sizes)),
                    "n_rooms_mean": float(np.mean(room_counts)),
                    "macro_f1": fold_mean(fold_metrics, "macro_f1"),
                    "balanced_acc": fold_mean(fold_metrics, "balanced_acc"),
                    "mcc": fold_mean(fold_metrics, "mcc"),
                })
                print(f"  {mode:8s} frac={frac:.2f} seed={seed} "
                      f"n={rows[-1]['n_train_mean']:.0f} ep={epochs} "
                      f"macroF1={rows[-1]['macro_f1']:.4f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["mlp", "stgcn"], default="mlp")
    ap.add_argument("--pretrained", action="store_true",
                    help="stgcn only: warm-start from the NTU-60 backbone (config B)")
    ap.add_argument("--fixed-steps", action="store_true",
                    help="scale epochs by 1/frac so every point gets the same "
                         "number of gradient updates")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = common.load_labeled_data(
        pt_dir=paths.SAV_WINDOWS_FILTERED_DIR,
        annotations_csv=paths.ANNOTATIONS_FILE,
    )
    num_classes = len(common.CLASS_NAMES)
    labels = np.asarray(data["labels"])
    feats = np.stack([bl.pose_stat_features(t) for t in data["tensors"]])
    groups = np.array([cv_utils.cv_group_key(v) for v in data["video_names"]])
    folds = list(cv_utils.group_kfold_indices(list(groups), k=K))

    print(f"{len(labels)} windows | {len(set(groups))} source videos | "
          f"{int((labels == 1).sum())} Distracted", flush=True)
    print("source videos per fold (train/test): " + ", ".join(
        f"{len(set(groups[tr]))}/{len(set(groups[te]))}" for tr, te in folds),
        flush=True)

    payload = {"n_windows": int(len(labels)), "n_rooms": int(len(set(groups))),
               "model": args.model, "pretrained": bool(args.pretrained),
               "fixed_steps": bool(args.fixed_steps)}

    # The linear control only concerns the 85 summary features, so it is run
    # once, with the MLP curve, and skipped for the ST-GCN sweep.
    if args.model == "mlp":
        print("\n[1] logistic-regression control", flush=True)
        lr_metrics = logistic_control(feats, labels, folds, num_classes)
        for key in ["macro_f1", "balanced_acc", "mcc"]:
            print(f"  LR {key:18s} {fold_mean(lr_metrics, key):.4f} "
                  f"(sd_fold {fold_sd(lr_metrics, key):.4f})", flush=True)
        payload["logistic_regression"] = {
            k: {"mean": fold_mean(lr_metrics, k),
                "sd_fold": fold_sd(lr_metrics, k)}
            for k in ["macro_f1", "balanced_acc", "mcc"]}

    if args.model == "mlp":
        X = torch.tensor(feats, dtype=torch.float32)
        device, backbone = DEVICE, None
    else:
        X = torch.stack(data["tensors"])
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        backbone = paths.NTU_BACKBONE_FILE if args.pretrained else None

    print(f"\n[2] learning curves | model={args.model} device={device} "
          f"pretrained={bool(backbone)} fixed_steps={args.fixed_steps}", flush=True)
    payload["learning_curve"] = learning_curve(
        X, labels, groups, folds, num_classes, args.model, device,
        args.fixed_steps, backbone_path=backbone)

    os.makedirs(paths.diagnostics_dir(), exist_ok=True)
    out = args.out or paths.data_need_file("data_need.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()

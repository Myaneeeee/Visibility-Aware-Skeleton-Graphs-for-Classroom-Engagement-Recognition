"""Why does the ST-GCN lose to an MLP on hand-made pose statistics?

`diagnose_data_need.py` answers "is the bottleneck data?" (no -- every curve is
still rising). This script answers the other half: what does the 2-D skeleton
representation actually carry, and how far is any model on it from a human?

Three measurements, none of which need a GPU or a training run.

1. FEATURE-GROUP ABLATION.
   The 85 baseline features are three groups of a very different character:
   per-joint mean position (static posture), per-joint std over time (how much
   the joint moves), and per-joint mean speed. Fitting a linear model to each
   group over the pre-registered folds says where the signal lives. If `std`
   alone nearly matches all 85, the task is closer to motion-energy detection
   than to posture recognition -- and the baseline is being handed a
   sufficient statistic that the ST-GCN has to learn from raw coordinates.

2. PER-JOINT KEYPOINT CONFIDENCE.
   The ST-GCN's adjacency matrix is a fixed human skeleton. If some joints are
   never actually observed, those graph edges propagate noise on every forward
   pass, and the MLP's advantage is partly that a linear layer can simply
   down-weight the corresponding columns.

3. HUMAN CEILING FROM THE BLIND OVERLAP.
   `agreement.py` reports kappa. Kappa is chance-corrected agreement, not an
   achievable score, so it cannot be compared to a model's macro-F1. This does
   the comparable thing: score each annotator against the other two where those
   two agree, which is the same "predict the consensus label" task the model is
   scored on. Annotators see the RGB clip, so this bounds what is reachable
   with information the skeleton throws away.

   Restricted to overlap windows all three judged trainable, so it is biased
   toward easy windows and is an UPPER bound, not an estimate.

Writes outputs/diagnostics/representation.json. Costs about a minute on CPU.
"""
import argparse
import collections
import csv
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

# The 85-d vector `baselines.pose_stat_features` builds, in the order it builds
# it: mean(2x17) then std(2x17) then speed(17). Slicing it is how we ask which
# kind of information the classifier is actually using.
FEATURE_GROUPS = [
    ("mean only (static posture)", 0, 34),
    ("std only (per-joint motion spread)", 34, 68),
    ("speed only (mean joint speed)", 68, 85),
    ("mean + std (no speed)", 0, 68),
    ("std + speed (motion only, no absolute position)", 34, 85),
    ("all 85 (the MLP/LR baseline features)", 0, 85),
]

# COCO-17, in the order the pose estimator emits them.
COCO17 = ["nose", "eye_L", "eye_R", "ear_L", "ear_R",
          "shoulder_L", "shoulder_R", "elbow_L", "elbow_R",
          "wrist_L", "wrist_R", "hip_L", "hip_R",
          "knee_L", "knee_R", "ankle_L", "ankle_R"]
# Joints below this mean confidence are reported as unobserved. 0.35 is well
# under the 0.5 a detector would need to be called a positive, and it cleanly
# separates the seated-behind-a-desk joints from everything else.
UNOBSERVED_CONF = 0.35


def linear_probe(feats, labels, folds, num_classes):
    """Out-of-fold metrics for a linear model on `feats` over `folds`.

    Deliberately the same estimator, scaler and metric code as
    `diagnose_data_need.logistic_control`, so a number here is comparable to
    the LR control already reported in 04_data_need.md.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    fold_metrics = []
    for train_idx, test_idx in folds:
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=3000))
        clf.fit(feats[train_idx], labels[train_idx])
        pred = clf.predict(feats[test_idx])
        fold_metrics.append(
            metrics_mod.compute_metrics(labels[test_idx], pred, num_classes))
    return fold_metrics


def feature_group_ablation(feats, labels, folds, num_classes,
                           groups=FEATURE_GROUPS):
    rows = []
    for name, lo, hi in groups:
        fm = linear_probe(feats[:, lo:hi], labels, folds, num_classes)
        rows.append({
            "name": name, "start": lo, "stop": hi, "n_features": hi - lo,
            "macro_f1": float(np.mean([m["macro_f1"] for m in fm])),
            "macro_f1_sd_fold": float(np.std([m["macro_f1"] for m in fm], ddof=1)),
            "balanced_acc": float(np.mean([m["balanced_acc"] for m in fm])),
            "mcc": float(np.mean([m["mcc"] for m in fm])),
        })
    return rows


def joint_confidence(tensors, joint_names=COCO17):
    """Mean detector confidence per joint, over every frame of every window.

    Channel 2 of a window is the per-keypoint confidence the pose estimator
    emitted, carried through `prenormalize_2d` untouched.
    """
    stacked = torch.stack(list(tensors))          # (N, 3, T, V)
    conf = stacked[:, 2].numpy()                  # (N, T, V)
    means = conf.mean(axis=(0, 1))
    return [{"joint": n, "index": i, "mean_confidence": float(means[i]),
             "observed": bool(means[i] >= UNOBSERVED_CONF)}
            for i, n in enumerate(joint_names)]


def read_ledgers(ledgers_dir):
    """{window_id: {annotator_id: label}} across every per-annotator ledger."""
    out = collections.defaultdict(dict)
    for name in sorted(os.listdir(ledgers_dir)):
        if not name.endswith(".csv"):
            continue
        who = os.path.splitext(name)[0]
        with open(os.path.join(ledgers_dir, name), newline="") as f:
            for row in csv.DictReader(f):
                wid, label = row.get("window_id"), row.get("label")
                if wid and label:
                    out[wid][who] = label
    return out


def human_ceiling(ledgers, class_to_idx=None):
    """Score each annotator against the consensus of the other two.

    This is the model's task, given to a human: predict the label the
    consolidation step would assign. Restricted to blind-overlap windows that
    all three annotators judged trainable -- a window someone called TRASH has
    no Focused/Distracted truth to be right or wrong about.

    Scored only where the other two AGREE, so the target is never the judge's
    own vote. That removes the bias in scoring against a majority the judge
    helped form.
    """
    class_to_idx = class_to_idx or common.CLASS_TO_IDX
    overlap = {w: d for w, d in ledgers.items() if len(d) >= 2}
    trainable = {w: d for w, d in overlap.items()
                 if len(d) == 3 and all(v in class_to_idx for v in d.values())}
    judges = sorted({j for d in trainable.values() for j in d})

    per_annotator = []
    for judge in judges:
        others = [j for j in judges if j != judge]
        y_true, y_pred = [], []
        for d in trainable.values():
            if d[others[0]] == d[others[1]]:
                y_true.append(class_to_idx[d[others[0]]])
                y_pred.append(class_to_idx[d[judge]])
        if not y_true:
            continue
        m = metrics_mod.compute_metrics(y_true, y_pred, len(class_to_idx))
        per_annotator.append({
            "annotator": judge, "n": len(y_true),
            "accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
            "macro_f1": m["macro_f1"], "mcc": m["mcc"],
        })
    return {
        "n_overlap_windows": len(overlap),
        "n_all_three_trainable": len(trainable),
        "per_annotator": per_annotator,
        "mean_accuracy": float(np.mean([a["accuracy"] for a in per_annotator]))
        if per_annotator else float("nan"),
        "mean_macro_f1": float(np.mean([a["macro_f1"] for a in per_annotator]))
        if per_annotator else float("nan"),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=5, help="CV folds (default 5)")
    ap.add_argument("--out", default=None,
                    help="output json (default outputs/diagnostics/representation.json)")
    args = ap.parse_args(argv)

    data = common.load_labeled_data(paths.SAV_WINDOWS_FILTERED_DIR,
                                    paths.ANNOTATIONS_FILE)
    tensors, labels = data["tensors"], np.asarray(data["labels"])
    video_names = data["video_names"]
    num_classes = len(common.CLASS_NAMES)

    folds = cv_utils.source_video_folds(video_names, k=args.k)
    groups = sorted({cv_utils.cv_group_key(v) for v in video_names})
    feats = np.stack([bl.pose_stat_features(t) for t in tensors])

    print(f"{len(labels)} windows across {len(groups)} source videos "
          f"-> {args.k} folds | features {feats.shape}")

    print("\n--- feature-group ablation (linear probe, pre-registered folds) ---")
    fg = feature_group_ablation(feats, labels, folds, num_classes)
    for r in fg:
        print(f"  {r['name']:48s} n={r['n_features']:3d}  "
              f"macro_f1 {r['macro_f1']:.4f}")

    print("\n--- per-joint keypoint confidence ---")
    jc = joint_confidence(tensors)
    for r in sorted(jc, key=lambda z: -z["mean_confidence"]):
        flag = "" if r["observed"] else "   <- effectively unobserved"
        print(f"  {r['joint']:11s} {r['mean_confidence']:.3f}{flag}")

    print("\n--- human ceiling on the blind overlap ---")
    hc = human_ceiling(read_ledgers(paths.SAV_LEDGERS_DIR))
    for a in hc["per_annotator"]:
        print(f"  {a['annotator']:14s} vs other-two-agree (n={a['n']:4d})  "
              f"accuracy {a['accuracy']:.4f}  macro_f1 {a['macro_f1']:.4f}")
    print(f"  MEAN                                    "
          f"accuracy {hc['mean_accuracy']:.4f}  macro_f1 {hc['mean_macro_f1']:.4f}")

    payload = {
        "n_windows": int(len(labels)),
        "n_source_videos": len(groups),
        "distracted_rate": float(labels.mean()),
        "k_folds": args.k,
        "feature_groups": fg,
        "joint_confidence": jc,
        "unobserved_conf_threshold": UNOBSERVED_CONF,
        "human_ceiling": hc,
    }
    out = args.out or paths.representation_file()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {os.path.relpath(out, paths.REPO_ROOT)}")
    return payload


if __name__ == "__main__":
    main()

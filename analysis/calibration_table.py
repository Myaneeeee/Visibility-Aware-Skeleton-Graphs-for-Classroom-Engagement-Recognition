"""Brier, NLL and ECE for the paper's main models (PREREGISTRATION.md §5b.5), and
selective prediction for the combined system against the floor on the retained samples.

Each metric is computed on a seed's pooled out-of-fold probabilities, then
averaged over seeds 51-60. "+temp" divides by a temperature fitted leave one
fold out, so it never sees the fold it calibrates. The prior row predicts the
training-set class rate for every sample: the floor any model must beat.

Usage:  .venv/bin/python analysis/calibration_table.py
"""
import os
import sys

import numpy as np

sys.path[:0] = [os.path.dirname(__file__), "lib", "steps"]
import calibration as cal  # noqa: E402
from e_ladder_table import SEEDS, load  # noqa: E402

ARMS = [
    ("full-skeleton ST-GCN", [("C", "stgcn", "oof_prob")]),
    ("full-skeleton ST-GCN + MC", [("C", "stgcn", "mc_prob")]),
    ("visible-joint ST-GCN", [("E1", "stgcn", "oof_prob")]),
    ("visible-joint ST-GCN + MC", [("E1", "stgcn", "mc_prob")]),
    ("MLP", [("C", "mlp", "oof_prob")]),
    ("MLP + visible-joint ST-GCN + MC", [("C", "mlp", "oof_prob"), ("E1", "stgcn", "mc_prob")]),
]


def seed_probs(specs, seed):
    blobs = [load(f"{label}_seed{seed}", model) for label, model, _ in specs]
    if any(b is None for b in blobs):
        return None
    prob = np.mean([np.asarray(b[k], dtype=np.float64)
                    for b, (_, _, k) in zip(blobs, specs)], axis=0)
    return prob, np.asarray(blobs[0]["oof_true"]), np.asarray(blobs[0]["fold_of"])


def scores(prob, y):
    return cal.brier_score(prob, y), cal.nll(prob, y), cal.ece_equal_mass(prob, y)


def main():
    print("Mean over seeds 51-60 (sd over seeds); ECE over 15 equal-mass bins\n")
    print(f"{'model':<34}{'':>6}{'Brier':>16}{'NLL':>16}{'ECE':>16}")
    print("-" * 88)
    prior = None
    for name, specs in ARMS:
        raw, temp = [], []
        for seed in SEEDS:
            got = seed_probs(specs, seed)
            if got is None:
                break
            prob, y, fold_of = got
            raw.append(scores(prob, y))
            temp.append(scores(cal.leave_one_fold_out_temperature(prob, y, fold_of)[0], y))
            if prior is None:
                p = np.zeros_like(prob)
                for f in np.unique(fold_of):
                    rate = (y[fold_of != f] == 1).mean()
                    p[fold_of == f] = [1 - rate, rate]
                prior = scores(p, y)
        if len(raw) < len(SEEDS):
            print(f"{name:<34}   (runs missing)")
            continue
        for tag, rows in (("raw", raw), ("+temp", temp)):
            a = np.array(rows)
            cells = "".join(f"{m:>9.4f} ({s:.4f})" for m, s in zip(a.mean(0), a.std(0, ddof=1)))
            print(f"{name if tag == 'raw' else '':<34}{tag:>6}{cells}")
    print("-" * 88)
    print(f"{'class-rate prior (no model)':<34}{'':>6}"
          + "".join(f"{m:>9.4f}{'':>7}" for m in prior))
    selective(ARMS[-1])


def selective(arm):
    """Accuracy at coverage for one arm, beside the majority floor on the retained samples."""
    from analyze_paper_arms import COVERAGE_POINTS, selective_against_floor
    name, specs = arm
    rows = [selective_against_floor(*seed_probs(specs, seed)[:2]) for seed in SEEDS]
    print(f"\nSelective prediction, {name} (mean over seeds 51-60)\n")
    print(f"{'coverage':>9}{'accuracy':>10}{'floor':>9}{'gap':>9}{'% off-task kept':>17}")
    for i, c in enumerate(COVERAGE_POINTS):
        col = lambda k: np.mean([r[i][k] for r in rows])   # noqa: E731
        print(f"{c:>9.0%}{col('accuracy'):>10.1%}{col('floor_accuracy'):>9.1%}"
              f"{col('gap') * 100:>+8.1f}{col('distracted_frac'):>17.1%}")


if __name__ == "__main__":
    main()

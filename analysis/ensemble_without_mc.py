"""Does the MLP + ST-GCN ensemble need MC dropout?

The paper's ensemble averages the MLP with the 13-joint ST-GCN's 20-pass MC-dropout
probabilities. This compares it with the same ensemble built from the ST-GCN's single
deterministic pass, and checks the one-pass ensemble against the member-matched
controls (two MLP seeds, two ST-GCN seeds). Equal-weight probability averages of saved
out-of-fold predictions; nothing is fitted.

"x2" arms pair seed s with the next seed (60 wraps to 51), so each instance appears in
two rows and their paired p-values are optimistic.

Usage:  .venv/bin/python analysis/ensemble_without_mc.py [--draws 2000]
"""
import argparse
import os
import sys

import numpy as np

sys.path[:0] = [os.path.dirname(__file__), "lib", "steps"]
from e_ladder_table import SEEDS, fold_rows  # noqa: E402
from paired_ci import prepare, print_pairs  # noqa: E402

MLP = ("C", "mlp", "oof_prob")

ARM = {
    "MLP": [MLP],
    "MLP x2": [MLP, MLP + (1,)],
    "E1": [("E1", "stgcn", "oof_prob")],
    "E1+mc": [("E1", "stgcn", "mc_prob")],
    "E1 x2": [("E1", "stgcn", "oof_prob"), ("E1", "stgcn", "oof_prob", 1)],
    "E1+mc x2": [("E1", "stgcn", "mc_prob"), ("E1", "stgcn", "mc_prob", 1)],
    "E1 & MLP": [("E1", "stgcn", "oof_prob"), MLP],
    "E1+mc & MLP": [("E1", "stgcn", "mc_prob"), MLP],
    "E6 & MLP": [("E6", "stgcn", "oof_prob"), MLP],
    "E6+mc & MLP": [("E6", "stgcn", "mc_prob"), MLP],
    "C & MLP": [("C", "stgcn", "oof_prob"), MLP],
    "C+mc & MLP": [("C", "stgcn", "mc_prob"), MLP],
}

PAIRS = [
    ("what MC dropout adds inside the ensemble", [
        ("E1 & MLP", "E1+mc & MLP"),
        ("E6 & MLP", "E6+mc & MLP"),
        ("C & MLP", "C+mc & MLP"),
    ]),
    ("the one-pass ensemble against its controls", [
        ("MLP", "E1 & MLP"),
        ("MLP x2", "E1 & MLP"),
        ("E1 x2", "E1 & MLP"),
        ("E1", "E1 & MLP"),
    ]),
    ("the paper's ensemble, for reference", [
        ("MLP", "E1+mc & MLP"),
        ("MLP x2", "E1+mc & MLP"),
        ("E1+mc x2", "E1+mc & MLP"),
    ]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    point, boot = prepare(ARM, args.draws, rng)
    for k in ("MLP", "E1+mc & MLP"):
        rows = [np.mean([r["macro_f1"] for r in fold_rows(ARM[k], s)]) for s in SEEDS]
        assert np.allclose(point[k], rows), f"{k}: estimator differs from the table"

    print("Macro-F1, mean over seeds 51-60 (sd(seed)); ST-GCN passes per window at inference\n")
    for name, specs in ARM.items():
        if name in point:
            passes = sum(20 if key == "mc_prob" else 1 for _, model, key, *_ in specs
                         if model == "stgcn")
            print(f"{name:<14}{point[name].mean():>9.4f}  ({point[name].std(ddof=1):.4f})"
                  f"{passes:>6}")

    print(f"\nMacro-F1 paired differences, seeds 51-60; video bootstrap {args.draws} draws")
    print_pairs(point, boot, PAIRS, rng)


if __name__ == "__main__":
    main()

"""Read the multi-seed MLP feature probe, paired over seeds on identical folds.

Each family's MLP is compared with the MLP on the family it is claimed to beat.
Rows pair only seeds present for both arms; the seed count is printed, so check
it before reading a p-value.

Usage:  .venv/bin/python analysis/probe_compare.py
"""
import sys

sys.path[:0] = ["analysis", "lib", "steps"]
from e_ladder_compare import report

MLP = lambda family: [(f"R-{family}", "mlp", "oof_prob")]   # noqa: E731

if __name__ == "__main__":
    print("#" * 62)
    print("#  Sec. 3.2 -- does dropping the dead joints help the MLP too?")
    print("#" * 62)
    report("marginals: 13 joints vs 17", "marg85", MLP("marg85"),
           "marg65", MLP("marg65"), per_seed=True)
    report("pairwise distance: 13 joints vs 17", "dist272", MLP("dist272"),
           "dist156", MLP("dist156"))

    print("\n" + "#" * 62)
    print("#  Sec. 3.3 -- does relative geometry beat per-joint marginals?")
    print("#" * 62)
    report("bone features vs marginals", "marg85", MLP("marg85"),
           "bone144", MLP("bone144"), per_seed=True)
    report("pairwise distance vs marginals", "marg85", MLP("marg85"),
           "dist272", MLP("dist272"))

    print("\n" + "#" * 62)
    print("#  Guard -- R-marg85 must reproduce C's MLP")
    print("#" * 62)
    report("R-marg85 vs C's saved MLP", "C mlp", [("C", "mlp", "oof_prob")],
           "R-marg85", MLP("marg85"))

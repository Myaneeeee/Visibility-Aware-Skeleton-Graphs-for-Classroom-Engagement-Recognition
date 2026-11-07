"""Full metric table for the E ladder, every arm on identical folds.

Protocol, unchanged from the pre-registered analysis: metrics are computed
per fold, averaged within a seed, then averaged over seeds 51-60. sd(seed) is
the bar for comparing two arms on the same folds; sd(fold) is what predicts a
new classroom, and both are printed because they answer different questions.

Ensembles are probability averages of saved out-of-fold predictions: no
retraining, no fitted parameters. A spec may carry a fourth element, a seed
offset: "x2 seeds" arms average seed s with the next seed (60 wraps to 51),
which is the member-count control for any two-model average. Folds do not
depend on the seed, so this pairing is legal, but each seed then appears in
two rows and those p-values are optimistic.

Usage:  .venv/bin/python analysis/e_ladder_table.py
"""
import os
import sys

import numpy as np
import torch
from scipy import stats

sys.path[:0] = ["lib", "steps"]
import metrics as metrics_mod
import paths

SEEDS = list(range(51, 61))
NUM_CLASSES = 2
KEYS = ["macro_f1", "accuracy", "balanced_acc", "mcc"]

ARMS = [
    ("C (ST-GCN-17)",      [("C", "stgcn", "oof_prob")]),
    ("E1 (ST-GCN-13)",     [("E1", "stgcn", "oof_prob")]),
    ("E2 (weighted-17)",   [("E2", "stgcn", "oof_prob")]),
    ("E3 (bone-13)",       [("E3", "stgcn", "oof_prob")]),
    ("E5 (readout-13)",    [("E5", "stgcn", "oof_prob")]),
    ("E6 (input-mask-13)", [("E6", "stgcn", "oof_prob")]),
    ("E7 (joint-bn-17)",   [("E7", "stgcn", "oof_prob")]),
    ("E1+E3 2-stream",    [("E1", "stgcn", "oof_prob"), ("E3", "stgcn", "oof_prob")]),
    ("C+mc (rung D)",      [("C", "stgcn", "mc_prob")]),
    ("E1+mc",              [("E1", "stgcn", "mc_prob")]),
    ("E2+mc",              [("E2", "stgcn", "mc_prob")]),
    ("E3+mc",              [("E3", "stgcn", "mc_prob")]),
    ("E5+mc",              [("E5", "stgcn", "mc_prob")]),
    ("E6+mc",              [("E6", "stgcn", "mc_prob")]),
    ("E7+mc",              [("E7", "stgcn", "mc_prob")]),
    ("E1+E3 2-stream+mc",  [("E1", "stgcn", "mc_prob"), ("E3", "stgcn", "mc_prob")]),
    ("GRU baseline",       [("C", "gru", "oof_prob")]),
    ("MLP baseline",       [("C", "mlp", "oof_prob")]),
    ("C+mc & MLP",         [("C", "stgcn", "mc_prob"), ("C", "mlp", "oof_prob")]),
    ("E1+mc & MLP",        [("E1", "stgcn", "mc_prob"), ("C", "mlp", "oof_prob")]),
    ("E3+mc & MLP",        [("E3", "stgcn", "mc_prob"), ("C", "mlp", "oof_prob")]),
    ("2-stream+mc & MLP",  [("E1", "stgcn", "mc_prob"), ("E3", "stgcn", "mc_prob"),
                            ("C", "mlp", "oof_prob")]),
    ("E1+mc x2 seeds",     [("E1", "stgcn", "mc_prob"), ("E1", "stgcn", "mc_prob", 1)]),
    ("MLP x2 seeds",       [("C", "mlp", "oof_prob"), ("C", "mlp", "oof_prob", 1)]),
]


def load(label, model):
    path = paths.oof_probs_file(label, model)
    return (torch.load(path, map_location="cpu", weights_only=False)
            if os.path.exists(path) else None)


def fold_rows(specs, seed):
    """Per-fold metric dicts at one seed, for one arm."""
    probs, ref = [], None
    for label, model, key, *offset in specs:
        shift = SEEDS.index(seed) + (offset[0] if offset else 0)
        blob = load(f"{label}_seed{SEEDS[shift % len(SEEDS)]}", model)
        if blob is None:
            return None
        prob = np.asarray(blob[key], dtype=np.float64)
        if not np.isfinite(prob).all():
            return None
        if ref is None:
            ref = blob
        else:
            assert blob["window_ids"] == ref["window_ids"]
            assert np.array_equal(blob["fold_of"], ref["fold_of"])
        probs.append(prob)
    pred = np.mean(probs, axis=0).argmax(axis=1)
    true = np.asarray(ref["oof_true"])
    fold_of = np.asarray(ref["fold_of"])
    rows = []
    for f in sorted(set(fold_of.tolist())):
        if f < 0:
            continue
        m = fold_of == f
        row = metrics_mod.compute_metrics(true[m], pred[m], NUM_CLASSES)
        row["accuracy"] = float((pred[m] == true[m]).mean())
        rows.append(row)
    return rows


def arm_stats(specs):
    """{metric: (mean over seeds, sd over seed means, sd over all folds)}."""
    per_seed, per_fold = {k: [] for k in KEYS}, {k: [] for k in KEYS}
    for k in ("f1_distracted", "prec_distracted", "rec_distracted", "f1_focused"):
        per_seed[k], per_fold[k] = [], []
    for seed in SEEDS:
        rows = fold_rows(specs, seed)
        if rows is None:
            return None
        def col(fn):
            return [fn(r) for r in rows]
        vals = {k: col(lambda r, k=k: r[k]) for k in KEYS}
        vals["f1_distracted"] = col(lambda r: r["per_class_f1"][1])
        vals["prec_distracted"] = col(lambda r: r["per_class_precision"][1])
        vals["rec_distracted"] = col(lambda r: r["per_class_recall"][1])
        vals["f1_focused"] = col(lambda r: r["per_class_f1"][0])
        for k, v in vals.items():
            per_seed[k].append(float(np.mean(v)))
            per_fold[k].extend(v)
    return {k: (float(np.mean(per_seed[k])), float(np.std(per_seed[k], ddof=1)),
                float(np.std(per_fold[k], ddof=1)), np.array(per_seed[k]))
            for k in per_seed}


def main():
    table = {}
    for name, specs in ARMS:
        st = arm_stats(specs)
        if st is not None:
            table[name] = st

    cols = ["macro_f1", "accuracy", "balanced_acc", "mcc",
            "f1_distracted", "prec_distracted", "rec_distracted", "f1_focused"]
    head = ["macro F1", "acc", "bal acc", "MCC",
            "F1 dist", "P dist", "R dist", "F1 foc"]

    print("\nMean over seeds 51-60 (per-fold metrics, averaged within seed)\n")
    print(f"{'arm':<20}" + "".join(f"{h:>10}" for h in head))
    print("-" * (20 + 10 * len(head)))
    for name in table:
        print(f"{name:<20}" + "".join(f"{table[name][c][0]:>10.4f}" for c in cols))

    print(f"\n{'arm':<20}" + "".join(f"{h:>10}" for h in head) + "   <- sd(seed)")
    print("-" * (20 + 10 * len(head)))
    for name in table:
        print(f"{name:<20}" + "".join(f"{table[name][c][1]:>10.4f}" for c in cols))

    print(f"\n{'arm':<20}" + "".join(f"{h:>10}" for h in head) + "   <- sd(fold)")
    print("-" * (20 + 10 * len(head)))
    for name in table:
        print(f"{name:<20}" + "".join(f"{table[name][c][2]:>10.4f}" for c in cols))

    pairs = [("C (ST-GCN-17)", "E1 (ST-GCN-13)"),
             ("C (ST-GCN-17)", "E5 (readout-13)"),
             ("E5 (readout-13)", "E1 (ST-GCN-13)"),
             ("C (ST-GCN-17)", "E6 (input-mask-13)"),
             ("E6 (input-mask-13)", "E1 (ST-GCN-13)"),
             ("C (ST-GCN-17)", "E7 (joint-bn-17)"),
             ("E7 (joint-bn-17)", "E1 (ST-GCN-13)"),
             ("E7 (joint-bn-17)", "E6 (input-mask-13)"),
             ("E1 (ST-GCN-13)", "E2 (weighted-17)"),
             ("E1 (ST-GCN-13)", "E3 (bone-13)"),
             ("E1 (ST-GCN-13)", "E1+E3 2-stream"),
             ("C+mc (rung D)", "E1+mc"),
             ("C+mc (rung D)", "E5+mc"),
             ("E5+mc", "E1+mc"),
             ("C+mc (rung D)", "E6+mc"),
             ("E6+mc", "E1+mc"),
             ("C+mc (rung D)", "E7+mc"),
             ("E7+mc", "E1+mc"),
             ("MLP baseline", "E7+mc"),
             ("E1+mc", "E3+mc"),
             ("E1+mc", "E1+E3 2-stream+mc"),
             ("MLP baseline", "E1+mc"),
             ("MLP baseline", "E1+E3 2-stream+mc"),
             ("C+mc & MLP", "E1+mc & MLP"),
             ("E1+mc & MLP", "2-stream+mc & MLP"),
             ("E1+mc x2 seeds", "E1+E3 2-stream+mc"),
             ("MLP x2 seeds", "E1+mc x2 seeds"),
             ("MLP baseline", "E1+mc & MLP"),
             ("MLP x2 seeds", "E1+mc & MLP"),
             ("E1+mc x2 seeds", "E1+mc & MLP")]
    print("\n\nPaired differences (second arm minus first), over 10 seeds\n")
    print(f"{'comparison':<42}{'metric':>16}{'diff':>10}{'sd':>9}"
          f"{'won':>7}{'p':>9}{'d':>7}")
    print("-" * 100)
    for a, b in pairs:
        if a not in table or b not in table:
            continue
        for c, h in zip(cols, head):
            da, db = table[a][c][3], table[b][c][3]
            diff = db - da
            t = stats.ttest_rel(db, da)
            d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else float("nan")
            print(f"{a + ' -> ' + b:<42}{h:>16}{diff.mean():>+10.4f}"
                  f"{diff.std(ddof=1):>9.4f}{int((diff > 0).sum()):>5}/10"
                  f"{t.pvalue:>9.4f}{d:>7.2f}")
        print()


if __name__ == "__main__":
    main()

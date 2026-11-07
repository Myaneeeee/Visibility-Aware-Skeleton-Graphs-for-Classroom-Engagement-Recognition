"""Paired comparison of an E rung against config C, over identical folds.

The E ladder is post-hoc (PREREGISTRATION.md Sec. 5b.8), so it stays out of make_report.py's
pre-registered tables and is read here instead. Pairing is by (seed, fold): both
runs use the same GroupKFold split by source video, so a difference is the
topology and not the partition. The script asserts that alignment rather than
assuming it.

The deterministic head and the MC-dropout head are reported separately -- the
latter is rung D, the best ST-GCN the study has. Ensembles are formed by
averaging saved out-of-fold probabilities, per Sec. 3.6: no retraining, no
free parameters, nothing fitted on the evaluation set.

Usage:  .venv/bin/python analysis/e_ladder_compare.py [rung]      (default: E1)
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


def load(label, model="stgcn"):
    path = paths.oof_probs_file(label, model)
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _scores(prob, true, fold_of):
    pred = np.asarray(prob).argmax(axis=1)
    return [metrics_mod.compute_metrics(true[fold_of == f], pred[fold_of == f],
                                        NUM_CLASSES)["macro_f1"]
            for f in sorted(set(fold_of.tolist())) if f >= 0]


def series(specs, seed):
    """Per-fold macro-F1 at one seed for a member, or the average of members.

    specs: [(label, model, key), ...]. More than one entry means the probability
    average -- the settled ensemble rule.
    """
    probs, ref = [], None
    for label, model, key in specs:
        blob = load(f"{label}_seed{seed}", model)
        if blob is None:
            return None
        prob = np.asarray(blob[key], dtype=np.float64)
        if not np.isfinite(prob).all():
            return None
        if ref is None:
            ref = blob
        else:
            assert blob["window_ids"] == ref["window_ids"], "window order differs"
            assert np.array_equal(blob["fold_of"], ref["fold_of"]), "folds differ"
        probs.append(prob)
    return _scores(np.mean(probs, axis=0), np.asarray(ref["oof_true"]),
                   np.asarray(ref["fold_of"]))


def paired(a_specs, b_specs):
    rows = []
    for seed in SEEDS:
        a, b = series(a_specs, seed), series(b_specs, seed)
        if a is None or b is None:
            continue
        rows.append((seed, float(np.mean(a)), float(np.mean(b)),
                     np.array(b) - np.array(a)))
    return rows


def report(title, a_name, a_specs, b_name, b_specs, per_seed=False):
    rows = paired(a_specs, b_specs)
    if not rows:
        print(f"\n=== {title} ===\n  no paired runs on disk")
        return
    am = np.array([r[1] for r in rows])
    bm = np.array([r[2] for r in rows])
    diffs = bm - am
    fold_diffs = np.concatenate([r[3] for r in rows])
    t = stats.ttest_rel(bm, am)
    w = stats.wilcoxon(bm, am) if len(rows) >= 6 else None

    print(f"\n=== {title} ===")
    print(f"  {a_name:<22}: {am.mean():.4f}  sd(seed) {am.std(ddof=1):.4f}")
    print(f"  {b_name:<22}: {bm.mean():.4f}  sd(seed) {bm.std(ddof=1):.4f}")
    print(f"  paired difference     : {diffs.mean():+.4f}  sd {diffs.std(ddof=1):.4f}"
          f"   ({len(rows)} seeds)")
    print(f"  seeds won              : {int((diffs > 0).sum())}/{len(diffs)}"
          f"      folds won: {int((fold_diffs > 0).sum())}/{len(fold_diffs)}")
    print(f"  paired t over seeds    : t={t.statistic:+.3f}  p={t.pvalue:.4f}"
          + (f"   |  Wilcoxon p={w.pvalue:.4f}" if w is not None else ""))
    if per_seed:
        print("   seed   " + f"{a_name[:8]:>8} {b_name[:8]:>8}     diff")
        for seed, a, b, _ in rows:
            print(f"   {seed}  {a:8.4f} {b:8.4f}  {b - a:+8.4f}")


if __name__ == "__main__":
    rung = sys.argv[1] if len(sys.argv) > 1 else "E1"
    C_ST = [("C", "stgcn", "oof_prob")]
    C_MC = [("C", "stgcn", "mc_prob")]
    E_ST = [(rung, "stgcn", "oof_prob")]
    E_MC = [(rung, "stgcn", "mc_prob")]
    MLP = [("C", "mlp", "oof_prob")]

    print("#" * 62)
    print("#  RQ2 -- does the observability-adapted graph help the ST-GCN?")
    print("#" * 62)
    report(f"{rung} vs C, deterministic head", "C", C_ST, rung, E_ST, per_seed=True)
    report(f"{rung} vs C, MC-dropout head (rung D)", "C+mc", C_MC, f"{rung}+mc", E_MC,
           per_seed=True)

    print("\n" + "#" * 62)
    print("#  RQ3 -- where does it now stand against the baselines?")
    print("#" * 62)
    report(f"{rung} vs the MLP baseline", "MLP", MLP, rung, E_ST)
    report(f"{rung}+mc vs the MLP baseline", "MLP", MLP, f"{rung}+mc", E_MC)
    report("C+mc vs the MLP baseline (the old gap)", "MLP", MLP, "C+mc", C_MC)

    print("\n" + "#" * 62)
    print("#  E4 -- does the ensemble survive a better member?")
    print("#" * 62)
    report("ensemble member swap", "C+mc & MLP", C_MC + MLP,
           f"{rung}+mc & MLP", E_MC + MLP)
    report("ensemble vs its best member", "MLP", MLP,
           f"{rung}+mc & MLP", E_MC + MLP)

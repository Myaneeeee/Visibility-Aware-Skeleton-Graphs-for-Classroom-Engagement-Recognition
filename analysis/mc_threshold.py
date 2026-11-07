"""Is MC dropout only a decision-threshold shift?

For each ST-GCN rung, four arms on the saved out-of-fold predictions:

- plain:        argmax of the deterministic probabilities (threshold 0.5)
- plain + thr:  the same probabilities, threshold on P(Distracted) tuned for macro-F1
- mc:           argmax of the 20-pass MC-dropout probabilities
- mc + thr:     MC-dropout probabilities with a tuned threshold

Thresholds are fitted leave-one-fold-out: the fold being scored contributes no labels
to its own threshold. AUROC needs no threshold, so it shows whether MC dropout ranks
windows better or only moves the operating point.

Usage:  .venv/bin/python analysis/mc_threshold.py [--draws 2000]
"""
import argparse
import os
import sys

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path[:0] = [os.path.dirname(__file__), "lib", "steps"]
import metrics as metrics_mod  # noqa: E402
from e_ladder_table import SEEDS  # noqa: E402
from paired_ci import prepare, print_pairs, probabilities  # noqa: E402

RUNGS = ["C", "E1", "E6"]
GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def macro_f1_per_threshold(p_dist, true):
    """Binary macro-F1 at every GRID threshold at once (zero_division=0, as sklearn)."""
    pos = p_dist[None, :] >= GRID[:, None]
    t1 = (true == 1)[None, :]
    tp, fp = (pos & t1).sum(1), (pos & ~t1).sum(1)
    fn, tn = (~pos & t1).sum(1), (~pos & ~t1).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1_pos = np.nan_to_num(2 * tp / (2 * tp + fp + fn))
        f1_neg = np.nan_to_num(2 * tn / (2 * tn + fp + fn))
    return (f1_pos + f1_neg) / 2


def best_threshold(p_dist, true):
    """Threshold on P(Distracted) with the highest pooled macro-F1; ties go to the lowest."""
    return float(GRID[int(np.argmax(macro_f1_per_threshold(p_dist, true)))])


def lofo_thresholds(p_dist, true, fold_of):
    """{fold: threshold fitted on every other fold}."""
    out = {}
    for f in sorted(set(fold_of.tolist())):
        if f < 0:
            continue
        train = (fold_of != f) & (fold_of >= 0)
        out[f] = best_threshold(p_dist[train], true[train])
    return out


def thresholded(p_dist, fold_of, thresholds):
    pred = np.zeros(len(p_dist), dtype=int)
    for f, t in thresholds.items():
        m = fold_of == f
        pred[m] = (p_dist[m] >= t).astype(int)
    return pred


def arm(rung, key, tuned):
    return (rung, "stgcn", key, tuned)


def predict(spec, seed):
    """paired_ci predictor for one (rung, model, key, tuned) arm."""
    rung, model, key, tuned = spec
    got = probabilities([(rung, model, key)], seed)
    if got is None:
        return None
    prob, true, fold_of, ids = got
    if not tuned:
        return prob.argmax(1), true, fold_of, ids
    th = lofo_thresholds(prob[:, 1], true, fold_of)
    return thresholded(prob[:, 1], fold_of, th), true, fold_of, ids


def operating_point(spec):
    """Mean over seeds of per-fold Distracted precision, recall, AUROC, and the mean threshold."""
    prec, rec, auc, thr = [], [], [], []
    for seed in SEEDS:
        got = predict(spec, seed)
        if got is None:
            return None
        pred, true, fold_of, _ = got
        prob = probabilities([spec[:3]], seed)[0][:, 1]
        folds = [f for f in sorted(set(fold_of.tolist())) if f >= 0]
        rows = [metrics_mod.compute_metrics(true[fold_of == f], pred[fold_of == f], 2)
                for f in folds]
        prec.append(np.mean([r["per_class_precision"][1] for r in rows]))
        rec.append(np.mean([r["per_class_recall"][1] for r in rows]))
        auc.append(np.mean([roc_auc_score(true[fold_of == f], prob[fold_of == f])
                            for f in folds]))
        thr.append(np.mean(list(lofo_thresholds(prob, true, fold_of).values()))
                   if spec[3] else 0.5)
    return np.mean(prec), np.mean(rec), np.array(auc), np.mean(thr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    names = {}
    for rung in RUNGS:
        names[rung] = arm(rung, "oof_prob", False)
        names[f"{rung} + thr"] = arm(rung, "oof_prob", True)
        names[f"{rung}+mc"] = arm(rung, "mc_prob", False)
        names[f"{rung}+mc + thr"] = arm(rung, "mc_prob", True)

    print("Operating points, mean over seeds 51-60 (per-fold, Distracted positive)\n")
    print(f"{'arm':<16}{'threshold':>10}{'P dist':>9}{'R dist':>9}{'AUROC':>9}")
    print("-" * 53)
    aucs = {}
    for name, spec in names.items():
        op = operating_point(spec)
        if op is None:
            print(f"{name:<16}   (runs missing)")
            continue
        prec, rec, auc, thr = op
        aucs[name] = auc
        print(f"{name:<16}{thr:>10.3f}{prec:>9.4f}{rec:>9.4f}{auc.mean():>9.4f}")

    print("\nAUROC paired over seeds (threshold-free: ranking quality only)\n")
    for rung in RUNGS:
        a, b = rung, f"{rung}+mc"
        if a in aucs and b in aucs:
            d = aucs[b] - aucs[a]
            p = stats.ttest_rel(aucs[b], aucs[a]).pvalue
            print(f"{a + ' -> ' + b:<20}{d.mean():>+9.4f}{int((d > 0).sum()):>4}/10{p:>9.4f}")

    point, boot = prepare(names, args.draws, rng, predict=predict)
    sections = []
    for rung in RUNGS:
        sections.append((f"{rung}: does MC dropout add anything a threshold does not?", [
            (rung, f"{rung}+mc"),
            (rung, f"{rung} + thr"),
            (f"{rung} + thr", f"{rung}+mc"),
            (f"{rung} + thr", f"{rung}+mc + thr"),
        ]))
    print(f"\nMacro-F1 paired differences, seeds 51-60; video bootstrap {args.draws} draws")
    print_pairs(point, boot, sections, rng, width=32)


if __name__ == "__main__":
    main()

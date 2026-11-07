"""Bootstrap 95% CIs for the paper's paired comparisons (PREREGISTRATION.md §5b.4).

Two intervals per comparison, both on the table's estimator (metric per fold,
averaged within a seed, then over seeds 51-60):

- seed bootstrap: resamples the 10 per-seed differences. Training noise only;
  the same variation the paired t-test measures.
- video bootstrap: resamples source videos with replacement within each fold,
  the same draw for both arms. Adds the uncertainty of which classrooms were
  sampled, so it is the wider and the one quoted in the paper.

The t-interval is printed beside them for reference.

Usage:  .venv/bin/python analysis/paired_ci.py [--draws 2000]
"""
import argparse
import os
import sys

import numpy as np
from scipy import stats

sys.path[:0] = [os.path.dirname(__file__), "lib", "steps"]
from e_ladder_table import SEEDS, fold_rows, load  # noqa: E402

ARM = {
    "A": [("A", "stgcn", "oof_prob")],
    "B": [("B", "stgcn", "oof_prob")],
    "C": [("C", "stgcn", "oof_prob")],
    "C+mc": [("C", "stgcn", "mc_prob")],
    "E1": [("E1", "stgcn", "oof_prob")],
    "E1+mc": [("E1", "stgcn", "mc_prob")],
    "E2": [("E2", "stgcn", "oof_prob")],
    "E3": [("E3", "stgcn", "oof_prob")],
    "E5": [("E5", "stgcn", "oof_prob")],
    "E6": [("E6", "stgcn", "oof_prob")],
    "E7": [("E7", "stgcn", "oof_prob")],
    "E7+mc": [("E7", "stgcn", "mc_prob")],
    "MLP": [("C", "mlp", "oof_prob")],
    "MLP-13": [("R-marg65", "mlp", "oof_prob")],
    "MLP-17": [("R-marg85", "mlp", "oof_prob")],
    "MLP x2": [("C", "mlp", "oof_prob"), ("C", "mlp", "oof_prob", 1)],
    "E1+mc x2": [("E1", "stgcn", "mc_prob"), ("E1", "stgcn", "mc_prob", 1)],
    "E1+mc & MLP": [("E1", "stgcn", "mc_prob"), ("C", "mlp", "oof_prob")],
    "E1 & E3 +mc": [("E1", "stgcn", "mc_prob"), ("E3", "stgcn", "mc_prob")],
}

PAIRS = [
    ("incremental ablation", [("A", "B"), ("B", "C"), ("C", "C+mc")]),
    ("the fix and the asymmetry", [("C", "E1"), ("MLP-17", "MLP-13")]),
    ("diagnostic ablation", [("C", "E2"), ("C", "E5"), ("C", "E6"), ("E6", "E1"),
                             ("C", "E7"), ("E7", "E1"), ("E1", "E3")]),
    ("against the baselines", [("MLP", "E1+mc"), ("C+mc", "E1+mc"),
                               ("MLP", "E7+mc"), ("MLP", "E1+mc & MLP"),
                               ("MLP x2", "E1+mc & MLP"),
                               ("E1+mc x2", "E1+mc & MLP"),
                               ("E1+mc x2", "E1 & E3 +mc")]),
]


def probabilities(specs, seed):
    """(averaged prob, true, fold_of, window_ids) for one arm at one seed, or None."""
    probs, ref = [], None
    for label, model, key, *offset in specs:
        shift = SEEDS.index(seed) + (offset[0] if offset else 0)
        blob = load(f"{label}_seed{SEEDS[shift % len(SEEDS)]}", model)
        if blob is None:
            return None
        probs.append(np.asarray(blob[key], dtype=np.float64))
        ref = ref or blob
    return (np.mean(probs, axis=0), np.asarray(ref["oof_true"]),
            np.asarray(ref["fold_of"]), list(ref["window_ids"]))


def predictions(specs, seed):
    """(pred, true, fold_of, window_ids) for one arm at one seed, or None."""
    got = probabilities(specs, seed)
    if got is None:
        return None
    prob, *rest = got
    return (prob.argmax(1), *rest)


def counts(specs, predict=predictions):
    """Confusion counts (seed, video, [tp, fp, fn, tn]) with Distracted positive."""
    out, index = [], None
    for seed in SEEDS:
        got = predict(specs, seed)
        if got is None:
            return None
        pred, true, fold_of, ids = got
        videos = np.array(["_".join(w.split("_")[:2]) for w in ids])
        if index is None:
            names = sorted(set(videos))
            fold = {v: int(fold_of[videos == v][0]) for v in names}
            index = (names, np.array([fold[v] for v in names]),
                     {v: i for i, v in enumerate(names)})
        vid = np.array([index[2][v] for v in videos])
        c = np.zeros((len(index[0]), 4))
        for k, (p, t) in enumerate([(1, 1), (1, 0), (0, 1), (0, 0)]):
            np.add.at(c[:, k], vid, (pred == p) & (true == t))
        out.append(c)
    return np.stack(out), index[1]


def macro_f1(c):
    """Macro-F1 from counts summed over videos: c[..., 4] -> (...)."""
    tp, fp, fn, tn = np.moveaxis(c, -1, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1_pos = np.nan_to_num(2 * tp / (2 * tp + fp + fn))
        f1_neg = np.nan_to_num(2 * tn / (2 * tn + fp + fn))
    return (f1_pos + f1_neg) / 2


def score(c, fold_of_video, weights):
    """Per-seed macro-F1: per fold under video weights, averaged over folds."""
    per_fold = []
    for f in np.unique(fold_of_video):
        m = fold_of_video == f
        per_fold.append(macro_f1(np.einsum("bv,svk->bsk", weights[:, m], c[:, m])))
    return np.mean(per_fold, axis=0)                  # (draws, seeds)


def video_weights(fold_of_video, draws, rng):
    w = np.zeros((draws, len(fold_of_video)))
    for f in np.unique(fold_of_video):
        idx = np.flatnonzero(fold_of_video == f)
        pick = rng.choice(idx, size=(draws, len(idx)), replace=True)
        for d in range(draws):
            np.add.at(w[d], pick[d], 1)
    return w


def prepare(arms, draws, rng, predict=predictions):
    """Per-seed macro-F1 (point) and its video-bootstrap draws (boot) for every arm."""
    cache = {name: counts(specs, predict) for name, specs in arms.items()}
    cache = {k: v for k, v in cache.items() if v is not None}
    fold_of_video = next(iter(cache.values()))[1]
    for c, f in cache.values():
        assert np.array_equal(f, fold_of_video), "arms disagree on folds"
    ones = np.ones((1, len(fold_of_video)))
    point = {k: score(c, fold_of_video, ones)[0] for k, (c, _) in cache.items()}
    w = video_weights(fold_of_video, draws, rng)
    boot = {k: score(c, fold_of_video, w) for k, (c, _) in cache.items()}
    return point, boot


def print_pairs(point, boot, sections, rng, width=30):
    """One row per comparison: diff, seeds won, paired p, and three 95% CIs."""
    head = (f"{'comparison':<{width}}{'diff':>9}{'won':>7}{'p':>9}"
            f"{'seed-boot 95% CI':>22}{'video-boot 95% CI':>22}{'t 95% CI':>22}")
    for section, pairs in sections:
        print(f"\n## {section}\n{head}\n" + "-" * len(head))
        for a, b in pairs:
            if a not in point or b not in point:
                print(f"{a + ' -> ' + b:<{width}}   (runs missing)")
                continue
            d = point[b] - point[a]
            sb = rng.choice(d, size=(10000, len(d))).mean(1)
            vb = (boot[b] - boot[a]).mean(1)
            half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            p = stats.ttest_rel(point[b], point[a]).pvalue
            ci = lambda lo, hi: f"[{lo:+.4f}, {hi:+.4f}]"   # noqa: E731
            print(f"{a + ' -> ' + b:<{width}}{d.mean():>+9.4f}{int((d > 0).sum()):>4}/10"
                  f"{p:>9.4f}{ci(*np.percentile(sb, [2.5, 97.5])):>22}"
                  f"{ci(*np.percentile(vb, [2.5, 97.5])):>22}"
                  f"{ci(d.mean() - half, d.mean() + half):>22}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    point, boot = prepare(ARM, args.draws, rng)
    for k in ("C", "E1", "MLP"):
        rows = [np.mean([r["macro_f1"] for r in fold_rows(ARM[k], s)]) for s in SEEDS]
        assert np.allclose(point[k], rows), f"{k}: estimator differs from the table"

    print(f"Macro-F1 paired differences, seeds 51-60; video bootstrap {args.draws} draws\n")
    print_pairs(point, boot, PAIRS, rng)


if __name__ == "__main__":
    main()

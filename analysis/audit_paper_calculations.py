"""Recompute paper results from saved probabilities and export the arithmetic.

Run from the repository root:
    .venv/bin/python analysis/audit_paper_calculations.py --draws 2000

No training. Inputs are read-only. Writes CSV/JSON under outputs/audit/ by default.
The confusion-count calculation is checked against sklearn and the existing table
code; model/window/label/fold alignment is checked before averaging probabilities.
"""
import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
import sklearn
import torch
from scipy import stats
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "analysis"), str(ROOT / "lib"), str(ROOT / "steps")]
from e_ladder_table import SEEDS, arm_stats  # noqa: E402
from make_paper_tables import COMPARISONS, DIAGNOSTIC, MODELS  # noqa: E402
from paired_ci import ARM  # noqa: E402


def macro_f1(c):
    tp, fp, fn, tn = np.moveaxis(c, -1, 0)
    positive = np.divide(2 * tp, 2 * tp + fp + fn,
                         out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) != 0)
    negative = np.divide(2 * tn, 2 * tn + fp + fn,
                         out=np.zeros_like(tn, dtype=float), where=(2 * tn + fp + fn) != 0)
    return (positive + negative) / 2


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "audit")
    args = ap.parse_args()
    if args.draws < 2:
        ap.error("--draws must be at least 2")
    args.out.mkdir(parents=True, exist_ok=True)

    pairs = [("diagnostic", "C", k) for _, _, k in DIAGNOSTIC if k != "C"]
    pairs += [("comparison", a, b) for _, a, b in COMPARISONS]
    pairs += [("incremental", a, b) for a, b in [("A", "B"), ("B", "C"), ("C", "C+mc")]]
    arms = sorted({k for _, k in MODELS if k} | {k for _, a, b in pairs for k in (a, b)})
    blobs, manifest = {}, []
    reference = None

    def load(config, model, seed):
        nonlocal reference
        path = ROOT / "outputs" / "runs" / f"{config}_seed{seed}" / model / "oof_probs.pt"
        if path not in blobs:
            blob = torch.load(path, map_location="cpu", weights_only=False)
            ids = list(blob["window_ids"])
            true = np.asarray(blob["oof_true"])
            folds = np.asarray(blob["fold_of"])
            if reference is None:
                reference = (ids, true, folds)
            assert ids == reference[0], f"window order differs: {path}"
            assert np.array_equal(true, reference[1]), f"labels differ: {path}"
            assert np.array_equal(folds, reference[2]), f"folds differ: {path}"
            assert len(set(ids)) == len(ids)
            assert set(true) == {0, 1} and set(folds) == set(range(5))
            manifest.append({"path": str(path.relative_to(ROOT)),
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            blobs[path] = blob
        return blobs[path]

    probs = {}
    for arm in arms:
        per_seed = []
        for i, seed in enumerate(SEEDS):
            members = []
            for config, model, key, *offset in ARM[arm]:
                member_seed = SEEDS[(i + (offset[0] if offset else 0)) % len(SEEDS)]
                p = np.asarray(load(config, model, member_seed)[key], dtype=np.float64)
                assert p.shape == (len(reference[0]), 2) and np.isfinite(p).all()
                assert (p >= 0).all() and np.allclose(p.sum(1), 1, atol=1e-5)
                members.append(p)
            per_seed.append(np.mean(members, axis=0))
        probs[arm] = np.stack(per_seed)

    ids, true, fold_of = reference
    annotations = {r["window_id"]: r["label"]
                   for r in csv.DictReader((ROOT / "datasets" / "annotations.csv").open())}
    expected = {w: int(label == "Distracted") for w, label in annotations.items()
                if label in ("Focused", "Distracted")}
    assert dict(zip(ids, true.tolist())) == expected, "predictions differ from final labels"
    source = np.array(["_".join(w.split("_")[:2]) for w in ids])
    names, video_idx = np.unique(source, return_inverse=True)
    source_folds = np.array([fold_of[source == v][0] for v in names])
    assert len(names) == 197
    assert all(len(set(fold_of[source == v])) == 1 for v in names)

    counts, point, fold_rows = {}, {}, []
    for arm, p in probs.items():
        pred = p.argmax(2)
        c = np.zeros((len(SEEDS), len(names), 4), dtype=np.int64)
        for s in range(len(SEEDS)):
            for k, (guess, label) in enumerate([(1, 1), (1, 0), (0, 1), (0, 0)]):
                c[s, :, k] = np.bincount(video_idx, weights=(pred[s] == guess) & (true == label),
                                         minlength=len(names))
        counts[arm] = c
        scores = []
        for f in range(5):
            summed = c[:, source_folds == f].sum(1)
            values = macro_f1(summed)
            for s, seed in enumerate(SEEDS):
                m = fold_of == f
                assert np.isclose(values[s], f1_score(true[m], pred[s, m], average="macro"))
                fold_rows.append({"arm": arm, "seed": seed, "fold": f,
                                  "windows": int(m.sum()), "source_videos": int((source_folds == f).sum()),
                                  "tp": int(summed[s, 0]), "fp": int(summed[s, 1]),
                                  "fn": int(summed[s, 2]), "tn": int(summed[s, 3]),
                                  "macro_f1": float(values[s])})
            scores.append(values)
        point[arm] = np.mean(scores, axis=0)
        assert np.allclose(point[arm], arm_stats(ARM[arm])["macro_f1"][3]), arm

    # Same RNG, source ordering, fold ordering and draw shape as paired_ci.py.
    rng = np.random.default_rng(0)
    weights = np.zeros((args.draws, len(names)), dtype=int)
    for f in range(5):
        idx = np.flatnonzero(source_folds == f)
        picks = rng.choice(idx, size=(args.draws, len(idx)), replace=True)
        for draw in range(args.draws):
            np.add.at(weights[draw], picks[draw], 1)
    bootstrap = {}
    for arm, c in counts.items():
        bootstrap[arm] = np.mean([
            macro_f1(np.einsum("bv,svk->bsk", weights[:, source_folds == f], c[:, source_folds == f]))
            for f in range(5)], axis=0).mean(1)

    comparisons, seed_rows, draw_rows = [], [], []
    for section, a, b in pairs:
        d = point[b] - point[a]
        sd = d.std(ddof=1)
        se = sd / np.sqrt(len(SEEDS))
        t = d.mean() / se
        p = 2 * stats.t.sf(abs(t), df=len(SEEDS) - 1)
        assert np.isclose(p, stats.ttest_rel(point[b], point[a]).pvalue)
        half = stats.t.ppf(0.975, len(SEEDS) - 1) * se
        boot_delta = bootstrap[b] - bootstrap[a]
        lo, hi = np.percentile(boot_delta, [2.5, 97.5], method="linear")
        comparison = f"{a} -> {b}"
        comparisons.append({"section": section, "comparison": comparison,
                            "mean_a": float(point[a].mean()), "mean_b": float(point[b].mean()),
                            "delta": float(d.mean()), "seed_sd_delta": float(sd), "seed_se_delta": float(se),
                            "t": float(t), "df": len(SEEDS) - 1, "paired_seed_p": float(p),
                            "seed_t_ci_low": float(d.mean() - half), "seed_t_ci_high": float(d.mean() + half),
                            "video_bootstrap_ci_low": float(lo), "video_bootstrap_ci_high": float(hi),
                            "seeds_better": int((d > 0).sum()),
                            "bootstrap_draws_at_or_below_zero": int((boot_delta <= 0).sum())})
        for i, seed in enumerate(SEEDS):
            seed_rows.append({"comparison": comparison, "seed": seed,
                              "next_seed": SEEDS[(i + 1) % len(SEEDS)],
                              "macro_f1_a": float(point[a][i]), "macro_f1_b": float(point[b][i]),
                              "delta": float(d[i])})
        draw_rows.extend({"comparison": comparison, "draw": i, "delta": float(value)}
                         for i, value in enumerate(boot_delta))

    write_csv(args.out / "fold_calculations.csv", fold_rows)
    write_csv(args.out / "seed_calculations.csv", seed_rows)
    write_csv(args.out / "comparisons.csv", comparisons)
    write_csv(args.out / "video_bootstrap_draws.csv", draw_rows)
    write_csv(args.out / "source_confusion_counts.csv", [
        {"arm": arm, "seed": seed, "source_video": str(v), "fold": int(source_folds[i]),
         **dict(zip(["tp", "fp", "fn", "tn"], c[s, i].tolist()))}
        for arm, c in counts.items() for s, seed in enumerate(SEEDS) for i, v in enumerate(names)])
    report = {"draws": args.draws, "bootstrap_rng_seed": 0, "seeds": SEEDS,
              "windows": len(ids), "source_videos": len(names),
              "folds": [{"fold": f, "windows": int((fold_of == f).sum()),
                         "source_videos": int((source_folds == f).sum())} for f in range(5)],
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "scipy": scipy.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__},
              "comparisons": comparisons, "inputs": manifest}
    (args.out / "calculation_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    final = next(r for r in comparisons if r["comparison"] == "E1+mc x2 -> E1+mc & MLP")
    print(json.dumps({"output": str(args.out), "checks": "alignment, labels, grouping, sklearn F1, table seed scores, scipy p: passed",
                      "final_row": final}, indent=2))


if __name__ == "__main__":
    main()

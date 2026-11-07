"""The paper's four tables, generated from the data and saved runs.

  Table 1  dataset funnel and annotator agreement
  Table 2  diagnostic ablation: what each intervention changes, with its paired CI
  Table 3  main results: every model and ensemble, with the majority floor
  Table 4  the remaining paired comparisons: the asymmetry, parity, the ensemble

Names follow PAPER_DESIGN.md §3: no configuration letters. CIs are the
source-video bootstrap of analysis/paired_ci.py. Writes Markdown and CSV to
outputs/tables/, so nothing in a table is typed by hand.

Usage:  .venv/bin/python analysis/make_paper_tables.py [--draws 2000]
"""
import argparse
import csv
import glob
import itertools
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path[:0] = [os.path.dirname(__file__), "lib", "steps"]
import common  # noqa: E402
import metrics as metrics_mod  # noqa: E402
import paths  # noqa: E402
import sav_actions as sa  # noqa: E402
from agreement import BINARY, cohen_kappa, load_ledger  # noqa: E402
from annotation_report import prior_hit_rate  # noqa: E402
from e_ladder_table import SEEDS, arm_stats, load  # noqa: E402
from paired_ci import ARM, counts, score, video_weights  # noqa: E402

OUT = os.path.join(os.path.dirname(paths.figures_dir()), "tables")

MODELS = [  # (paper name, arm key or None for rows computed separately)
    ("majority (always on-task)", None),
    ("logistic regression, summary statistics", None),
    ("GRU", "GRU"),
    ("MLP, summary statistics", "MLP"),
    ("ST-GCN, from scratch", "A"),
    ("  + NTU-60 pretraining", "B"),
    ("  + augmentation (full skeleton)", "C"),
    ("  + MC dropout", "C+mc"),
    ("visible-joint ST-GCN", "E1"),
    ("  + MC dropout", "E1+mc"),
    ("MLP + visible-joint ST-GCN + MC dropout", "E1+mc & MLP"),
]
ARM["GRU"] = [("C", "gru", "oof_prob")]

DIAGNOSTIC = [  # (intervention, path it tests, arm)
    ("none: full skeleton (reference)", "—", "C"),
    ("confidence-weighted adjacency", "messages", "E2"),
    ("visible-joint pooling", "pooling", "E5"),
    ("input masking", "input", "E6"),
    ("per-joint input normalization", "input normalization design", "E7"),
    ("visible-joint graph (the fix)", "all three", "E1"),
]

COMPARISONS = [  # (label, from arm, to arm)
    ("full skeleton → visible-joint graph (MLP)", "MLP-17", "MLP-13"),
    ("full skeleton → visible-joint graph (ST-GCN, both + MC dropout)", "C+mc", "E1+mc"),
    ("input masking → visible-joint graph", "E6", "E1"),
    ("MLP → visible-joint ST-GCN + MC dropout", "MLP", "E1+mc"),
    ("MLP → MLP + visible-joint ST-GCN + MC", "MLP", "E1+mc & MLP"),
    ("MLP, two seeds → MLP + visible-joint ST-GCN + MC", "MLP x2", "E1+mc & MLP"),
    ("visible-joint ST-GCN + MC, two seeds → MLP + visible-joint ST-GCN + MC",
     "E1+mc x2", "E1+mc & MLP"),
]


def f4(x):
    return f"{x:.4f}".replace("-", "−")


def s4(x):
    return f"{x:+.4f}".replace("-", "−")


def pval(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def write(name, header, rows, note):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{name}.csv"), "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    with open(os.path.join(OUT, f"{name}.md"), "w") as f:
        f.write("\n".join(md) + f"\n\n{note}\n")
    print(f"\n## {name}\n" + "\n".join(md) + f"\n\n{note}")


def window_ids(window_dir):
    import torch
    return {common.make_window_id(w) for p in glob.glob(os.path.join(window_dir, "*_windows.pt"))
            for w in torch.load(p, weights_only=False)}


def table1():
    """Dataset counts: a label counts only if its window passed the quality filter."""
    stored = {r["window_id"]: r["label"] for r in csv.DictReader(open(paths.ANNOTATIONS_FILE))}
    extracted = window_ids(paths.SAV_WINDOWS_DIR)
    filtered = window_ids(paths.SAV_WINDOWS_FILTERED_DIR)
    excluded = common.load_excluded_ids()
    labels = {w: l for w, l in stored.items() if w in filtered and w not in excluded}
    video = lambda w: "_".join(w.split("_")[:2])          # noqa: E731
    trainable = {w: l for w, l in labels.items() if l in ("Focused", "Distracted")}
    clips = len(open(paths.SAV_CLIP_LIST_FILE).read().split())
    ann = os.path.join(paths.SAV_DIR, "annotations")
    actions = sa.load_clip_person_actions(
        [os.path.join(ann, "train.csv"), os.path.join(ann, "val.csv")],
        sa.load_label_map(os.path.join(ann, "education_first_label.pbtxt")))
    prior = prior_hit_rate(labels, actions)
    off = sum(1 for l in trainable.values() if l == "Distracted")
    rows = [
        ["source videos (lessons)", 200, ""],
        ["3-s classroom clips", clips, "80% targeted, 20% control"],
        ["per-student samples extracted", len(extracted), "tracked in ≥ 80% of frames"],
        ["samples after the quality filter", len(filtered), ""],
        ["  labeled", len(labels), "" if len(labels) == len(filtered) else f"{len(filtered) - len(labels)} kept out of training"],
        ["rejected by annotators", sum(1 for l in labels.values() if l == "TRASH"), ""],
        ["trainable samples", len(trainable), f"from {len({video(w) for w in trainable})} source videos"],
        ["  on-task", len(trainable) - off, ""],
        ["  off-task", off, f"{off / len(trainable):.1%}"],
        ["off-task rate, targeted sample", f"{prior['distracted_rate_in_prior']:.2%}", ""],
        ["off-task rate, control sample", f"{prior['distracted_rate_in_rest']:.2%}", ""],
    ]
    ledgers = {os.path.basename(p)[:-4]: load_ledger(p)
               for p in sorted(glob.glob(os.path.join(paths.SAV_LEDGERS_DIR, "annotator_*.csv")))}
    overlap = set.intersection(*(set(v) for v in ledgers.values()))
    for (a, la), (b, lb) in itertools.combinations(ledgers.items(), 2):
        k3, n3, _ = cohen_kappa(la, lb)
        k2, n2, _ = cohen_kappa(la, lb, restrict=BINARY)
        rows.append([f"κ, annotators {a[-1]}–{b[-1]}",
                     f"{k3:.3f} (3-class, n = {n3})", f"{k2:.3f} (on/off-task, n = {n2})"])
    write("table1_dataset", ["", "value", "note"], rows,
          f"Agreement on the blind overlap of {len(overlap)} samples placed in all three "
          "worklists, measured when the samples were labeled. 3-class: on-task, off-task, rejected. "
          "Final labels were reviewed jointly by the three annotators.")


def majority_row():
    blob = load(f"C_seed{SEEDS[0]}", "mlp")
    true, fold_of = np.asarray(blob["oof_true"]), np.asarray(blob["fold_of"])
    rows = []
    for f in np.unique(fold_of):
        m = fold_of == f
        pred = np.zeros(m.sum(), dtype=int)
        r = metrics_mod.compute_metrics(true[m], pred, 2)
        r["accuracy"] = float((pred == true[m]).mean())
        rows.append(r)
    mean = lambda k: float(np.mean([r[k] for r in rows]))   # noqa: E731
    return [f4(mean("macro_f1")), f4(mean("mcc")), f"{mean('accuracy'):.1%}", "0.0000",
            "0.0000", "—"]


def table3():
    rep = json.load(open(paths.representation_file()))
    lr = next(g for g in rep["feature_groups"] if g["n_features"] == 85)
    rows = []
    for name, key in MODELS:
        if key is None and name.startswith("majority"):
            rows.append([name] + majority_row())
        elif key is None:
            rows.append([name, f4(lr["macro_f1"]), f4(lr["mcc"]), "—", "—", "—", "—"])
        else:
            st = arm_stats(ARM[key])
            rows.append([name, f4(st["macro_f1"][0]), f4(st["mcc"][0]),
                         f"{st['accuracy'][0]:.1%}", f4(st["prec_distracted"][0]),
                         f4(st["rec_distracted"][0]), f4(st["macro_f1"][1])])
    write("table3_main_results",
          ["model", "macro-F1", "MCC", "accuracy", "off-task precision", "off-task recall",
           "seed SD (macro-F1)"], rows,
          "Mean over 10 seeds of per-fold metrics, 5-fold cross-validation grouped by source "
          "video. Logistic regression is deterministic, so it has no seed SD. Accuracy is "
          "shown beside the majority floor and is never the headline.")


def paired(cache, boot, a, b):
    d = cache[b] - cache[a]
    vb = (boot[b] - boot[a]).mean(1)
    lo, hi = np.percentile(vb, [2.5, 97.5])
    return (s4(d.mean()), f"[{s4(lo)}, {s4(hi)}]", f"{int((d > 0).sum())} / 10",
            pval(stats.ttest_rel(cache[b], cache[a]).pvalue))


def tables2_and_4(draws):
    needed = {k for _, _, k in DIAGNOSTIC} | {x for _, a, b in COMPARISONS for x in (a, b)}
    got = {k: counts(ARM[k]) for k in needed}
    fold_of_video = next(iter(got.values()))[1]
    ones = np.ones((1, len(fold_of_video)))
    cache = {k: score(c, fold_of_video, ones)[0] for k, (c, _) in got.items()}
    w = video_weights(fold_of_video, draws, np.random.default_rng(0))
    boot = {k: score(c, fold_of_video, w) for k, (c, _) in got.items()}

    rows = []
    for name, path, key in DIAGNOSTIC:
        if key == "C":
            rows.append([name, path, f4(cache[key].mean()), "—", "—", "—", "—"])
        else:
            rows.append([name, path, f4(cache[key].mean()), *paired(cache, boot, "C", key)])
    write("table2_diagnostic_ablation",
          ["intervention", "path tested", "macro-F1", "Δ vs full skeleton", "95% CI",
           "seeds better", "p"], rows,
          "Each row changes one thing in the full-skeleton ST-GCN (pretrained, augmented). "
          "CI: source-video bootstrap within folds, 2,000 draws. p: paired t-test over seeds.")

    rows = [[label, *paired(cache, boot, a, b)] for label, a, b in COMPARISONS]
    write("table4_paired_comparisons",
          ["comparison", "Δ macro-F1", "95% CI", "seeds better", "p"], rows,
          "Paired seed by seed on identical folds. CI: source-video bootstrap within folds. "
          "Two-seed ensembles pair seed s with s + 1, so their p is slightly optimistic.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    args = ap.parse_args()
    table1()
    tables2_and_4(args.draws)
    table3()
    print(f"\nwrote {os.path.relpath(OUT, paths.REPO_ROOT)}/")


if __name__ == "__main__":
    main()

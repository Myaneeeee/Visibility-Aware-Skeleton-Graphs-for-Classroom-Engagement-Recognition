"""Step: turn everything under outputs/runs/ into readable reports and figures.

Reads only saved artifacts -- it never trains anything, so it is safe to re-run
as often as you like and always reflects what is currently on disk.

Produces::

    outputs/README.md                  index: what exists, how it was made
    outputs/reports/01_headline.md     the numbers to quote, in one page
    outputs/reports/02_ablation.md     full A/B/C ladder + baselines, per fold
    outputs/reports/03_confidence.md   calibration + selective prediction
    outputs/reports/04_data_need.md    learning curves: does more data help?
    outputs/figures/*.png              confusion matrices, loss, reliability,
                                       risk-coverage

Every table states its protocol: 5-fold cross-validation grouped by source video.
The grouping unit is the source video, never a room -- SAV publishes no room
metadata.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import torch

import _bootstrap  # noqa: F401  -- puts ../lib on sys.path
import analyze_confidence
import calibration as cal
import common
import metrics as metrics_mod
import paths
from scipy import stats

MODELS = ["stgcn", "gru", "mlp"]
MODEL_LABEL = {"stgcn": "ST-GCN", "gru": "GRU baseline", "mlp": "MLP baseline"}
# Config D's MC-dropout half, treated as a rung in its own right: unlike the
# temperature half it changes the argmax, so it has discriminative numbers that
# are genuinely its own and must be compared under the same paired protocol.
MC_RUNG = "D+mc"
CONFIG_DESC = {
    "A": "ST-GCN from scratch",
    "B": "A + NTU-60 warm start",
    "C": "B + skeleton augmentation",
    MC_RUNG: "C + MC-dropout averaging (20 passes)",
}
# The pre-registered ladder (PREREGISTRATION.md Sec. 5b.3). Everything else found on
# disk -- currently the "+w"/"+f" imbalance arm -- is exploratory and must never
# be reported as a rung, least of all as "the best rung": Sec. 9.1 fixed the
# objective in advance, so a weighted run that scores higher has not beaten the
# ladder, it has answered a different question.
LADDER_CONFIGS = ("A", "B", "C", MC_RUNG)
LOSS_ARM_SUFFIX = {"+w": "class-weighted CE", "+f": "class-weighted focal (γ=2)"}


def loss_arm_of(config):
    """('C', '+w') for an imbalance-arm config, else (config, None)."""
    for suffix in LOSS_ARM_SUFFIX:
        if config.endswith(suffix):
            return config[: -len(suffix)], suffix
    return config, None


def best_ladder_config_per_model(seed_means):
    """{model: (config, mean macro-F1)} over the PRE-REGISTERED rungs only.

    Kept apart from build_ablation so the exclusion is testable: the headline
    "Best ST-GCN rung" is the study's central negative result, and an off-ladder
    arm scoring higher would otherwise rewrite it with nothing to flag it.
    """
    best = {}
    for (config, model), per_seed in seed_means.items():
        if config not in LADDER_CONFIGS:
            continue
        f1 = float(np.mean([v["macro_f1"] for v in per_seed.values()]))
        if model not in best or f1 > best[model][1]:
            best[model] = (config, f1)
    return best
# metrics.compute_metrics deliberately does not return accuracy: at this
# imbalance it is uninformative (always predicting the majority class already
# scores 1 - minority_rate). It is recomputed only where a table needs it
# beside the calibration numbers.
SCALARS = ["macro_f1", "balanced_acc", "mcc",
           "precision_distracted", "recall_distracted", "f1_distracted"]
# Index of the minority class in common.CLASS_NAMES.
DISTRACTED_IDX = 1


def minority_rate(runs):
    """Distracted fraction of the evaluated set, read from any saved run."""
    for path in runs.values():
        blob = torch.load(path, map_location="cpu", weights_only=False)
        true = np.asarray(blob["oof_true"])
        if true.size:
            return float((true == DISTRACTED_IDX).mean())
    return float("nan")


def majority_macro_f1(rate):
    """Macro-F1 of always predicting the majority class at this minority rate."""
    p = 1.0 - rate
    return (2 * p / (1 + p)) / 2 if 0 < p <= 1 else float("nan")


# ---------------------------------------------------------------- loading

def discover_runs():
    """{(config, seed, model): blob} for everything saved under outputs/runs/."""
    found = {}
    for path in sorted(glob.glob(os.path.join(paths.runs_dir(), "*", "*", "oof_probs.pt"))):
        label = os.path.basename(os.path.dirname(os.path.dirname(path)))
        model = os.path.basename(os.path.dirname(path))
        if "_seed" not in label:
            continue
        config, seed = label.split("_seed", 1)
        try:
            seed = int(seed)
        except ValueError:
            continue
        found[(config, seed, model)] = path
    return found


def per_fold_metrics(path, prob_key="oof_prob"):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    prob = np.asarray(blob[prob_key], dtype=np.float64)
    true = np.asarray(blob["oof_true"])
    if not np.isfinite(prob).all():
        return [], blob
    fold_of = np.asarray(blob["fold_of"])
    pred = prob.argmax(axis=1)
    rows = []
    for f in sorted(set(fold_of.tolist())):
        if f < 0:
            continue
        m = fold_of == f
        row = metrics_mod.compute_metrics(true[m], pred[m],
                                          len(common.CLASS_NAMES))
        # Flatten the minority class out of the per-class lists. Macro-F1 prices
        # precision and recall together, so a change that trades one for the other
        # nets to zero -- these two columns are what make that trade visible.
        row["precision_distracted"] = row["per_class_precision"][DISTRACTED_IDX]
        row["recall_distracted"] = row["per_class_recall"][DISTRACTED_IDX]
        row["f1_distracted"] = row["per_class_f1"][DISTRACTED_IDX]
        rows.append(row)
    return rows, blob


def selective_rows(prob, true, points=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)):
    """Accuracy-vs-coverage rows from a risk-coverage curve."""
    cov, risk = cal.risk_coverage_curve(prob, true)
    cov, risk = np.asarray(cov), np.asarray(risk)
    rows = []
    for t in points:
        j = int(np.argmin(np.abs(cov - t)))
        rows.append([f"{cov[j]:.0%}", int(round(cov[j] * len(true))),
                     f"{risk[j]:.1%}", f"{1 - risk[j]:.1%}"])
    return rows


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------- reports

def build_ablation(runs):
    """Seed-level means per (config, model), plus the paired B-vs-A style deltas."""
    seed_means = defaultdict(dict)      # (config, model) -> {seed: {metric: mean}}
    fold_vals = defaultdict(dict)       # (config, model) -> {seed: {metric: array}}
    for (c, s, m), path in runs.items():
        rows, _ = per_fold_metrics(path)
        if not rows:
            continue
        fold_vals[(c, m)][s] = {k: np.array([r[k] for r in rows]) for k in SCALARS}
        seed_means[(c, m)][s] = {k: float(np.mean([r[k] for r in rows]))
                                 for k in SCALARS}

    lines = ["# 02 — Ablation ladder", "",
             "**Protocol:** 5-fold cross-validation, folds grouped by *source "
             "video* (an unseen recording at test time). Every number below is "
             "out-of-fold.", "",
             "`sd(seed)` is training noise across seeds. `sd(fold)` is the "
             "spread across videos — the honest generalization error bar. "
             "A rung whose gain is smaller than sd(seed) is not a result.", "",
             "`P/R (Distr)` are precision and recall on the minority class. "
             "Macro-F1 prices both, so a change that trades one for the other "
             "nets to zero — these columns show the trade instead of hiding it.",
             ""]

    headers = ["config", "model", "seeds", "macro F1", "sd(seed)", "sd(fold)",
               "balanced acc", "MCC", "P (Distr)", "R (Distr)"]

    def _stats_row(label, model_key, per_seed, folds_by_seed):
        get = lambda k: np.array([v[k] for v in per_seed.values()])
        f1 = get("macro_f1")
        sdf = np.mean([np.std(folds_by_seed[s]["macro_f1"], ddof=1)
                       for s in per_seed])
        return [label, MODEL_LABEL[model_key], len(per_seed),
                f"{f1.mean():.4f}",
                f"{np.std(f1, ddof=1):.4f}" if len(f1) > 1 else "-",
                f"{sdf:.4f}",
                f"{get('balanced_acc').mean():.4f}",
                f"{get('mcc').mean():.4f}",
                f"{get('precision_distracted').mean():.4f}",
                f"{get('recall_distracted').mean():.4f}"]

    # The config letter only means something for the ST-GCN. evaluate.py hands the
    # warm start and the augmenter to the ST-GCN alone, so the MLP and GRU are
    # retrained identically in every config and their numbers are bit-identical
    # across A/B/C at a given seed. Printing them once per config invites the
    # reader to compare rows that cannot differ.
    baseline_rows, ladder_rows = [], []
    for (c, m) in sorted(seed_means, key=lambda k: (k[0], MODELS.index(k[1]))):
        if m == "stgcn":
            ladder_rows.append(_stats_row(c, m, seed_means[(c, m)],
                                          fold_vals[(c, m)]))
        elif c == "A":
            baseline_rows.append(_stats_row("—", m, seed_means[(c, m)],
                                            fold_vals[(c, m)]))

    if baseline_rows:
        lines += ["## Baselines", "",
                  "Config-independent: the MLP and GRU take neither the NTU warm "
                  "start nor the augmenter, so they are retrained identically in "
                  "every config and score identically at a given seed. They have "
                  "no rung — the config column is left blank on purpose.", "",
                  md_table(headers, baseline_rows), "",
                  "## ST-GCN ladder", "",
                  "Each rung adds exactly one component to the rung below it.", ""]
    rows = ladder_rows

    # PREREGISTRATION.md Sec. 5b.3 pre-commits to carrying D in this table with an
    # explicit footnote rather than as a duplicated row: D-temp's discriminative
    # metrics are C's by construction, so a silent duplicate row would read as
    # an independent result.
    def _row(label, per_seed, folds_by_seed):
        return _stats_row(label, "stgcn", per_seed, folds_by_seed)

    c_stgcn = seed_means.get(("C", "stgcn"))
    if c_stgcn:
        rows.append(_row("D +temp †", c_stgcn, fold_vals[("C", "stgcn")]))
    # D's OTHER half. MC-dropout averaging is not monotonic, so unlike +temp it
    # CAN move the argmax -- reporting only the +temp row would hide the one
    # part of config D that changes the classifier's decisions.
    mc_seed_means, mc_fold_vals = {}, {}
    for (c, s, m), path in runs.items():
        if c != "C" or m != "stgcn":
            continue
        mc_rows, _ = per_fold_metrics(path, prob_key="mc_prob")
        if not mc_rows:
            continue
        mc_fold_vals[s] = {k: np.array([r[k] for r in mc_rows]) for k in SCALARS}
        mc_seed_means[s] = {k: float(np.mean([r[k] for r in mc_rows]))
                            for k in SCALARS}
    if mc_seed_means:
        rows.append(_row("D +mc ‡", mc_seed_means, mc_fold_vals))
    lines += [md_table(headers, rows), ""]
    if c_stgcn:
        lines += ["† **`D +temp` is C's classifier, unchanged.** D is C plus "
                  "post-hoc calibrated confidence (`PREREGISTRATION.md` §5b.3), "
                  "and temperature scaling is monotonic — it cannot move any "
                  "window's argmax. Its accuracy, macro-F1, balanced accuracy "
                  "and MCC are **identical to C's by construction**, which is "
                  "why that row repeats C's numbers rather than reporting new "
                  "ones. Judge `+temp` on calibration instead (Brier/NLL/ECE in "
                  "`03_confidence.md`), never on this table.", ""]
    if mc_seed_means:
        d_f1 = np.mean([v["macro_f1"] for v in mc_seed_means.values()])
        c_f1 = np.mean([v["macro_f1"] for v in c_stgcn.values()])
        lines += [f"‡ **`D +mc` is NOT identical to C** — MC-dropout averages 20 "
                  "stochastic passes, which is not monotonic and does move the "
                  f"argmax. Here it changes macro-F1 by **{d_f1 - c_f1:+.4f}** "
                  "over C. Note this is a *different mechanism* from the "
                  "temperature row above, and it is the only part of the "
                  "confidence layer that alters what the system predicts rather "
                  "than how confident it claims to be. On an imbalanced set MC "
                  "averaging can also flip predictions toward the majority "
                  "class, so it is judged on macro-F1 and MCC, never accuracy.", ""]

    # Paired deltas between ST-GCN rungs, over the seeds both rungs share.
    lines += ["## Paired comparisons (ST-GCN rungs)", "",
              "Folds are seed-independent (`GroupKFold`, no shuffle), so runs "
              "are paired fold-for-fold and the difference is what matters, "
              "not the two means separately.", ""]
    # Register +mc as a rung so it is compared under the same paired protocol as
    # every other rung, rather than being asserted from its mean alone. Added
    # after the ladder table is built so it does not appear there twice.
    if mc_seed_means:
        seed_means[(MC_RUNG, "stgcn")] = mc_seed_means
        fold_vals[(MC_RUNG, "stgcn")] = mc_fold_vals

    # The ladder's own rungs, then each imbalance-arm run against the rung it
    # modifies -- C+w vs C, never C+w vs B, because the arm changes the objective
    # and nothing else.
    comparisons = [("A", "B"), ("B", "C"), ("A", "C"), ("C", MC_RUNG)]
    arm_pairs = sorted({(base, c) for (c, m) in seed_means if m == "stgcn"
                        for base, suffix in [loss_arm_of(c)] if suffix})
    comparisons += arm_pairs

    drows = []
    for lo, hi in comparisons:
        a, b = seed_means.get((lo, "stgcn"), {}), seed_means.get((hi, "stgcn"), {})
        shared = sorted(set(a) & set(b))
        if not shared:
            continue
        for metric in ["macro_f1", "balanced_acc", "mcc"]:
            d = np.array([b[s][metric] - a[s][metric] for s in shared])
            # Per-fold pairs: fold f is the same source videos in every run, so
            # the rungs are paired fold-for-fold as well as seed-for-seed.
            d_fold = np.concatenate([fold_vals[(hi, "stgcn")][s][metric]
                                     - fold_vals[(lo, "stgcn")][s][metric]
                                     for s in shared])
            wins = int((d_fold > 0).sum())
            p = (float(stats.ttest_1samp(d, 0.0).pvalue) if len(d) > 1
                 and np.std(d) > 0 else float("nan"))
            drows.append([f"{hi} vs {lo}", metric, len(shared),
                          f"{d.mean():+.4f}",
                          f"{np.std(d, ddof=1):.4f}" if len(d) > 1 else "-",
                          "-" if np.isnan(p) else f"{p:.3f}",
                          f"{wins}/{len(d_fold)}"])
    if drows:
        n_seeds = max(r[2] for r in drows)
        n_pairs = max(int(r[6].split("/")[1]) for r in drows)
        df = n_seeds - 1
        crit = float(stats.t.ppf(0.975, df)) if df > 0 else float("nan")
        lines += [md_table(["comparison", "metric", "n seeds", "delta",
                            "sd(seed)", "p", "per-fold wins"], drows), "",
                  f"`p` is a two-sided paired t-test over the {n_seeds} "
                  f"seed-level means ({df} df, critical t {crit:.2f}). Read it "
                  "together with the effect size and the win count, not as a "
                  "gate. The per-fold win count is **descriptive, not a sign "
                  "test**: folds repeat the same source videos across seeds, so "
                  f"the {n_pairs} pairs are not independent.", ""]

    if arm_pairs:
        lines += ["The `+w` / `+f` rows are the **off-ladder imbalance arm** "
                  "(`PREREGISTRATION.md` §5b.8), not rungs: they change the training "
                  "objective, which §5b.3 fixed in advance. Each is "
                  "paired against the rung it modifies. "
                  + " · ".join(f"`{s}` = {d}" for s, d in LOSS_ARM_SUFFIX.items())
                  + ".", ""]

    # Best ST-GCN vs best baseline -- the study's central negative result.
    # Restricted to the pre-registered ladder on purpose: an exploratory arm that
    # happened to score higher must not silently become "the best ST-GCN rung".
    best = best_ladder_config_per_model(seed_means)
    if "stgcn" in best and "mlp" in best:
        lines += ["## Headline comparison", "",
                  f"Best ST-GCN rung: **config {best['stgcn'][0]}, macro F1 "
                  f"{best['stgcn'][1]:.4f}**  \n"
                  f"MLP baseline: **macro F1 {best['mlp'][1]:.4f}**  \n"
                  f"Difference: **{best['mlp'][1] - best['stgcn'][1]:+.4f} in "
                  "favour of the baseline.**", "",
                  "The simplest model wins. This is a result, not a bug: it "
                  "bounds what the skeleton representation supports at this "
                  "dataset size.", ""]

    lines += build_ensemble_section(runs)
    return "\n".join(lines), seed_means


# Which saved runs the exploratory ensemble averages. The MLP is
# config-independent so config A is an arbitrary-but-fixed choice; the ST-GCN
# is taken at C with MC-dropout, which is the best ladder rung (D+mc).
ENSEMBLE_MEMBERS = [("A", "mlp", "oof_prob", "MLP baseline"),
                    ("C", "stgcn", "mc_prob", "ST-GCN D+mc")]


def ensemble_seed_means(runs, members=ENSEMBLE_MEMBERS):
    """{label: {seed: macro_f1}} for each member and their probability average.

    Metrics are computed per fold and then averaged, exactly as every other row
    in this report, so the ensemble number is directly comparable to the ladder
    rather than to a pooled out-of-fold score.
    """
    seeds = sorted({s for (c, s, m) in runs
                    if all((cfg, s, mdl) in runs for cfg, mdl, _, _ in members)})
    out = defaultdict(dict)
    for seed in seeds:
        probs, true, fold_of = [], None, None
        for cfg, mdl, key, label in members:
            blob = torch.load(runs[(cfg, seed, mdl)], map_location="cpu",
                              weights_only=False)
            prob = np.asarray(blob[key], dtype=np.float64)
            if not np.isfinite(prob).all():
                probs = []
                break
            probs.append(prob)
            true = np.asarray(blob["oof_true"])
            fold_of = np.asarray(blob["fold_of"])
            out[label][seed] = _fold_mean_macro_f1(prob, true, fold_of)
        if probs:
            out["ensemble"][seed] = _fold_mean_macro_f1(
                np.mean(probs, axis=0), true, fold_of)
    return out


def _fold_mean_macro_f1(prob, true, fold_of):
    pred = prob.argmax(axis=1)
    per_fold = [metrics_mod.compute_metrics(true[fold_of == f], pred[fold_of == f],
                                            len(common.CLASS_NAMES))["macro_f1"]
                for f in sorted(set(fold_of.tolist())) if f >= 0]
    return float(np.mean(per_fold))


def build_ensemble_section(runs, members=ENSEMBLE_MEMBERS):
    """The exploratory model-averaging arm.

    NOT pre-registered. `PREREGISTRATION.md` 5b fixed the ladder and the baselines before
    labeling; averaging two of them is a post-hoc idea and is reported under
    5b.8's rule for exploratory arms -- shown in full, kept out of the headline.
    """
    em = ensemble_seed_means(runs, members)
    if "ensemble" not in em or len(em["ensemble"]) < 2:
        return []
    labels = [lab for _, _, _, lab in members]
    best_member = max(labels, key=lambda k: np.mean(list(em[k].values())))
    shared = sorted(set(em["ensemble"]) & set(em[best_member]))
    d = np.array([em["ensemble"][s] - em[best_member][s] for s in shared])
    p = (float(stats.ttest_1samp(d, 0.0).pvalue)
         if len(d) > 1 and np.std(d) > 0 else float("nan"))

    rows = []
    for lab in labels + ["ensemble"]:
        vals = np.array([em[lab][s] for s in sorted(em[lab])])
        name = ("**" + " + ".join(labels) + " (probability average)**"
                if lab == "ensemble" else lab)
        rows.append([name, len(vals), f"{vals.mean():.4f}",
                     f"{np.std(vals, ddof=1):.4f}" if len(vals) > 1 else "-"])

    return ["## Exploratory: model averaging", "",
            "**Not a rung and not pre-registered.** `PREREGISTRATION.md` §5b fixed the "
            "ladder and the baselines before any window was labeled; averaging "
            "two of them is a post-hoc idea, so it is reported here under "
            "§5b.8's rule for exploratory arms and is deliberately **excluded "
            "from the headline comparison above**.", "",
            "The two models make *different* mistakes, so averaging their "
            "predicted probabilities — no retraining, no new parameters — "
            "recovers some of what neither has alone.", "",
            md_table(["model", "seeds", "macro F1", "sd(seed)"], rows), "",
            f"**Ensemble − {best_member} = {d.mean():+.4f}**, "
            f"p = {p:.5f}, better in **{int((d > 0).sum())}/{len(d)}** seeds "
            f"(paired across seeds, same folds).", "",
            "Read this as a statement about the *task*, not a proposed system: "
            "if a two-line average beats every rung, the ladder's ceiling was "
            "never a property of any single architecture. See "
            "`05_representation.md` for what each model is missing.", ""]


def build_confidence(runs):
    """Calibration + selective prediction for every saved run, per model.

    This report **is** config D. PREREGISTRATION.md Sec. 5b.3 defines D as config C
    plus calibrated confidence, applied post-hoc, so a `+temp` or `+mc` row over
    a config C run is a D number. The rows are named for the transform rather
    than the rung because the same transform is applied to every run here.
    """
    lines = ["# 03 — Confidence and calibration (config D)", "",
             "**This report is config D.** `PREREGISTRATION.md` §5b.3 defines D as "
             "C + calibrated confidence, computed **post-hoc over C's saved "
             "predictions** — nothing is retrained, so D's classifier is "
             "bit-identical to C's. Rows are named for the *transform* (`raw`, "
             "`+temp`, `+mc`) rather than the rung, because the same three "
             "transforms are applied to every saved run: `+temp`/`+mc` over a "
             "**config C ST-GCN** row is what the thesis calls D.", "",
             "Temperature is fit **leave-one-fold-out**: each fold is calibrated "
             "with a T fit on the other folds. Fitting one T on all out-of-fold "
             "predictions would be fitting on the evaluation set.", "",
             "Reading the table:", "",
             "- `+temp` **cannot** change accuracy, macro-F1 or MCC — temperature "
             "scaling is monotonic and cannot move any single window's argmax. "
             "Identical values in those three columns are the expected result, "
             "not a copy-paste error. Judge `+temp` on **Brier and NLL** "
             "(primary) and ECE (secondary, 15 equal-mass bins).",
             f"- `+mc` (20 MC-dropout passes) **can** change them: averaging "
             f"flips predictions, and on a {minority_rate(runs):.1%} minority it "
             "tends to flip them toward the majority class. Judge it on "
             "**macro-F1 and MCC**, never on accuracy alone.",
             "- **AURC is comparable within a row-group but not across the "
             "`raw`/`+temp` boundary.** Each fold gets its own T, so windows "
             "re-rank across fold boundaries; a raw-vs-temp AURC gap measures "
             "fold-to-fold temperature spread, not better abstention.", ""]

    headers = ["run", "model", "variant", "accuracy", "macro F1", "MCC",
               "Brier", "NLL", "ECE", "AURC"]
    keys = ["accuracy", "macro_f1", "mcc", "brier", "nll", "ece", "aurc"]
    rows = []
    best_for_curve = None
    best_rung = None  # best config C ST-GCN -- the ablation's actual D rung
    curves = {}       # (run_label, model) -> (calibrated_prob, true)
    skipped_mc = set()
    for (c, s, m), path in sorted(runs.items()):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        prob = np.asarray(blob.get("oof_prob"), dtype=np.float64)
        if prob.ndim != 2 or (prob.sum(axis=1) <= 0).any():
            continue
        try:
            report = analyze_confidence.confidence_report(blob)
        except (KeyError, ValueError, RuntimeError):
            continue
        label = f"{c}_seed{s}"
        for name in analyze_confidence.VARIANTS:
            got = report["variants"][name]
            # evaluate.py writes NaN into mc_prob for a model with no dropout
            # layer, so +mc does not exist for every architecture.
            if not all(np.isfinite(got[k]) for k in keys):
                if name == "+mc":
                    skipped_mc.add(MODEL_LABEL[m])
                continue
            rows.append([label, MODEL_LABEL[m], name]
                        + [f"{got[k]:.4f}" for k in keys])
        tempered = np.asarray(report["prob"]["+temp"], dtype=np.float64)
        true = np.asarray(report["y"])
        curves[(label, m)] = (tempered, true)
        score = report["variants"]["+temp"]["macro_f1"]
        if best_for_curve is None or score > best_for_curve[0]:
            best_for_curve = (score, label, m, tempered, true)
        if c == "C" and m == "stgcn" and (best_rung is None
                                          or score > best_rung[0]):
            best_rung = (score, label, m, tempered, true)
    lines += [md_table(headers, rows), ""]
    if skipped_mc:
        lines += ["**No `+mc` row for the "
                  + ", ".join(sorted(skipped_mc))
                  + ".** MC-dropout needs a dropout layer to sample; that "
                  "architecture has none, so `evaluate.py` stores NaN rather "
                  "than an MC estimate. This is an architectural fact, not a "
                  "missing run — do not report it as an MC-dropout failure.", ""]

    if best_for_curve is not None:
        _, label, model, p, true = best_for_curve
        lines += ["## Selective prediction — best calibrated run "
                  f"({label}, {MODEL_LABEL[model]})", "",
                  "The system answers only on the windows it is most confident "
                  "about and defers the rest to the teacher. This is the best "
                  "*system* in the study, which is a baseline, not the thesis "
                  "model — the D rung of the ablation is the config C ST-GCN "
                  "below.", "",
                  md_table(["coverage", "windows answered", "error rate",
                            "accuracy"], selective_rows(p, true)), ""]

        # The ablation's D rung specifically. Reported separately so the
        # headline number (a baseline) is never mistaken for the rung.
        if best_rung is not None and best_rung[1:3] != (label, model):
            _, rlabel, rmodel, rp, rt = best_rung
            lines += [f"## Selective prediction — the D rung ({rlabel}, "
                      f"{MODEL_LABEL[rmodel]})", "",
                      "Config C's ST-GCN with leave-one-fold-out temperature. "
                      "This is the row the ablation ladder's D rung refers to.", "",
                      md_table(["coverage", "windows answered", "error rate",
                                "accuracy"], selective_rows(rp, rt)), ""]

        lines += ["*Caveat:* pooled AURC shifts slightly under leave-one-fold-out "
                  "temperature because each fold gets its own T, which re-ranks "
                  "windows across fold boundaries. Report ECE/NLL/Brier for "
                  "calibration; treat AURC as within-fold only.", ""]
    return "\n".join(lines), best_for_curve, curves


# ---------------------------------------------------------------- figures

# ------------------------------------------------------------ learning curves

# diagnose_data_need.py writes one JSON per curve. The labels are the ladder's:
# a warm-started ST-GCN curve is config B's architecture, so B-A at each point
# is the within-dataset test of whether pretraining's advantage shrinks (P3).
DATA_NEED_CURVES = [
    ("mlp", "data_need.json", "MLP baseline"),
    ("stgcn", "data_need_stgcn.json", "ST-GCN scratch (A)"),
    ("stgcn_pre", "data_need_stgcn_pretrained.json", "ST-GCN pretrained (B)"),
]
FRACS = (0.25, 0.50, 0.75, 1.00)
# The JSON key "rooms" is the group-subsampling mode, which drops whole source videos.
MODE_LABEL = {"rooms": "by source video", "windows": "by window"}


def load_data_need():
    """{key: parsed json} for whichever learning-curve files exist on disk."""
    found = {}
    for key, name, _ in DATA_NEED_CURVES:
        path = paths.data_need_file(name)
        if os.path.exists(path):
            with open(path) as f:
                found[key] = json.load(f)
    return found


def curve_points(payload, mode):
    """{frac: {'mean','sd','n'}} over the seeds at each fraction."""
    by_frac = defaultdict(list)
    for row in payload.get("learning_curve", []):
        if row["mode"] == mode:
            by_frac[round(row["frac"], 2)].append(row)
    out = {}
    for frac, rows in by_frac.items():
        vals = np.array([r["macro_f1"] for r in rows], dtype=float)
        out[frac] = {"mean": float(vals.mean()),
                     "sd": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan"),
                     "n": float(np.mean([r["n_train_mean"] for r in rows]))}
    return out


def build_data_need(found):
    """04_data_need.md -- the learning curves behind predictions P2 and P3."""
    if not found:
        return ("# 04 — Learning curves\n\nNo curves on disk. Run "
                "`steps/diagnose_data_need.py` (see EXECUTION.md §6.2).\n")

    any_payload = next(iter(found.values()))
    out = ["# 04 — Learning curves (data need)",
           "",
           "Does more data help? Each point trains on a fraction of the **training** "
           "folds; the test fold is always whole, so every point is scored on the "
           "same held-out data. 3 seeds per point, 5-fold CV grouped by source video.",
           "",
           f"Full dataset: **{any_payload['n_windows']} windows** across "
           f"**{any_payload['n_rooms']} source videos**.",
           ""]

    lr = found.get("mlp", {}).get("logistic_regression")
    if lr:
        out += ["## Linear control",
                "",
                "Logistic regression on the same 85 pose-summary features, same folds:",
                "",
                md_table(["metric", "mean", "sd(fold)"],
                         [[k, f"{v['mean']:.4f}", f"{v['sd_fold']:.4f}"]
                          for k, v in lr.items()]),
                ""]

    for mode in ("rooms", "windows"):
        curves = {k: curve_points(p, mode) for k, p in found.items()}
        if not any(curves.values()):
            continue
        out += [f"## Subsampled {MODE_LABEL[mode]}", ""]
        headers = ["train frac", "n train"]
        for key, _, label in DATA_NEED_CURVES:
            if key in curves:
                headers.append(f"{label} macro F1")
        show_delta = "stgcn" in curves and "stgcn_pre" in curves
        if show_delta:
            headers.append("B − A")

        rows = []
        for frac in FRACS:
            present = [c[frac] for c in curves.values() if frac in c]
            if not present:
                continue
            row = [f"{frac:.0%}", f"{present[0]['n']:.0f}"]
            for key, _, _lbl in DATA_NEED_CURVES:
                if key in curves:
                    pt = curves[key].get(frac)
                    row.append(f"{pt['mean']:.4f} ± {pt['sd']:.4f}" if pt else "—")
            if show_delta:
                a, b = curves["stgcn"].get(frac), curves["stgcn_pre"].get(frac)
                row.append(f"{b['mean'] - a['mean']:+.4f}" if a and b else "—")
            rows.append(row)
        out += [md_table(headers, rows), ""]

        # Is the curve still climbing? Judged on the 25%->100% total, not on the
        # last step: a single step is the difference of two 3-seed means and is
        # the same size as seed noise, so it flips sign on resampling. sd(seed)
        # is pooled across the curve's four points (12 seeds) for the same
        # reason -- any one point's sd is an unusable 2-df estimate.
        verdict = []
        for key, _, label in DATA_NEED_CURVES:
            c = curves.get(key)
            if not c or not {0.25, 0.75, 1.00} <= set(c):
                continue
            sds = [c[f]["sd"] for f in sorted(c) if np.isfinite(c[f]["sd"])]
            pooled = float(np.sqrt(np.mean(np.square(sds)))) if sds else float("nan")
            total = c[1.00]["mean"] - c[0.25]["mean"]
            step = c[1.00]["mean"] - c[0.75]["mean"]
            verdict.append([label, f"{total:+.4f}", f"{step:+.4f}", f"{pooled:.4f}",
                            "**still rising**" if total > 2 * pooled else "flat"])
        if verdict:
            out += ["**Is it still climbing?** Judged on the 25% → 100% total against "
                    "pooled sd(seed); a single step is the difference of two 3-seed "
                    "means and is the same size as seed noise.", "",
                    md_table(["curve", "25% → 100%", "last step", "pooled sd(seed)",
                              "verdict"], verdict),
                    ""]
    return "\n".join(out)


def plot_data_need(found):
    """One figure per subsampling mode: every curve, with sd(seed) bands."""
    if not found:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(paths.figures_dir(), exist_ok=True)
    made = []
    for mode in ("rooms", "windows"):
        curves = {k: curve_points(p, mode) for k, p in found.items()}
        if not any(curves.values()):
            continue
        plt.figure(figsize=(7, 4.5))
        for key, _, label in DATA_NEED_CURVES:
            c = curves.get(key)
            if not c:
                continue
            fracs = sorted(c)
            mean = np.array([c[f]["mean"] for f in fracs])
            sd = np.array([c[f]["sd"] for f in fracs])
            xs = [c[f]["n"] for f in fracs]
            plt.plot(xs, mean, marker="o", label=label)
            plt.fill_between(xs, mean - sd, mean + sd, alpha=.15)
        plt.xlabel("training windows")
        plt.ylabel("macro F1 (out-of-fold)")
        plt.title(f"Learning curves — subsampled {MODE_LABEL[mode]}\n"
                  "3 seeds per point, test fold always whole")
        plt.legend(fontsize=8)
        plt.grid(alpha=.3)
        plt.tight_layout()
        out = paths.figure_file(f"data_need_{mode}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        made.append(out)
    return made


def plot_loss_curves(runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(paths.figures_dir(), exist_ok=True)
    made = []
    for (c, s, m), path in sorted(runs.items()):
        lf = paths.train_loss_file(f"{c}_seed{s}", m)
        if not os.path.exists(lf):
            continue
        with open(lf) as f:
            hist = json.load(f).get("per_fold_epoch_loss", [])
        if not hist:
            continue
        plt.figure(figsize=(7, 4.5))
        for i, fold in enumerate(hist):
            plt.plot(range(1, len(fold) + 1), fold, alpha=.75, label=f"fold {i}")
        plt.xlabel("epoch")
        plt.ylabel("mean training cross-entropy")
        plt.title(f"Training loss per CV fold\n{c}_seed{s} | model={m}")
        plt.legend(fontsize=8)
        plt.grid(alpha=.3)
        plt.tight_layout()
        out = paths.loss_curve_file(f"{c}_seed{s}", m)
        plt.savefig(out, dpi=150)
        plt.close()
        made.append(out)
    return made


def plot_confidence_figures(curves):
    """Reliability + risk-coverage for EVERY calibrated model-run.

    These are config D's figures. Emitting them only for the single
    best-scoring run left 44 of 45 model-runs with no calibration evidence at
    all, so no per-config claim could be supported by a figure.
    """
    if not curves:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(paths.figures_dir(), exist_ok=True)
    made = []

    for (label, model), (p, true) in sorted(curves.items()):
        conf, acc, counts = cal.reliability_curve(p, true)
        plt.figure(figsize=(5.5, 5.5))
        plt.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        plt.plot(conf, acc, "o-", label="model")
        plt.xlabel("mean predicted confidence")
        plt.ylabel("observed accuracy")
        plt.title(f"Reliability diagram (equal-mass bins)\n{label} | {model} | "
                  f"temperature-scaled")
        plt.legend()
        plt.grid(alpha=.3)
        plt.tight_layout()
        out = paths.figure_file(f"{label}_{model}_reliability.png")
        plt.savefig(out, dpi=150)
        plt.close()
        made.append(out)

        cov, risk = cal.risk_coverage_curve(p, true)
        plt.figure(figsize=(6.5, 4.5))
        plt.plot(cov, risk, lw=2)
        plt.xlabel("coverage (fraction of windows the model answers on)")
        plt.ylabel("error rate on answered windows")
        plt.title(f"Risk-coverage curve\n{label} | {model} | temperature-scaled")
        plt.grid(alpha=.3)
        plt.tight_layout()
        out = paths.figure_file(f"{label}_{model}_risk_coverage.png")
        plt.savefig(out, dpi=150)
        plt.close()
        made.append(out)
    return made


# ---------------------------------------------------------------- index

def load_representation():
    """Parsed representation.json, or None when the diagnostic has not run."""
    path = paths.representation_file()
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def build_representation(payload, seed_means=None):
    """05_representation.md -- what the skeleton carries, and how far it falls short.

    Answers the question 02_ablation.md raises but cannot settle: the ST-GCN
    loses to an MLP on hand-made statistics -- why? Three measurements, all
    exploratory (none is in the PREREGISTRATION.md 5b pre-registration).
    """
    if not payload:
        return ("# 05 — What the representation carries\n\n"
                "Not measured. Run `steps/diagnose_representation.py` "
                "(see `EXECUTION.md` §6.3).\n")

    out = ["# 05 — What the representation carries", "",
           "**Exploratory.** None of the three measurements below is in the "
           "`PREREGISTRATION.md` §5b pre-registration; they were added after the ladder "
           "came out to explain *why* it came out that way. Reported under "
           "§5b.8's rule for exploratory work: shown in full, kept out of "
           "every headline.", "",
           f"Measured on the same **{payload['n_windows']} windows** across "
           f"**{payload['n_source_videos']} source videos**, the same "
           f"{payload['k_folds']}-fold grouping, and the same metric code as "
           "the ladder.", "",
           "---", "",
           "## 1. Where the signal lives",
           "",
           "The 85 baseline features are three groups with very different "
           "meanings: per-joint **mean** position (what posture the student "
           "holds), per-joint **std** over time (how much each joint moves), "
           "and per-joint mean **speed**. A linear probe on each group, over "
           "the pre-registered folds, says which one the classifier is "
           "actually using.", ""]

    fg = payload["feature_groups"]
    out += [md_table(["feature group", "n features", "macro F1", "sd(fold)",
                      "balanced acc", "MCC"],
                     [[r["name"], r["n_features"], f"{r['macro_f1']:.4f}",
                       f"{r['macro_f1_sd_fold']:.4f}",
                       f"{r['balanced_acc']:.4f}", f"{r['mcc']:.4f}"]
                      for r in fg]), ""]

    mean_only = next((r for r in fg if r["name"].startswith("mean only")), None)
    std_only = next((r for r in fg if r["name"].startswith("std only")), None)
    all85 = next((r for r in fg if r["name"].startswith("all 85")), None)
    if mean_only and std_only and all85:
        out += [f"**Static posture is nearly worthless "
                f"({mean_only['macro_f1']:.4f}); movement is almost "
                f"everything.** Per-joint variability alone reaches "
                f"{std_only['macro_f1']:.4f} — within "
                f"{abs(all85['macro_f1'] - std_only['macro_f1']):.4f} of all 85 "
                "features together.", "",
                "Two consequences follow, and together they are the most "
                "likely explanation for the ladder's result:", "",
                "1. **The task is closer to motion-energy detection than to "
                "posture recognition.** Whatever the annotators were "
                "responding to, its skeleton-visible shadow is mostly *how "
                "much the student moved*.",
                "2. **The baseline is handed a sufficient statistic; the "
                "ST-GCN has to learn one.** `std` and `speed` are literally "
                "input columns for the MLP and the logistic regression. The "
                "ST-GCN sees raw coordinates and must rediscover per-joint "
                "variance through stacked graph convolutions — from "
                f"{payload['n_windows']} examples. It is being asked to learn "
                "the very feature it is competing against.", ""]

    out += ["---", "", "## 2. Which joints actually exist", "",
            "The ST-GCN's adjacency matrix is a fixed human skeleton: "
            "information is forced to flow hip → knee → ankle whether or not "
            "those joints were ever observed. Mean detector confidence per "
            "joint, over every frame of every window:", ""]
    jc = sorted(payload["joint_confidence"], key=lambda r: -r["mean_confidence"])
    out += [md_table(["joint", "mean confidence", "observed?"],
                     [[r["joint"], f"{r['mean_confidence']:.3f}",
                       "yes" if r["observed"] else "**no**"] for r in jc]), ""]
    dead = [r for r in jc if not r["observed"]]
    if dead:
        names = ", ".join(f"`{r['joint']}`" for r in dead)
        out += [f"**{len(dead)} of {len(jc)} joints ({len(dead) / len(jc):.0%}) "
                f"are effectively unobserved** (mean confidence below "
                f"{payload['unobserved_conf_threshold']}): {names}. Students "
                "are seated behind desks.", "",
                "This penalises the two model families very differently. A "
                "linear layer can down-weight the corresponding input columns "
                "to nothing. The ST-GCN cannot: its graph edges are fixed a "
                "priori, so every forward pass propagates those coordinates "
                "through the spatial convolution. The NTU-60 warm start "
                "compounds it — those filters were fitted on full-body actions "
                "where the legs are visible and informative.", ""]

    hc = payload.get("human_ceiling") or {}
    if hc.get("per_annotator"):
        out += ["---", "", "## 3. The human ceiling", "",
                "κ answers *\"do the annotators agree?\"*. It is "
                "chance-corrected agreement, not an achievable score, so it "
                "cannot be set beside a model's macro-F1. This measures the "
                "comparable thing: **score each annotator on the model's own "
                "task** — predict the consensus label — using the blind "
                "overlap.", "",
                f"Restricted to the **{hc['n_all_three_trainable']} overlap "
                f"windows all three annotators judged trainable** (of "
                f"{hc['n_overlap_windows']} overlap windows), and scored only "
                "where the *other two* agree, so a judge is never scored "
                "against a consensus they helped form.", ""]
        out += [md_table(["annotator", "n", "accuracy", "macro F1", "MCC"],
                         [[a["annotator"], a["n"], f"{a['accuracy']:.4f}",
                           f"{a['macro_f1']:.4f}", f"{a['mcc']:.4f}"]
                          for a in hc["per_annotator"]]
                         + [["**mean human**", "",
                             f"**{hc['mean_accuracy']:.4f}**",
                             f"**{hc['mean_macro_f1']:.4f}**", ""]]), ""]
        best_model = None
        if seed_means:
            best_model = max(
                ((k, float(np.mean([v["macro_f1"] for v in per.values()])))
                 for k, per in seed_means.items()), key=lambda kv: kv[1])
        if best_model:
            gap = hc["mean_macro_f1"] - best_model[1]
            out += [f"**Best model on this dataset: "
                    f"{MODEL_LABEL.get(best_model[0][1], best_model[0][1])} at "
                    f"macro F1 {best_model[1]:.4f}. Mean human: "
                    f"{hc['mean_macro_f1']:.4f}. Gap: {gap:+.4f}.**", ""]
        out += ["> ⚠️ **This is an upper bound, not an estimate.** It is "
                "measured only on windows all three annotators judged "
                "trainable, which selects for easy ones; the ceiling over all "
                "windows is lower. Do not quote it as \"human accuracy\".", "",
                "**The gap is a statement about the modality, not the models.** "
                "`annotate.py` shows annotators the **RGB clip** with a "
                "skeleton overlay, so a human sees the phone, the book, the "
                "gaze direction, and who the student is turned toward. The "
                "models see 17 2-D points, four of which "
                "(§2) are not really there.", ""]

    out += ["---", "", "## What this does and does not establish", "",
            "**Supported:** within a 2-D skeleton the task largely reduces to "
            "motion energy, which a linear model on summary statistics already "
            "captures — so extra architectural capacity buys little, while "
            "*model diversity* still buys something "
            "(`02_ablation.md`, exploratory section).", "",
            "**Not supported:** that this is a ceiling on the *problem*. "
            "Humans reach far higher on the same windows using information the "
            "skeleton discards. Nothing here says a richer representation "
            "would fail — only that this one is the binding constraint.", "",
            "**Also not supported:** any claim that the ST-GCN is a poor "
            "architecture. It is a poor fit *for this input at this scale*, "
            "for the two concrete reasons above.", ""]
    return "\n".join(out)


def build_readme(runs, figures, seed_means):
    configs = sorted({c for (c, _, _) in runs})
    seeds = sorted({s for (_, s, _) in runs})
    models = sorted({m for (_, _, m) in runs}, key=MODELS.index)
    lines = [
        "# outputs/", "",
        "Everything here is **generated**. Delete the whole folder and "
        "regenerate it with the commands at the bottom — nothing in it is "
        "irreplaceable.", "",
        "## Start here", "",
        "| file | what it is |",
        "|---|---|",
        "| `reports/01_headline.md` | the handful of numbers to quote |",
        "| `reports/02_ablation.md` | full A/B/C ladder and baselines |",
        "| `reports/03_confidence.md` | calibration + selective prediction |",
        "| `figures/` | .png figures, named `<run>_<model>_<what>.png` |",
        "| `runs/<run>/<model>/` | raw artifacts these reports were built from |",
        "| `logs/` | stdout/stderr of each sweep run |","",
        "## What is currently on disk", "",
        f"- configs: {', '.join(configs) if configs else '(none)'}",
        f"- seeds: {', '.join(map(str, seeds)) if seeds else '(none)'}",
        f"- models per run: {', '.join(MODEL_LABEL[m] for m in models) if models else '(none)'}",
        f"- runs saved: {len(runs)}",
        f"- figures: {len(figures)}", "",
        "## Regenerate", "",
        "```bash",
        "# 1. the ablation sweep; resumable, skips finished runs",
        "bash steps/run_sweep.sh",
        "",
        "# 2. these reports and figures (reads saved artifacts only, trains nothing)",
        ".venv/bin/python steps/make_report.py",
        "```", "",
    ]
    return "\n".join(lines)


def build_headline(runs, seed_means, best_for_curve):
    lines = ["# 01 — Headline numbers", "",
             "Protocol for every number here: **5-fold cross-validation, folds "
             "grouped by source video.** Out-of-fold predictions only.", ""]
    if not seed_means:
        lines += ["No runs found under `outputs/runs/`. Run the sweep first — "
                  "see `outputs/README.md`.", ""]
        return "\n".join(lines)

    _true_headline = np.asarray(torch.load(
        next(iter(runs.values())), map_location="cpu",
        weights_only=False)["oof_true"])
    _rate_headline = float((_true_headline == DISTRACTED_IDX).mean())

    rows = []
    for (c, m), per_seed in sorted(seed_means.items(),
                                   key=lambda kv: -np.mean(
                                       [v["macro_f1"] for v in kv[1].values()])):
        f1 = [v["macro_f1"] for v in per_seed.values()]
        # Only the ST-GCN takes the warm start and the augmentation, so a config
        # letter describes the baselines' training not at all -- they are the
        # same model in every rung and are repeated only as a fixed reference.
        what = (CONFIG_DESC.get(c, "") if m == "stgcn"
                else "baseline — identical in every config")
        rows.append([f"{c} / {MODEL_LABEL[m]}", what, f"{np.mean(f1):.4f}",
                     f"{np.std(f1, ddof=1):.4f}" if len(f1) > 1 else "-"])
    lines += ["## Every model, ranked by macro F1", "",
              md_table(["config / model", "what it is", "macro F1",
                        "sd(seed)"], rows), "",
              f"Evaluated set: **{len(_true_headline)} windows**, "
              f"**{_rate_headline:.1%}** Distracted. Always predicting the "
              f"majority class scores **{majority_macro_f1(_rate_headline):.4f}** "
              f"macro F1 and **{1 - _rate_headline:.1%}** accuracy — that is the "
              "floor, and it is why accuracy is not a headline number here.", ""]
    if best_for_curve is not None:
        _, label, model, p, true = best_for_curve
        cov, risk = cal.risk_coverage_curve(p, true)
        cov, risk = np.asarray(cov), np.asarray(risk)
        j60 = int(np.argmin(np.abs(cov - 0.6)))
        j100 = int(np.argmin(np.abs(cov - 1.0)))
        lines += ["## The confidence result", "",
                  f"Best calibrated run: **{label}, {MODEL_LABEL[model]}**.", "",
                  f"- Answering on **every** window: {1 - risk[j100]:.1%} accurate.",
                  f"- Answering only on the most-confident **60%**: "
                  f"**{1 - risk[j60]:.1%}** accurate, deferring the rest.", "",
                  "See `03_confidence.md` for the full curve.", ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    runs = discover_runs()
    if not runs:
        print(f"No saved runs under {paths.runs_dir()}. Run the sweep first:")
        print("  setsid nohup bash steps/run_sweep.sh >> outputs/logs/driver.log 2>&1 &")

    os.makedirs(paths.reports_dir(), exist_ok=True)
    os.makedirs(paths.figures_dir(), exist_ok=True)
    os.makedirs(paths.logs_dir(), exist_ok=True)

    ablation_md, seed_means = build_ablation(runs)
    confidence_md, best_for_curve, curves = build_confidence(runs)
    headline_md = build_headline(runs, seed_means, best_for_curve)
    data_need = load_data_need()
    data_need_md = build_data_need(data_need)
    representation_md = build_representation(load_representation(), seed_means)

    figures = []
    if not args.no_figures:
        figures += plot_loss_curves(runs)
        figures += plot_confidence_figures(curves)
        figures += plot_data_need(data_need)
    figures += sorted(glob.glob(os.path.join(paths.figures_dir(), "*.png")))
    figures = sorted(set(figures))

    written = []
    for name, text in [("01_headline.md", headline_md),
                       ("02_ablation.md", ablation_md),
                       ("03_confidence.md", confidence_md),
                       ("04_data_need.md", data_need_md),
                       ("05_representation.md", representation_md)]:
        path = paths.report_file(name)
        with open(path, "w") as f:
            f.write(text)
        written.append(path)
    with open(paths.outputs_readme_file(), "w") as f:
        f.write(build_readme(runs, figures, seed_means))
    written.append(paths.outputs_readme_file())

    print(f"runs found: {len(runs)} | figures: {len(figures)}")
    for p in written:
        print("  wrote", os.path.relpath(p, paths.REPO_ROOT))


if __name__ == "__main__":
    main()

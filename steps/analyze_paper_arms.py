"""The four exploratory arms the conference paper needs, over saved runs only.

1. Selective prediction on the ENSEMBLE (not the MLP), so the paper's coverage
   number and its headline classifier are the same system.
2. The majority floor recomputed on each retained subset. Abstention keeps the
   easy windows, which are majority-skewed, so a raw accuracy-at-coverage gain
   partly measures skew rather than the model.
3. Calibration of the ensemble: Brier, NLL, ECE, AURC, raw and leave-one-fold-out
   temperature-scaled.
4. Ensemble variants (which members, simple vs weighted) and member diversity.

Reads outputs/runs/*/*/oof_probs.pt. Trains nothing. CPU, about a minute.
Writes outputs/reports/06_paper_arms.md, outputs/diagnostics/paper_arms.json and
outputs/figures/ensemble_risk_coverage.png.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

import _bootstrap  # noqa: F401  -- puts ../lib on sys.path
import calibration as cal
import common
import metrics as metrics_mod
import paths

COVERAGE_POINTS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)

# (config, model, probability key, label). The MLP is config-independent, so A
# is arbitrary-but-fixed; the ST-GCN is taken at C with MC-dropout (rung D+mc).
MEMBERS = {
    "MLP": ("A", "mlp", "oof_prob"),
    "GRU": ("A", "gru", "oof_prob"),
    "ST-GCN D+mc": ("C", "stgcn", "mc_prob"),
    "ST-GCN C": ("C", "stgcn", "oof_prob"),
}

VARIANTS = [
    ("MLP + ST-GCN D+mc", ["MLP", "ST-GCN D+mc"]),
    ("MLP + GRU", ["MLP", "GRU"]),
    ("GRU + ST-GCN D+mc", ["GRU", "ST-GCN D+mc"]),
    ("MLP + ST-GCN C", ["MLP", "ST-GCN C"]),
    ("MLP + GRU + ST-GCN D+mc", ["MLP", "GRU", "ST-GCN D+mc"]),
]


def discover_runs():
    """{(config, seed, model): path} for everything saved under outputs/runs/."""
    found = {}
    for path in sorted(glob.glob(os.path.join(paths.runs_dir(), "*", "*", "oof_probs.pt"))):
        label = os.path.basename(os.path.dirname(os.path.dirname(path)))
        model = os.path.basename(os.path.dirname(path))
        if "_seed" not in label:
            continue
        config, seed = label.split("_seed", 1)
        try:
            found[(config, int(seed), model)] = path
        except ValueError:
            continue
    return found


def load_members(runs, seed):
    """{label: prob} plus (true, fold_of) for one seed, or None if incomplete."""
    probs, true, fold_of = {}, None, None
    for label, (config, model, key) in MEMBERS.items():
        path = runs.get((config, seed, model))
        if path is None:
            return None
        blob = torch.load(path, map_location="cpu", weights_only=False)
        prob = np.asarray(blob[key], dtype=np.float64)
        if not np.isfinite(prob).all():
            return None
        probs[label] = prob
        true = np.asarray(blob["oof_true"])
        fold_of = np.asarray(blob["fold_of"])
    return probs, true, fold_of


def fold_mean_macro_f1(prob, true, fold_of):
    """Per-fold macro-F1 averaged, matching every other table in the reports."""
    pred = prob.argmax(axis=1)
    per_fold = [metrics_mod.compute_metrics(true[fold_of == f], pred[fold_of == f],
                                            len(common.CLASS_NAMES))["macro_f1"]
                for f in sorted(set(fold_of.tolist())) if f >= 0]
    return float(np.mean(per_fold))


def selective_against_floor(prob, true, points=COVERAGE_POINTS):
    """Accuracy at each coverage next to the always-majority floor on the same windows.

    Retaining the most confident windows keeps the easy ones, which are skewed
    toward the majority class, so the floor has to be recomputed on the retained
    subset rather than taken from the full set.
    """
    conf = prob.max(axis=1)
    pred = prob.argmax(axis=1)
    order = np.argsort(-conf)
    n = len(true)
    rows = []
    for c in points:
        k = max(1, int(round(c * n)))
        keep = order[:k]
        y, p = true[keep], pred[keep]
        floor_acc = float((y == 0).mean())          # always predict Focused
        rows.append({
            "coverage": float(c),
            "n": int(k),
            "accuracy": float((y == p).mean()),
            "floor_accuracy": floor_acc,
            "gap": float((y == p).mean() - floor_acc),
            "macro_f1": float(metrics_mod.compute_metrics(
                y, p, len(common.CLASS_NAMES))["macro_f1"]),
            "distracted_frac": float((y == 1).mean()),
        })
    return rows


def calibration_row(prob, true, fold_of):
    """Brier / NLL / ECE / AURC for a probability vector, raw and LOFO-temperature."""
    scaled, temps = cal.leave_one_fold_out_temperature(prob, true, fold_of)
    out = {}
    for name, p in (("raw", prob), ("+temp", scaled)):
        out[name] = {
            "brier": cal.brier_score(p, true),
            "nll": cal.nll(p, true),
            "ece": cal.ece_equal_mass(p, true),
            "aurc": cal.aurc(p, true),
        }
    out["temperatures"] = [float(t) for t in temps]
    return out


def diversity(probs, true, pair):
    """Error overlap and oracle accuracy for two members."""
    a, b = (probs[pair[0]].argmax(axis=1), probs[pair[1]].argmax(axis=1))
    ea, eb = a != true, b != true
    both = float((ea & eb).sum())
    return {
        "err_a": int(ea.sum()),
        "err_b": int(eb.sum()),
        "shared": int(both),
        "shared_frac_of_a": float(both / ea.sum()) if ea.sum() else float("nan"),
        "oracle_accuracy": float((~(ea & eb)).mean()),
    }


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def mean_sd(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else float("nan"))


def analyse(runs):
    seeds = sorted({s for (_, s, _) in runs})
    per_seed = {}
    for seed in seeds:
        loaded = load_members(runs, seed)
        if loaded is None:
            continue
        probs, true, fold_of = loaded

        variants = {}
        for label, names in VARIANTS:
            avg = np.mean([probs[n] for n in names], axis=0)
            variants[label] = fold_mean_macro_f1(avg, true, fold_of)

        weighted = {}
        for w in np.round(np.arange(0.1, 1.0, 0.1), 1):
            avg = w * probs["MLP"] + (1 - w) * probs["ST-GCN D+mc"]
            weighted[float(w)] = fold_mean_macro_f1(avg, true, fold_of)

        ens = np.mean([probs["MLP"], probs["ST-GCN D+mc"]], axis=0)
        per_seed[seed] = {
            "members": {k: fold_mean_macro_f1(v, true, fold_of) for k, v in probs.items()},
            "variants": variants,
            "weighted": weighted,
            "selective": selective_against_floor(ens, true),
            "calibration": calibration_row(ens, true, fold_of),
            "calibration_mlp": calibration_row(probs["MLP"], true, fold_of),
            "diversity": diversity(probs, true, ("MLP", "ST-GCN D+mc")),
            "n": int(len(true)),
        }
    return per_seed


def build_report(per_seed):
    seeds = sorted(per_seed)
    first = per_seed[seeds[0]]
    lines = [
        "# 06 — Ensemble and selective prediction (paper arms)",
        "",
        "**Exploratory.** None of this is in the `PREREGISTRATION.md` §5b pre-registration; it was added "
        "to support the conference paper's ensemble-plus-confidence framing. Reported under "
        "§5b.8: shown in full, kept out of the pre-registered headline comparison.",
        "",
        f"Same {first['n']} windows, same 5-fold grouping by source video, same metric code as "
        f"the ladder. {len(seeds)} seeds. The ensemble is the probability average of the MLP "
        "baseline and the ST-GCN at rung D+mc — computed per fold, out-of-fold for both "
        "members, with no retraining and no test-set tuning.",
        "",
        "---",
        "",
        "## 1. Which members to average",
        "",
    ]

    rows = []
    for label in MEMBERS:
        m, s = mean_sd([per_seed[x]["members"][label] for x in seeds])
        rows.append([label, f"{m:.4f}", f"{s:.4f}"])
    for label, _ in VARIANTS:
        m, s = mean_sd([per_seed[x]["variants"][label] for x in seeds])
        rows.append([f"**{label}**", f"{m:.4f}", f"{s:.4f}"])
    lines += [md_table(["model / combination", "macro F1", "sd(seed)"], rows), ""]

    lines += ["### Weighted average (MLP weight w, ST-GCN D+mc weight 1−w)", ""]
    wrows = []
    for w in sorted(first["weighted"]):
        m, s = mean_sd([per_seed[x]["weighted"][w] for x in seeds])
        wrows.append([f"{w:.1f}", f"{m:.4f}", f"{s:.4f}"])
    lines += [md_table(["w (MLP)", "macro F1", "sd(seed)"], wrows), "",
              "Equal weighting is w = 0.5. A tuned w would be fit on the evaluation set, so "
              "the paper reports the simple average and shows this sweep only to establish "
              "that the result is not a knife-edge.", ""]

    lines += ["---", "", "## 2. Why averaging helps: the members fail differently", ""]
    drows = []
    d0 = first["diversity"]
    sf, sf_sd = mean_sd([per_seed[x]["diversity"]["shared_frac_of_a"] for x in seeds])
    orc, orc_sd = mean_sd([per_seed[x]["diversity"]["oracle_accuracy"] for x in seeds])
    drows.append(["MLP errors also made by ST-GCN D+mc", f"{sf:.1%}", f"{sf_sd:.1%}"])
    drows.append(["oracle accuracy (either member correct)", f"{orc:.4f}", f"{orc_sd:.4f}"])
    lines += [md_table(["quantity", "mean", "sd(seed)"], drows), "",
              f"On seed {seeds[0]} the MLP makes {d0['err_a']} errors and the ST-GCN "
              f"{d0['err_b']}; {d0['shared']} are shared. Disjoint failures are the mechanism "
              "the ensemble exploits — averaging lets each member's confident votes override "
              "the other's uncertain ones.", ""]

    lines += ["---", "", "## 3. Selective prediction, against the floor on the same windows", "",
              "At each coverage the system answers only on the most confident windows and "
              "defers the rest. **The comparison that matters is against the majority "
              "baseline recomputed on the retained subset**, not on the full set: abstention "
              "keeps easy windows, which are skewed toward Focused, so part of any raw "
              "accuracy gain is skew rather than skill.", ""]
    srows = []
    for i, point in enumerate(COVERAGE_POINTS):
        acc, acc_sd = mean_sd([per_seed[x]["selective"][i]["accuracy"] for x in seeds])
        flr, _ = mean_sd([per_seed[x]["selective"][i]["floor_accuracy"] for x in seeds])
        gap, gap_sd = mean_sd([per_seed[x]["selective"][i]["gap"] for x in seeds])
        mf1, _ = mean_sd([per_seed[x]["selective"][i]["macro_f1"] for x in seeds])
        dfr, _ = mean_sd([per_seed[x]["selective"][i]["distracted_frac"] for x in seeds])
        srows.append([f"{point:.0%}", first["selective"][i]["n"], f"{acc:.4f} ± {acc_sd:.4f}",
                      f"{flr:.4f}", f"**{gap:+.4f}** ± {gap_sd:.4f}", f"{mf1:.4f}",
                      f"{dfr:.1%}"])
    lines += [md_table(["coverage", "answered", "ensemble accuracy", "majority floor",
                        "gap over floor", "macro F1", "% Distracted retained"], srows), "",
              "The floor column rises with confidence because the retained subset becomes "
              "more majority-skewed — that is exactly the artefact this table exists to "
              "separate out. Read the **gap** column, not the accuracy column.", ""]

    lines += ["---", "", "## 4. Is the ensemble's confidence calibrated?", ""]
    crows = []
    for name, key in (("MLP baseline", "calibration_mlp"), ("Ensemble", "calibration")):
        for variant in ("raw", "+temp"):
            vals = {k: mean_sd([per_seed[x][key][variant][k] for x in seeds])[0]
                    for k in ("brier", "nll", "ece", "aurc")}
            crows.append([name, variant, f"{vals['brier']:.4f}", f"{vals['nll']:.4f}",
                          f"{vals['ece']:.4f}", f"{vals['aurc']:.4f}"])
    lines += [md_table(["model", "variant", "Brier", "NLL", "ECE", "AURC"], crows), "",
              "Temperature is fit **leave-one-fold-out**: each fold is calibrated with a T "
              "fit on the other folds, so no fold is calibrated on itself. Brier and NLL are "
              "primary (binning-free); ECE is secondary (15 equal-mass bins). AURC is not "
              "comparable across the raw/`+temp` boundary — each fold has its own T, so "
              "windows re-rank across fold boundaries.", ""]
    return "\n".join(lines)


def make_figure(per_seed, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = sorted(per_seed)
    cov = np.array(COVERAGE_POINTS)
    acc = np.array([[per_seed[s]["selective"][i]["accuracy"]
                     for i in range(len(cov))] for s in seeds])
    flr = np.array([[per_seed[s]["selective"][i]["floor_accuracy"]
                     for i in range(len(cov))] for s in seeds])

    plt.figure(figsize=(6, 4))
    plt.plot(cov, acc.mean(axis=0), "o-", lw=2, label="ensemble")
    plt.fill_between(cov, acc.mean(axis=0) - acc.std(axis=0),
                     acc.mean(axis=0) + acc.std(axis=0), alpha=0.2)
    plt.plot(cov, flr.mean(axis=0), "s--", lw=2, color="grey",
             label="majority floor on retained subset")
    plt.xlabel("coverage (fraction of windows answered)")
    plt.ylabel("accuracy")
    plt.title("Selective prediction vs the floor it must beat")
    plt.gca().invert_xaxis()
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    runs = discover_runs()
    per_seed = analyse(runs)
    if not per_seed:
        raise SystemExit("no complete seeds found under outputs/runs/")

    os.makedirs(paths.diagnostics_dir(), exist_ok=True)
    os.makedirs(paths.reports_dir(), exist_ok=True)
    json_path = os.path.join(paths.diagnostics_dir(), "paper_arms.json")
    with open(json_path, "w") as fh:
        json.dump({str(k): v for k, v in per_seed.items()}, fh, indent=2)

    report_path = paths.report_file("06_paper_arms.md")
    with open(report_path, "w") as fh:
        fh.write(build_report(per_seed) + "\n")

    print(f"seeds analysed: {sorted(per_seed)}")
    print(f"wrote {report_path}")
    print(f"wrote {json_path}")
    if not args.no_figure:
        os.makedirs(paths.figures_dir(), exist_ok=True)
        fig = paths.figure_file("ensemble_risk_coverage.png")
        make_figure(per_seed, fig)
        print(f"wrote {fig}")


if __name__ == "__main__":
    main()

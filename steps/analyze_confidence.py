"""Step: turn one saved cross-validation run into config D's calibration table.

Entirely post-hoc. Reads outputs/runs/<run_label>/<model>/oof_probs.pt and retrains nothing,
so D's classifier is bit-identical to C's.

Three variants are reported side by side so a calibration gain is attributable
to one mechanism rather than a bundle:

  C-raw   the run's own softmax output, uncalibrated
  D-temp  C-raw with leave-one-fold-out temperature scaling
  D-mc    the average of 20 MC-dropout passes taken during the run

D-temp's accuracy is identical to C-raw's by construction -- temperature scaling
is monotonic and cannot change which class wins. D-mc's is not.
"""
import _bootstrap  # noqa: F401  -- puts ../lib on sys.path

import argparse

import numpy as np
import torch

import calibration
import common
import metrics as metrics_mod

# Named for the TRANSFORM, not the config. The script runs on any saved run file,
# so hardcoding "C-raw" printed a config-C label over config A's numbers whenever
# you pointed it at runs/A_seed42/stgcn/oof_probs.pt. Which rung a row belongs to is carried
# by the run label in the header instead. Applying +temp/+mc to config C's run is
# what PREREGISTRATION.md Sec. 5b.3 calls config D.
VARIANTS = ("raw", "+temp", "+mc")
_METRIC_ORDER = ("accuracy", "macro_f1", "mcc", "brier", "nll", "ece", "aurc")


def _metrics(prob, y):
    # macro-F1 and MCC are PREREGISTRATION.md Sec. 5b.3's PRIMARY metrics -- the
    # trainable set is 17.4% Distracted, so accuracy is not informative. D-mc is
    # the one variant whose argmax can move relative to C, and if MC averaging
    # flips minority-class predictions, accuracy alone would not show it.
    pred = prob.argmax(axis=1)
    discriminative = metrics_mod.compute_metrics(
        y, pred, len(common.CLASS_NAMES))
    return {
        "accuracy": float((pred == y).mean()),
        "macro_f1": discriminative["macro_f1"],
        "mcc": discriminative["mcc"],
        "brier": calibration.brier_score(prob, y),
        "nll": calibration.nll(prob, y),
        "ece": calibration.ece_equal_mass(prob, y),
        "aurc": calibration.aurc(prob, y),
    }


def confidence_report(blob):
    """Compute every variant's metrics from one saved run dict."""
    for key in ("oof_prob", "mc_prob", "oof_true", "fold_of"):
        if key not in blob:
            raise KeyError(
                f"{key} is missing from this run file. Re-run evaluate.py: runs "
                "saved before MC-dropout landed do not carry it, and it cannot be "
                "reconstructed without the trained fold models.")

    raw = np.asarray(blob["oof_prob"], dtype=np.float64)
    mc = np.asarray(blob["mc_prob"], dtype=np.float64)
    y = np.asarray(blob["oof_true"])
    fold_of = np.asarray(blob["fold_of"])

    tempered, temperatures = calibration.leave_one_fold_out_temperature(
        raw, y, fold_of)

    prob = {"raw": raw, "+temp": tempered, "+mc": mc}
    return {
        "variants": {name: _metrics(p, y) for name, p in prob.items()},
        # The probabilities themselves, so a caller that wants a reliability or
        # risk-coverage curve does not have to refit the leave-one-fold-out
        # temperature and risk drifting from the numbers in this table.
        "prob": prob,
        "y": y,
        "temperatures": temperatures,
    }


def _print_report(report, run_label):
    print(f"\n=== Confidence report: {run_label} ===")
    print("rows are transforms of THIS run, not ablation rungs. Applying +temp/+mc")
    print("to a config C run is what PREREGISTRATION.md Sec. 5b.3 calls config D.")
    header = f"{'variant':<8}" + "".join(f"{k:>10}" for k in _METRIC_ORDER)
    print(header)
    print("-" * len(header))
    for name in VARIANTS:
        metrics = report["variants"][name]
        print(f"{name:<8}" + "".join(f"{metrics[k]:>10.4f}"
                                     for k in _METRIC_ORDER))

    temps = report["temperatures"]
    values = np.array(list(temps.values()))
    print("\nper-fold temperature: "
          + "  ".join(f"fold{f}={t:.3f}" for f, t in sorted(temps.items())))
    print(f"  mean {values.mean():.3f} +/- {values.std():.3f}  "
          "(a wide spread is itself a finding about video-to-video variation)")
    print("\nNote: +temp's accuracy, macro_f1 and mcc are identical to raw's by"
          "\n      construction -- temperature scaling is monotonic and cannot change"
          "\n      the argmax of any single window."
          "\n"
          "\n      +mc's are NOT: MC averaging can flip predictions, and on an"
          "\n      imbalanced set it tends to flip them toward the majority class."
          "\n      Judge it on macro_f1 and mcc, never on accuracy alone."
          "\n"
          "\n      +temp's AURC, however, is NOT identical, and the difference is an"
          "\n      artifact rather than a selective-prediction gain: risk-coverage"
          "\n      ranks by confidence, which one temperature would preserve exactly,"
          "\n      but each fold gets its own T, so windows re-rank ACROSS fold"
          "\n      boundaries. Read a C-raw/D-temp AURC gap as fold-to-fold"
          "\n      temperature spread, not as better abstention."
          "\n"
          "\n      ECE is secondary: 15 equal-mass bins. Brier and NLL carry the claims.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Config D calibration report over one saved CV run.")
    parser.add_argument("run_file",
                        help="path to outputs/runs/<run_label>/<model>/oof_probs.pt")
    args = parser.parse_args(argv)

    blob = torch.load(args.run_file, weights_only=False)
    report = confidence_report(blob)
    _print_report(report, blob.get("run_label", args.run_file))
    return report


if __name__ == "__main__":
    main()

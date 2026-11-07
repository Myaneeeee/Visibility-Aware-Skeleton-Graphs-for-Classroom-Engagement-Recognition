"""Multi-seed MLP check of the representation findings.

The linear probe is deterministic and the folds do not depend on the seed, so
its numbers cannot be confirmed over seeds. This trains the MLP baseline on each
feature family at seeds 51-60, on the same source-video folds as every model
run, so each family is paired against C's MLP seed for seed.

marg85 is the baseline's own vector, so R-marg85_seed<n> must reproduce
C_seed<n>/mlp; the script checks and reports it.

Usage:  .venv/bin/python -u steps/probe_features.py [--seeds 51 52 ...]
"""
import argparse
import os

import numpy as np
import torch

import _bootstrap  # noqa: F401  -- puts ../lib on sys.path
import common
import evaluate
import paths
import representations


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(51, 61)))
    ap.add_argument("--families", nargs="+", default=list(representations.FEATURE_DIMS),
                    choices=list(representations.FEATURE_DIMS))
    args = ap.parse_args(argv)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    data = common.load_labeled_data(pt_dir=paths.SAV_WINDOWS_FILTERED_DIR,
                                    annotations_csv=paths.ANNOTATIONS_FILE)
    print(f"Loaded {len(data['labels'])} windows | device {device}")

    for seed in args.seeds:
        for family in args.families:
            label = f"R-{family}_seed{seed}"
            if os.path.exists(paths.oof_probs_file(label, "mlp")):
                print(f"SKIP {label} (on disk)")
                continue
            result = evaluate.run_cross_validation(
                data, device=device, model_kind="mlp", seed=seed, features=family)
            evaluate.save_run_outputs(result, label, model="mlp")
            f1 = np.mean([m["macro_f1"] for m in result["fold_metrics"]])
            line = f"DONE {label}  macro_f1 {f1:.4f}"
            if family == "marg85":
                ref = paths.oof_probs_file(f"C_seed{seed}", "mlp")
                if os.path.exists(ref):
                    c = torch.load(ref, map_location="cpu", weights_only=False)
                    diff = np.abs(np.asarray(c["oof_prob"]) - result["oof_prob"]).max()
                    line += f"  | vs C_seed{seed}/mlp max|diff| {diff:.2e}"
            print(line, flush=True)
    print("PROBE DONE")


if __name__ == "__main__":
    main()

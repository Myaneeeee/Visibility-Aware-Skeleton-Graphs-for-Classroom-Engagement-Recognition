"""Apply the quality filter to SAV window .pt files, writing a filtered copy + report."""
import os
import glob
import argparse
import torch

import _bootstrap  # noqa: F401  -- puts ../lib on sys.path
import quality_filter as qf
from paths import SAV_WINDOWS_DIR, SAV_WINDOWS_FILTERED_DIR


def filter_windows(windows):
    kept = []
    for w in windows:
        m = qf.pose_quality_metrics(w["keypoints"], w["bboxes"])
        if qf.is_usable(m):
            kept.append(w)
    stats = {"total": len(windows), "kept": len(kept), "dropped": len(windows) - len(kept)}
    return kept, stats


def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    totals = {"total": 0, "kept": 0, "dropped": 0}
    for pt in sorted(glob.glob(os.path.join(input_dir, "*_windows.pt"))):
        windows = torch.load(pt, weights_only=False)
        kept, stats = filter_windows(windows)
        for k in totals:
            totals[k] += stats[k]
        if kept:
            torch.save(kept, os.path.join(output_dir, os.path.basename(pt)))
        print(f"{os.path.basename(pt)}: kept {stats['kept']}/{stats['total']}")
    keep_rate = totals["kept"] / totals["total"] if totals["total"] else 0.0
    print(f"\nTOTAL kept {totals['kept']}/{totals['total']} ({keep_rate:.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=SAV_WINDOWS_DIR)
    parser.add_argument("--output-dir", default=SAV_WINDOWS_FILTERED_DIR)
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)

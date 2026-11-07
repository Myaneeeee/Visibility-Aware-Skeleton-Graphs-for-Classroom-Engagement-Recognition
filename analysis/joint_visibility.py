"""Per-joint visibility across samples: is the lower body invisible for nearly everyone?

The mean confidence per joint (diagnose_representation.py) could hide a split
population. This reports, per joint, the distribution of each sample's mean
confidence and the share of samples and of source videos below the 0.35
visibility threshold. Writes outputs/diagnostics/joint_visibility.json and the
per-sample matrix joint_visibility_samples.npy (samples x 17), which the
visibility figure reads.

Usage:  .venv/bin/python analysis/joint_visibility.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path[:0] = ["lib", "steps"]
import common  # noqa: E402
import paths  # noqa: E402
from diagnose_representation import COCO17, UNOBSERVED_CONF  # noqa: E402

LOWER = [13, 14, 15, 16]


def main():
    data = common.load_labeled_data(paths.SAV_WINDOWS_FILTERED_DIR, paths.ANNOTATIONS_FILE)
    conf = torch.stack(list(data["tensors"]))[:, 2].numpy()      # (N, T, V)
    per_sample = conf.mean(axis=1)                               # (N, V)
    # video_names holds clip ids ({video}_{clip}); the source video is the first two fields.
    videos = np.array(["_".join(c.split("_")[:2]) for c in data["video_names"]])
    names = sorted(set(videos))
    per_video = np.stack([per_sample[videos == v].mean(0) for v in names])

    rows = []
    print(f"{len(per_sample)} samples, {len(names)} source videos, threshold {UNOBSERVED_CONF}\n")
    print(f"{'joint':<11}{'mean':>7}{'median':>8}{'IQR':>16}"
          f"{'samples < thr':>15}{'videos < thr':>14}")
    for i, name in enumerate(COCO17):
        s = per_sample[:, i]
        q1, med, q3 = np.percentile(s, [25, 50, 75])
        row = {"joint": name, "index": i, "mean": float(conf[:, :, i].mean()),
               "median": float(med), "q1": float(q1), "q3": float(q3),
               "share_samples_below": float((s < UNOBSERVED_CONF).mean()),
               "share_videos_below": float((per_video[:, i] < UNOBSERVED_CONF).mean())}
        rows.append(row)
        print(f"{name:<11}{row['mean']:>7.3f}{med:>8.3f}   [{q1:.3f}, {q3:.3f}]"
              f"{row['share_samples_below']:>14.1%}{row['share_videos_below']:>14.1%}")

    all_lower = (per_sample[:, LOWER] < UNOBSERVED_CONF).all(axis=1)
    any_upper = (per_sample[:, [i for i in range(17) if i not in LOWER]]
                 < UNOBSERVED_CONF).any(axis=1)
    summary = {"samples": len(per_sample), "videos": len(names),
               "threshold": UNOBSERVED_CONF,
               "share_samples_all_lower_below": float(all_lower.mean()),
               "share_samples_any_visible_joint_below": float(any_upper.mean())}
    print(f"\nsamples with all 4 lower-body joints below threshold: {all_lower.mean():.1%}")
    print(f"samples with any of the 13 visible joints below threshold: {any_upper.mean():.1%}")

    # Input level and jitter of the two joint groups after prenormalization: what a
    # pooled or a per-joint input normalization sees (the E7 explanation).
    xyz = torch.stack(list(data["tensors"])).numpy()                # (N, 3, T, V)
    groups = {"visible 13": [i for i in range(17) if i not in LOWER], "invisible 4": LOWER}
    summary["input_by_group"] = {}
    print("\ninput after prenormalization (mean over joints in the group):")
    for name, idx in groups.items():
        x = xyz[:, :, :, idx]
        level = x.mean(axis=(0, 2)).mean(1)
        jitter = x[:, :2].std(axis=2).mean(axis=(0, 2))
        summary["input_by_group"][name] = {"mean_x": float(level[0]), "mean_y": float(level[1]),
                                           "mean_conf": float(level[2]),
                                           "jitter_x": float(jitter[0]), "jitter_y": float(jitter[1])}
        print(f"  {name:12s} mean x {level[0]:+.3f}  mean y {level[1]:+.3f}  "
              f"mean conf {level[2]:.3f}  frame-to-frame jitter x {jitter[0]:.3f}  y {jitter[1]:.3f}")

    np.save(os.path.join(paths.diagnostics_dir(), "joint_visibility_samples.npy"), per_sample)
    out = os.path.join(paths.diagnostics_dir(), "joint_visibility.json")
    with open(out, "w") as f:
        json.dump({"summary": summary, "joints": rows}, f, indent=2)
    print(f"\nwrote {os.path.relpath(out, paths.REPO_ROOT)}")


if __name__ == "__main__":
    main()

"""Emit `clip_list.txt`, the clips to cut from the downloaded source videos.

Takes every clip whose distracted-prior score clears `--threshold`, then adds the
highest-scoring remaining clips as a control arm until they are `1 - high_frac` of
the list. The control arm is what lets `annotation_report.py` measure whether the
prior earned its keep.
"""
import _bootstrap  # noqa: F401

import argparse
import os
from collections import defaultdict

import paths
import sav_actions as sa


def select_clips(clip_person_actions, video_ids, high_frac=0.80,
                 threshold=sa.PRIOR_SCORE_THRESHOLD):
    """Return (clips, stats) for the given source videos, high-prior clips first."""
    wanted = set(video_ids)
    scores = {}
    for clip in clip_person_actions:
        if sa.source_video_id(clip) in wanted:
            scores[clip] = sa.clip_distracted_score(clip, clip_person_actions)

    high = sorted((c for c, s in scores.items() if s >= threshold),
                  key=lambda c: (-scores[c], c))
    low = sorted((c for c, s in scores.items() if s < threshold),
                 key=lambda c: (-scores[c], c))

    n_low = min(len(low), int(round(len(high) * (1 - high_frac) / high_frac)))
    chosen = high + low[:n_low]

    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    stats = {
        "n_high": len(high), "n_low": n_low, "n_total": len(chosen),
        "mean_high": mean([scores[c] for c in high]),
        "mean_low": mean([scores[c] for c in low[:n_low]]),
        "mean_all": mean([scores[c] for c in chosen]),
        "n_videos": len({sa.source_video_id(c) for c in chosen}),
    }
    return chosen, stats


def downloaded_video_ids(source_dir):
    if not os.path.isdir(source_dir):
        return []
    return sorted(os.path.splitext(f)[0]
                  for f in os.listdir(source_dir) if f.endswith(".mp4"))


def main(ann_dir, source_dir, out_path, high_frac, threshold, shortlist_only):
    label_map = sa.load_label_map(os.path.join(ann_dir, "education_first_label.pbtxt"))
    cpa = sa.load_clip_person_actions(
        [os.path.join(ann_dir, "train.csv"), os.path.join(ann_dir, "val.csv")],
        label_map)

    videos = downloaded_video_ids(source_dir)
    if not videos:
        raise SystemExit(f"no .mp4 sources in {source_dir}")

    if shortlist_only:
        shortlist_path = os.path.join(ann_dir, "video_shortlist.txt")
        listed = {l.split("\t")[0].strip() for l in open(shortlist_path)
                  if l.strip() and not l.startswith("#")}
        skipped = [v for v in videos if v not in listed]
        videos = [v for v in videos if v in listed]
        if skipped:
            print(f"ignoring {len(skipped)} on-disk videos not in the shortlist: "
                  f"{', '.join(skipped)}")

    clips, stats = select_clips(cpa, videos, high_frac, threshold)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("\n".join(clips) + "\n")

    per_video = defaultdict(int)
    for c in clips:
        per_video[sa.source_video_id(c)] += 1

    print(f"wrote {out_path}")
    print(f"  source videos    : {stats['n_videos']} of {len(videos)} downloaded")
    print(f"  clips to cut     : {stats['n_total']}")
    print(f"    high-prior     : {stats['n_high']:5d}  mean score {stats['mean_high']:.3f}")
    print(f"    control        : {stats['n_low']:5d}  mean score {stats['mean_low']:.3f}")
    print(f"  worklist meanS   : {stats['mean_all']:.3f} "
          f"(SAV-wide {sa.SAV_WIDE_DISTRACTED_MEAN})")
    print(f"  clips per video  : min {min(per_video.values())} "
          f"median {sorted(per_video.values())[len(per_video) // 2]} "
          f"max {max(per_video.values())}")
    return 0


if __name__ == "__main__":
    ann = os.path.join(paths.SAV_DIR, "annotations")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations-dir", default=ann)
    ap.add_argument("--source-dir", default=paths.SAV_VIDEOS_SOURCE_DIR)
    ap.add_argument("--out", default=paths.SAV_CLIP_LIST_FILE)
    ap.add_argument("--high-frac", type=float, default=0.80)
    ap.add_argument("--threshold", type=float, default=sa.PRIOR_SCORE_THRESHOLD)
    ap.add_argument("--all-on-disk", action="store_true",
                    help="use every downloaded video, not just the shortlist")
    args = ap.parse_args()
    raise SystemExit(main(args.annotations_dir, args.source_dir, args.out,
                          args.high_frac, args.threshold,
                          shortlist_only=not args.all_on_disk))

"""Emit `video_shortlist.txt`, the list of SAV source videos to download.

Ranks source videos by how many of their clips clear the distracted-prior threshold.
"""
import _bootstrap  # noqa: F401

import argparse
import os

import paths
import sav_actions as sa


def build(clip_person_actions, n, threshold, max_per_date):
    """Return [(video_id, n_high, n_clips, mean_score)] for the chosen videos."""
    chosen = sa.select_source_videos(clip_person_actions, n=n, threshold=threshold,
                                     max_per_date=max_per_date)
    stats = {s["video"]: s
             for s in sa.rank_source_videos(clip_person_actions, threshold)}
    return [(v, stats[v]["n_high"], stats[v]["n_clips"], stats[v]["mean_score"])
            for v in chosen]


def write_shortlist(rows, out_path, n, threshold, max_per_date):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(f"# n={n} threshold={threshold} max_per_date={max_per_date}\n")
        fh.write("# ranked by n_high desc, mean_score desc, video_id\n")
        fh.write("# video_id\tn_high\tn_clips\tmean_score\n")
        for vid, n_high, n_clips, mean_score in rows:
            fh.write(f"{vid}\t{n_high}\t{n_clips}\t{mean_score:.3f}\n")
    return out_path


def main(ann_dir, out_path, n, threshold, max_per_date):
    label_map = sa.load_label_map(os.path.join(ann_dir, "education_first_label.pbtxt"))
    cpa = sa.load_clip_person_actions(
        [os.path.join(ann_dir, "train.csv"), os.path.join(ann_dir, "val.csv")],
        label_map)

    rows = build(cpa, n, threshold, max_per_date)
    if len(rows) < n:
        print(f"WARNING: only {len(rows)} videos satisfy the caps (asked for {n}).")

    write_shortlist(rows, out_path, n, threshold, max_per_date)

    print(f"wrote {out_path}")
    print(f"  videos           : {len(rows)}")
    print(f"  annotated clips  : {sum(r[2] for r in rows)}")
    print(f"  high-prior clips : {sum(r[1] for r in rows)} (score >= {threshold})")
    print(f"  mean video score : {sum(r[3] for r in rows) / len(rows):.3f} "
          f"(SAV-wide {sa.SAV_WIDE_DISTRACTED_MEAN})")

    src = paths.SAV_VIDEOS_SOURCE_DIR
    if os.path.isdir(src):
        on_disk = {os.path.splitext(f)[0] for f in os.listdir(src) if f.endswith(".mp4")}
        chosen = {r[0] for r in rows}
        print(f"  already on disk  : {len(chosen & on_disk)}")
        print(f"  to download      : {len(chosen - on_disk)}")
        stale = on_disk - chosen
        if stale:
            print(f"  on disk, not selected ({len(stale)}): {', '.join(sorted(stale))}")
    return 0


if __name__ == "__main__":
    ann = os.path.join(paths.SAV_DIR, "annotations")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations-dir", default=ann)
    ap.add_argument("--out", default=os.path.join(ann, "video_shortlist.txt"))
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--threshold", type=float, default=sa.PRIOR_SCORE_THRESHOLD)
    ap.add_argument("--max-per-date", type=int, default=None)
    args = ap.parse_args()
    raise SystemExit(main(args.annotations_dir, args.out, args.n,
                          args.threshold, args.max_per_date))

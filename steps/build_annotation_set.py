"""Build the annotation worklist and split it across annotators with blind overlap.

Every window goes to exactly one annotator, except an `--overlap` fraction that goes
to all of them. Overlap windows are shuffled into each annotator's list at random
positions, so nobody can tell which of their windows are being cross-checked.

Assignment is spread across source videos rather than assigning whole videos to one
annotator: that keeps annotator identity from being confounded with video, so a
per-annotator bias averages out across CV folds instead of aligning with them.

Each annotator writes to their own ledger, so overlapping worklists never produce
conflicting rows. `agreement.py` measures agreement on the overlap afterwards and
`consolidate_labels.py` merges the ledgers into the training set.
"""
import _bootstrap  # noqa: F401

import argparse
import glob
import os
import random
from collections import Counter, defaultdict

import torch

import common
import paths
import sav_actions as sa


def load_window_ids(windows_dir):
    ids = []
    for pt in sorted(glob.glob(os.path.join(windows_dir, "*_windows.pt"))):
        for w in torch.load(pt, weights_only=False):
            ids.append(common.make_window_id(w))
    return ids


def group_by_source(window_ids):
    groups = defaultdict(list)
    for wid in window_ids:
        groups[sa.source_video_id(sa.clip_key_for_window_id(wid))].append(wid)
    return dict(groups)


def spread_across_videos(window_ids, seed=42):
    """Order windows so consecutive entries come from different source videos."""
    pools = {k: sorted(v) for k, v in group_by_source(window_ids).items()}
    rng = random.Random(seed)
    for lst in pools.values():
        rng.shuffle(lst)
    keys = sorted(pools)
    out, i = [], 0
    while any(pools[k] for k in keys):
        k = keys[i % len(keys)]
        if pools[k]:
            out.append(pools[k].pop())
        i += 1
    return out


def assign(window_ids, n_annotators=3, overlap=0.10, seed=42):
    """Return (per_annotator_lists, overlap_ids).

    The overlap set is drawn spread across videos, then every remaining window is
    dealt round-robin so each annotator's share is also video-spread.
    """
    ordered = spread_across_videos(window_ids, seed)
    n_overlap = int(round(len(ordered) * overlap))
    overlap_ids = ordered[:n_overlap]
    rest = ordered[n_overlap:]

    lists = [list(overlap_ids) for _ in range(n_annotators)]
    for i, wid in enumerate(rest):
        lists[i % n_annotators].append(wid)

    for i, lst in enumerate(lists):
        random.Random(seed + 1000 + i).shuffle(lst)
    return lists, overlap_ids


def prior_summary(window_ids, clip_person_actions, threshold=sa.PRIOR_SCORE_THRESHOLD):
    """Mean distracted-prior score of the high-prior and control halves."""
    high, rest = [], []
    for wid in window_ids:
        ck = sa.clip_key_for_window_id(wid)
        score = sa.clip_distracted_score(ck, clip_person_actions)
        target = high if score >= threshold else rest
        target.append(score)
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return {
        "n_high": len(high), "n_rest": len(rest),
        "mean_high": mean(high), "mean_rest": mean(rest),
        "enrichment": mean(high) / sa.SAV_WIDE_DISTRACTED_MEAN if high else 0.0,
    }


def write_lines(path, lines):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def main(windows_dir, ann_dir, n_annotators, overlap, seed, budget,
         worklist_file, worklists_dir, threshold):
    label_map = sa.load_label_map(os.path.join(ann_dir, "education_first_label.pbtxt"))
    cpa = sa.load_clip_person_actions(
        [os.path.join(ann_dir, "train.csv"), os.path.join(ann_dir, "val.csv")],
        label_map)

    window_ids = load_window_ids(windows_dir)
    if not window_ids:
        raise SystemExit(f"no windows found in {windows_dir}")

    worklist = spread_across_videos(window_ids, seed)
    if budget:
        worklist = worklist[:budget]

    lists, overlap_ids = assign(worklist, n_annotators, overlap, seed)

    write_lines(worklist_file, sorted(worklist))
    for i, lst in enumerate(lists, start=1):
        write_lines(os.path.join(worklists_dir, f"annotator_{i}.txt"), lst)

    s = prior_summary(worklist, cpa, threshold)
    judgements = sum(len(l) for l in lists)
    print(f"worklist: {len(worklist)} windows across "
          f"{len(group_by_source(worklist))} source videos")
    print(f"  high-prior : n={s['n_high']:5d}  mean {s['mean_high']:.1%}  "
          f"-> {s['enrichment']:.2f}x vs the {sa.SAV_WIDE_DISTRACTED_MEAN:.1%} baseline")
    print(f"  control    : n={s['n_rest']:5d}  mean {s['mean_rest']:.1%}")
    print(f"\nblind overlap: {len(overlap_ids)} windows ({overlap:.0%}) in every list")
    for i, lst in enumerate(lists, start=1):
        vids = len(group_by_source(lst))
        print(f"  annotator {i}: {len(lst):5d} windows across {vids} videos "
              f"-> {os.path.join(worklists_dir, f'annotator_{i}.txt')}")
    print(f"\ntotal judgements: {judgements} "
          f"({judgements / n_annotators:.0f} each, ~{judgements / n_annotators / 110:.1f} h)")

    per_video = Counter(sa.source_video_id(sa.clip_key_for_window_id(w))
                        for w in worklist)
    print(f"windows per video: min {min(per_video.values())} "
          f"median {sorted(per_video.values())[len(per_video) // 2]} "
          f"max {max(per_video.values())}")
    return 0


if __name__ == "__main__":
    ann = os.path.join(paths.SAV_DIR, "annotations")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows-dir", default=paths.SAV_WINDOWS_FILTERED_DIR)
    ap.add_argument("--annotations-dir", default=ann)
    ap.add_argument("--annotators", type=int, default=3)
    ap.add_argument("--overlap", type=float, default=0.10,
                    help="fraction labeled by every annotator (default %(default)s)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget", type=int, default=None,
                    help="cap the worklist; default is every usable window")
    ap.add_argument("--worklist", default=paths.SAV_WORKLIST_FILE)
    ap.add_argument("--worklists-dir", default=paths.SAV_WORKLISTS_DIR)
    ap.add_argument("--prior-threshold", type=float, default=sa.PRIOR_SCORE_THRESHOLD)
    args = ap.parse_args()
    raise SystemExit(main(args.windows_dir, args.annotations_dir, args.annotators,
                          args.overlap, args.seed, args.budget, args.worklist,
                          args.worklists_dir, args.prior_threshold))

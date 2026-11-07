"""Post-labeling QA: class balance, per-video coverage, prior hit-rate, and
inter-annotator agreement (Cohen's kappa)."""
import _bootstrap  # noqa: F401  -- puts ../lib on sys.path
import os
import csv as _csv
import argparse
from collections import Counter

import cv_utils
import sav_actions as sa
import paths

_VALID = {"Focused", "Distracted", "TRASH"}
_TRAINABLE = {"Focused", "Distracted"}
# Deliberately outside _VALID: a Skip means "I could not tell", which is neither a
# class nor a disagreement. Counting it as a third category would drag kappa toward
# zero and misreport real agreement. It still has to be *findable* -- Phase C of
# LABELING_GUIDELINES adjudicates every one -- hence load_skipped below.
SKIP_LABEL = "Skip"


def load_labels(csv_path):
    labels = {}
    with open(csv_path, newline="") as f:
        reader = _csv.reader(f)
        next(reader, None)                       # skip header
        for row in reader:
            if len(row) == 2 and row[1] in _VALID:
                labels[row[0]] = row[1]
    return labels


def load_skipped(csv_path):
    """Window ids the annotator declined to decide, in file order -- the
    adjudication worklist."""
    skipped = []
    with open(csv_path, newline="") as f:
        reader = _csv.reader(f)
        next(reader, None)                       # skip header
        for row in reader:
            if len(row) == 2 and row[1] == SKIP_LABEL:
                skipped.append(row[0])
    return skipped


def summarize(labels):
    counts = Counter(labels.values())
    videos = {sa.source_video_id(sa.clip_key_for_window_id(w)) for w in labels}
    return {"counts": dict(counts), "total": len(labels), "source_videos": len(videos)}


def _distracted_rate(wids, labels):
    graded = [labels[w] for w in wids if labels.get(w) in ("Focused", "Distracted")]
    if not graded:
        return 0.0
    return sum(1 for x in graded if x == "Distracted") / len(graded)


def prior_hit_rate(labels, clip_person_actions, threshold=sa.PRIOR_SCORE_THRESHOLD):
    """Did the sampling prior earn its keep? Compares the human-labeled Distracted
    rate inside high-prior clips against everything else. If these two numbers are
    equal, the prior bought nothing -- report that honestly either way."""
    prior, rest = [], []
    for w in labels:
        ck = sa.clip_key_for_window_id(w)
        is_prior = sa.clip_is_distracted_prior(ck, clip_person_actions, threshold)
        (prior if is_prior else rest).append(w)
    return {"distracted_rate_in_prior": _distracted_rate(prior, labels),
            "distracted_rate_in_rest": _distracted_rate(rest, labels),
            "n_prior": len(prior), "n_rest": len(rest)}


def cv_group_report(labels, k=5):
    """Preview the GroupKFold the sweep will run: group sizes and per-fold balance.

    Trainable windows only. A video holding nothing but TRASH contributes no
    training rows, so it is not a CV group. Keys on `cv_utils.cv_group_key`, the
    same function `evaluate.py` splits on, so this is the real grouping and not a
    lookalike.
    """
    trainable = {w: l for w, l in labels.items() if l in _TRAINABLE}
    wids = sorted(trainable)
    groups = [cv_utils.cv_group_key(sa.clip_key_for_window_id(w)) for w in wids]
    sizes = Counter(groups)
    with_distracted = {g for g, w in zip(groups, wids) if trainable[w] == "Distracted"}

    # One group is not a cross-video split, and sklearn rejects n_splits=1 outright.
    folds = []
    if len(sizes) >= 2:
        for _, te in cv_utils.group_kfold_indices(groups, k=k):
            folds.append({
                "n": len(te),
                "videos": len({groups[i] for i in te}),
                "distracted": sum(1 for i in te if trainable[wids[i]] == "Distracted"),
            })
    return {"n_trainable": len(wids),
            "n_groups": len(sizes),
            "sizes": sorted(sizes.values()),
            "groups_without_distracted": len(sizes) - len(with_distracted),
            "folds": folds}


def cohen_kappa(labels_a, labels_b):
    keys = sorted(set(labels_a) & set(labels_b))
    if not keys:
        return float("nan")
    a = [labels_a[k] for k in keys]
    b = [labels_b[k] for k in keys]
    cats = sorted(set(a) | set(b))
    n = len(keys)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return 1.0 if pe == 1.0 else (po - pe) / (1.0 - pe)


def main(annotations_csv=None, train_csv=None, val_csv=None, pbtxt=None,
         annotations_b=None, list_skipped=False, folds=5):
    annotations_csv = annotations_csv or paths.ANNOTATIONS_FILE
    labels = load_labels(annotations_csv)
    skipped = load_skipped(annotations_csv)
    s = summarize(labels)
    print(f"labeled: {s['total']} windows across {s['source_videos']} source videos")
    print(f"  counts: {s['counts']}")
    if skipped:
        print(f"  {len(skipped)} skipped -- these need adjudication "
              f"(--list-skipped to dump the ids)")
        if list_skipped:
            for wid in skipped:
                print(f"    {wid}")
    if pbtxt and (train_csv or val_csv):
        label_map = sa.load_label_map(pbtxt)
        clip_actions = sa.load_clip_person_actions(
            [c for c in (train_csv, val_csv) if c], label_map)
        r = prior_hit_rate(labels, clip_actions)
        print(f"  distracted rate in prior clips: {r['distracted_rate_in_prior']:.2%} "
              f"(n={r['n_prior']})")
        print(f"  distracted rate in rest clips:  {r['distracted_rate_in_rest']:.2%} "
              f"(n={r['n_rest']})")
        lift = r['distracted_rate_in_prior'] - r['distracted_rate_in_rest']
        print(f"  prior lift: {lift:+.2%} "
              f"({'the prior earned its keep' if lift > 0.05 else 'the prior bought little -- report this'})")
    try:
        cg = cv_group_report(labels, k=folds)
    except ValueError as exc:
        print(f"\nCV grouping: SKIPPED -- {exc}")
        cg = {"n_groups": 0}
    if cg["n_groups"] and cg["folds"]:
        sizes = cg["sizes"]
        print(f"\nCV grouping (trainable only): {cg['n_trainable']} windows across "
              f"{cg['n_groups']} source videos")
        print(f"  windows per video: min {sizes[0]}, "
              f"median {sizes[len(sizes) // 2]}, max {sizes[-1]}")
        print(f"  videos with zero Distracted: {cg['groups_without_distracted']} "
              f"of {cg['n_groups']}")
        print(f"  GroupKFold({len(cg['folds'])}) class balance:")
        for i, f in enumerate(cg["folds"]):
            rate = f["distracted"] / f["n"] if f["n"] else 0.0
            print(f"    fold {i}: n={f['n']:5d}  videos={f['videos']:4d}  "
                  f"Distracted={f['distracted']:4d} ({rate:.1%})")

    if annotations_b:
        kb = load_labels(annotations_b)
        k = cohen_kappa(labels, kb)
        print(f"  inter-annotator kappa (vs {annotations_b}): {k:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", default=paths.ANNOTATIONS_FILE)
    ap.add_argument("--train-csv", default=os.path.join(paths.SAV_DIR, "annotations", "train.csv"))
    ap.add_argument("--val-csv", default=os.path.join(paths.SAV_DIR, "annotations", "val.csv"))
    ap.add_argument("--pbtxt", default=os.path.join(paths.SAV_DIR, "annotations", "education_first_label.pbtxt"))
    ap.add_argument("--annotations-b", default=None)
    ap.add_argument("--folds", type=int, default=5,
                    help="k for the GroupKFold class-balance preview")
    ap.add_argument("--list-skipped", action="store_true",
                    help="dump every skipped window id (the adjudication worklist)")
    args = ap.parse_args()
    main(args.annotations, args.train_csv, args.val_csv, args.pbtxt,
         args.annotations_b, args.list_skipped, args.folds)

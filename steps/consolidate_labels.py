"""Merge the per-annotator ledgers into the training set.

Resolution per window:

  single judgement          -> that label
  unanimous on the overlap  -> that label
  majority (e.g. 2 of 3)    -> the majority label, recorded as `majority`
  tie, or anyone said Skip  -> withheld, written to the adjudication queue

Writes `datasets/annotations.csv` (window_id,label) for training, plus a provenance
CSV recording how every row was resolved and an adjudication queue of the windows a
human still has to settle.
"""
import _bootstrap  # noqa: F401

import argparse
import csv
import glob
import os
from collections import Counter, defaultdict

import paths

VALID = {"Focused", "Distracted", "TRASH"}
SKIP_LABEL = "Skip"
TRAINABLE = {"Focused", "Distracted"}


def load_ledger(path):
    """{window_id: label}, including Skip rows so they can block consolidation."""
    out = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header and header[0] != "window_id":
            f.seek(0)
            reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2 and (row[1] in VALID or row[1] == SKIP_LABEL):
                out[row[0]] = row[1]
    return out


def resolve(judgements):
    """judgements: {annotator: label}. Returns (label_or_None, how)."""
    labels = list(judgements.values())
    if any(l == SKIP_LABEL for l in labels):
        return None, "skip"
    if len(labels) == 1:
        return labels[0], "single"
    counts = Counter(labels)
    top, n_top = counts.most_common(1)[0]
    if n_top == len(labels):
        return top, "unanimous"
    if sum(1 for c in counts.values() if c == n_top) > 1:
        return None, "tie"
    return top, "majority"


def consolidate(ledgers):
    """ledgers: {annotator: {window_id: label}}. Returns (rows, queue)."""
    by_window = defaultdict(dict)
    for name, lab in ledgers.items():
        for wid, label in lab.items():
            by_window[wid][name] = label

    rows, queue = [], []
    for wid in sorted(by_window):
        judgements = by_window[wid]
        label, how = resolve(judgements)
        if label is None:
            queue.append({"window_id": wid, "reason": how,
                          **{f"annotator_{n}": l for n, l in sorted(judgements.items())}})
        else:
            rows.append({"window_id": wid, "label": label, "resolution": how,
                         "n_judgements": len(judgements),
                         "judgements": ";".join(f"{n}={l}"
                                                for n, l in sorted(judgements.items()))})
    return rows, queue


def write_annotations(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window_id", "label"])
        for r in rows:
            w.writerow([r["window_id"], r["label"]])


def write_csv(rows, path, fields):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main(ledgers_dir, explicit, out_file, provenance_file, queue_file,
         adjudication_file=paths.SAV_ADJUDICATION_LABELS_FILE):
    paths_ = list(explicit) if explicit else sorted(glob.glob(
        os.path.join(ledgers_dir, "*.csv")))
    if not paths_:
        raise SystemExit(f"no ledgers found in {ledgers_dir}")
    if not explicit and adjudication_file and os.path.exists(adjudication_file):
        paths_.append(adjudication_file)

    ledgers = {os.path.splitext(os.path.basename(p))[0]: load_ledger(p) for p in paths_}
    for name, lab in ledgers.items():
        print(f"{name}: {len(lab)} rows  {dict(Counter(lab.values()))}")

    rows, queue = consolidate(ledgers)
    write_annotations(rows, out_file)
    write_csv(rows, provenance_file,
              ["window_id", "label", "resolution", "n_judgements", "judgements"])

    queue_fields = ["window_id", "reason"] + [f"annotator_{n}" for n in sorted(ledgers)]
    write_csv(queue, queue_file, queue_fields)

    how = Counter(r["resolution"] for r in rows)
    labels = Counter(r["label"] for r in rows)
    trainable = sum(v for k, v in labels.items() if k in TRAINABLE)

    print(f"\nconsolidated {len(rows)} windows -> {out_file}")
    for k in ("single", "unanimous", "majority"):
        if how[k]:
            print(f"  {k:<10}: {how[k]}")
    print(f"\nlabels: {dict(labels)}")
    if trainable:
        n_dist = labels.get("Distracted", 0)
        print(f"trainable: {trainable}  ({n_dist} Distracted, "
              f"{n_dist / trainable:.1%})")
    if queue:
        reasons = Counter(q["reason"] for q in queue)
        print(f"\nadjudication queue: {len(queue)} windows -> {queue_file}")
        for r, c in sorted(reasons.items()):
            print(f"  {r}: {c}")
        print("Settle these with steps/annotate.py, then re-run this script.")
    else:
        print("\nadjudication queue: empty")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledgers-dir", default=paths.SAV_LEDGERS_DIR)
    ap.add_argument("--ledger", action="append", default=None)
    ap.add_argument("--out", default=paths.ANNOTATIONS_FILE)
    ap.add_argument("--provenance", default=os.path.join(
        paths.SAV_DIR, "label_provenance.csv"))
    ap.add_argument("--queue", default=paths.SAV_ADJUDICATION_FILE)
    ap.add_argument("--adjudication", default=paths.SAV_ADJUDICATION_LABELS_FILE,
                    help="human verdicts on queued windows, read alongside the ledgers")
    args = ap.parse_args()
    raise SystemExit(main(args.ledgers_dir, args.ledger, args.out,
                          args.provenance, args.queue, args.adjudication))

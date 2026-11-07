"""Measure inter-annotator agreement on the blind overlap.

Finds every window judged by two or more annotators and reports pairwise Cohen's
kappa, three-class (Focused/Distracted/TRASH) and binary (Focused/Distracted only).
Skip rows are excluded: a Skip means "I could not tell", so scoring it as a category
would drag kappa toward agreement on indecision.

The overlap is discovered from the ledgers, not from a list, so this works whatever
`build_annotation_set.py --overlap` was set to.
"""
import _bootstrap  # noqa: F401

import argparse
import csv
import glob
import itertools
import os
from collections import Counter

VALID = {"Focused", "Distracted", "TRASH"}
BINARY = {"Focused", "Distracted"}
SKIP_LABEL = "Skip"


def load_ledger(path):
    """{window_id: label} for rows carrying a real judgement."""
    out = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header and header[0] != "window_id":
            f.seek(0)
            reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2 and row[1] in VALID:
                out[row[0]] = row[1]
    return out


def cohen_kappa(labels_a, labels_b, restrict=None):
    """Cohen's kappa over the windows both annotators judged.

    `restrict` limits the label set (pass BINARY to drop TRASH). Returns (kappa,
    n, percent_agreement); kappa is nan when fewer than 2 windows overlap.
    """
    keys = sorted(set(labels_a) & set(labels_b))
    if restrict:
        keys = [k for k in keys
                if labels_a[k] in restrict and labels_b[k] in restrict]
    n = len(keys)
    if n < 2:
        return float("nan"), n, float("nan")
    a = [labels_a[k] for k in keys]
    b = [labels_b[k] for k in keys]
    cats = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    kappa = 1.0 if pe == 1.0 else (po - pe) / (1.0 - pe)
    return kappa, n, po


def find_ledgers(ledgers_dir, explicit):
    if explicit:
        return list(explicit)
    return sorted(glob.glob(os.path.join(ledgers_dir, "*.csv")))


def disagreements(ledgers, window_id):
    return {name: lab for name, lab in
            ((n, l.get(window_id)) for n, l in ledgers.items()) if lab}


def main(ledgers_dir, explicit, gate, show_disagreements):
    paths_ = find_ledgers(ledgers_dir, explicit)
    if len(paths_) < 2:
        raise SystemExit(f"need at least 2 ledgers, found {len(paths_)} in {ledgers_dir}")

    ledgers = {os.path.splitext(os.path.basename(p))[0]: load_ledger(p) for p in paths_}
    for name, lab in ledgers.items():
        print(f"{name}: {len(lab)} judged  {dict(Counter(lab.values()))}")

    counts = Counter()
    for lab in ledgers.values():
        counts.update(lab.keys())
    overlap = [w for w, c in counts.items() if c >= 2]
    all_three = [w for w, c in counts.items() if c >= len(ledgers)]
    print(f"\noverlap: {len(overlap)} windows judged by >=2 annotators, "
          f"{len(all_three)} by all {len(ledgers)}")
    if not overlap:
        raise SystemExit("no overlap between ledgers -- nothing to measure")

    print(f"\n{'pair':<28} {'kappa(3)':>9} {'n':>5} {'agree':>7} "
          f"{'kappa(bin)':>11} {'n':>5}")
    failures = []
    for a, b in itertools.combinations(sorted(ledgers), 2):
        k3, n3, po = cohen_kappa(ledgers[a], ledgers[b])
        kb, nb, _ = cohen_kappa(ledgers[a], ledgers[b], restrict=BINARY)
        print(f"{a + ' vs ' + b:<28} {k3:>9.3f} {n3:>5} {po:>6.0%} "
              f"{kb:>11.3f} {nb:>5}")
        if k3 == k3 and k3 < gate:
            failures.append((a, b, k3))

    if show_disagreements:
        print("\ndisagreements on the overlap:")
        for w in sorted(overlap):
            judged = disagreements(ledgers, w)
            if len(set(judged.values())) > 1:
                print(f"  {w}  " + "  ".join(f"{n}={l}" for n, l in sorted(judged.items())))

    print()
    if failures:
        for a, b, k in failures:
            print(f"GATE FAILED: {a} vs {b} kappa={k:.3f} < {gate}")
        print("Run the shared criteria briefing and re-label before mass labeling.")
        return 1
    print(f"gate met: every pair clears kappa >= {gate} (n={len(overlap)} overlap windows)")
    return 0


if __name__ == "__main__":
    import paths
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledgers-dir", default=paths.SAV_LEDGERS_DIR)
    ap.add_argument("--ledger", action="append", default=None,
                    help="explicit ledger path; repeatable")
    ap.add_argument("--gate", type=float, default=0.6)
    ap.add_argument("--show-disagreements", action="store_true")
    args = ap.parse_args()
    raise SystemExit(main(args.ledgers_dir, args.ledger, args.gate,
                          args.show_disagreements))

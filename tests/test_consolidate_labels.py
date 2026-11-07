import csv

import agreement
import consolidate_labels as cl


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window_id", "label"])
        w.writerows(rows)
    return str(path)


class TestResolve:
    def test_single_judgement_is_taken_as_is(self):
        assert cl.resolve({"a": "Focused"}) == ("Focused", "single")

    def test_unanimous_agreement(self):
        assert cl.resolve({"a": "Distracted", "b": "Distracted",
                           "c": "Distracted"}) == ("Distracted", "unanimous")

    def test_majority_wins_and_is_flagged(self):
        assert cl.resolve({"a": "Focused", "b": "Focused",
                           "c": "TRASH"}) == ("Focused", "majority")

    def test_three_way_split_goes_to_adjudication(self):
        assert cl.resolve({"a": "Focused", "b": "Distracted",
                           "c": "TRASH"}) == (None, "tie")

    def test_two_way_tie_goes_to_adjudication(self):
        assert cl.resolve({"a": "Focused", "b": "Distracted"}) == (None, "tie")

    def test_any_skip_blocks_consolidation(self):
        """A Skip means 'I could not tell'; it must not be outvoted silently."""
        assert cl.resolve({"a": "Focused", "b": "Focused",
                           "c": "Skip"}) == (None, "skip")


def test_consolidate_splits_rows_and_queue():
    ledgers = {
        "annotator_1": {"w1": "Focused", "w2": "Distracted", "w3": "Focused"},
        "annotator_2": {"w1": "Focused", "w2": "Focused"},
        "annotator_3": {"w1": "TRASH", "w4": "Distracted"},
    }
    rows, queue = cl.consolidate(ledgers)
    by_id = {r["window_id"]: r for r in rows}

    assert by_id["w1"]["label"] == "Focused"
    assert by_id["w1"]["resolution"] == "majority"
    assert by_id["w3"]["resolution"] == "single"
    assert by_id["w4"]["label"] == "Distracted"
    assert [q["window_id"] for q in queue] == ["w2"]
    assert queue[0]["reason"] == "tie"


def test_consolidate_records_provenance():
    ledgers = {"a": {"w1": "Focused"}, "b": {"w1": "Focused"}}
    rows, _ = cl.consolidate(ledgers)
    assert rows[0]["n_judgements"] == 2
    assert rows[0]["judgements"] == "a=Focused;b=Focused"


def test_written_annotations_carry_only_id_and_label(tmp_path):
    ledgers = {"a": {"w1": "Focused", "w2": "TRASH"}}
    rows, _ = cl.consolidate(ledgers)
    out = tmp_path / "annotations.csv"
    cl.write_annotations(rows, str(out))
    got = list(csv.reader(open(out)))
    assert got[0] == ["window_id", "label"]
    assert all(len(r) == 2 for r in got[1:])


class TestAgreement:
    def test_kappa_is_one_on_perfect_agreement(self):
        a = {"w1": "Focused", "w2": "Distracted", "w3": "TRASH"}
        k, n, po = agreement.cohen_kappa(a, dict(a))
        assert k == 1.0 and n == 3 and po == 1.0

    def test_kappa_uses_only_shared_windows(self):
        a = {"w1": "Focused", "w2": "Distracted"}
        b = {"w1": "Focused", "w9": "TRASH"}
        _, n, _ = agreement.cohen_kappa(a, b)
        assert n == 1

    def test_binary_restriction_drops_trash(self):
        a = {"w1": "Focused", "w2": "TRASH", "w3": "Distracted"}
        b = {"w1": "Focused", "w2": "Focused", "w3": "Distracted"}
        _, n_all, _ = agreement.cohen_kappa(a, b)
        _, n_bin, _ = agreement.cohen_kappa(a, b, restrict=agreement.BINARY)
        assert n_all == 3 and n_bin == 2

    def test_skip_rows_are_not_loaded_as_judgements(self, tmp_path):
        p = _write(tmp_path / "a.csv",
                   [["w1", "Focused"], ["w2", "Skip"], ["w3", "TRASH"]])
        assert agreement.load_ledger(p) == {"w1": "Focused", "w3": "TRASH"}

    def test_consolidation_does_load_skip_rows(self, tmp_path):
        """agreement.py ignores Skip; consolidate must see it to block the row."""
        p = _write(tmp_path / "a.csv", [["w1", "Focused"], ["w2", "Skip"]])
        assert cl.load_ledger(p)["w2"] == "Skip"


def test_main_settles_ties_from_the_adjudication_file(tmp_path):
    ledgers = tmp_path / "ledgers"
    ledgers.mkdir()
    _write(ledgers / "annotator_1.csv", [("w1", "Focused"), ("w2", "TRASH")])
    _write(ledgers / "annotator_2.csv", [("w1", "Distracted")])
    _write(ledgers / "annotator_3.csv", [("w1", "TRASH")])
    adjudication = _write(tmp_path / "adjudication_labels.csv", [("w1", "Distracted")])
    out = tmp_path / "annotations.csv"

    cl.main(str(ledgers), None, str(out), str(tmp_path / "prov.csv"),
            str(tmp_path / "queue.csv"), adjudication)

    assert list(csv.reader(open(out)))[1:] == [["w1", "Distracted"], ["w2", "TRASH"]]

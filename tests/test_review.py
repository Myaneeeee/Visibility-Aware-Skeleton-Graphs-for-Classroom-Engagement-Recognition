import csv

import pytest

import review


def _w(clip, track, end=53):
    return {"video_name": clip, "track_id": track, "start_frame": 0, "end_frame": end}


def _index():
    windows = [_w("v_01_1", 1), _w("v_01_1", 2), _w("v_01_2", 3), _w("v_01_2", 4)]
    filtered = {"v_01_1_id_1_f0_f53", "v_01_1_id_2_f0_f53", "v_01_2_id_3_f0_f53"}
    clips = ["v_01_1", "v_01_2", "v_01_3"]
    labels = {"v_01_1_id_1_f0_f53": "Focused", "v_01_2_id_3_f0_f53": "TRASH",
              "v_01_9_id_7_f0_f53": "Distracted"}
    excluded = {"v_01_2_id_3_f0_f53"}
    return review.build_index(windows, filtered, clips, labels, excluded)


def test_index_covers_every_window_label_and_clip():
    items = {i["id"]: i for i in _index()}
    assert items["v_01_1_id_1_f0_f53"]["kind"] == "used"
    assert items["v_01_1_id_1_f0_f53"]["label"] == "Focused"
    assert items["v_01_1_id_2_f0_f53"]["kind"] == "used"
    assert items["v_01_1_id_2_f0_f53"]["label"] is None
    assert items["v_01_2_id_4_f0_f53"]["kind"] == "filtered_out"
    assert items["v_01_9_id_7_f0_f53"]["kind"] == "missing"
    assert items["v_01_9_id_7_f0_f53"]["clip"] == "v_01_9"
    assert items["v_01_3"]["kind"] == "no_window"
    assert items["v_01_2_id_3_f0_f53"]["excluded"] is True
    assert len(items) == 6


def test_index_is_sorted_by_clip_then_track():
    ids = [i["id"] for i in _index()]
    assert ids == sorted(ids, key=lambda x: (x.split("_id_")[0], x))


@pytest.mark.parametrize("name,ok", [("annotator_1", True), ("panel", True),
                                     ("../x", False), ("", False), ("a b", False)])
def test_reviewer_name_validation(name, ok):
    assert review.valid_reviewer(name) is ok


def test_reviews_append_and_latest_wins(tmp_path):
    review.append_review(tmp_path, "panel", "a_id_1_f0_f53", "Focused")
    review.append_review(tmp_path, "panel", "a_id_1_f0_f53", "Distracted")
    review.append_review(tmp_path, "annotator_2", "a_id_1_f0_f53", "TRASH")
    got = review.read_reviews(tmp_path)
    assert got == {"panel": {"a_id_1_f0_f53": "Distracted"},
                   "annotator_2": {"a_id_1_f0_f53": "TRASH"}}


def test_append_rejects_bad_label_and_reviewer(tmp_path):
    with pytest.raises(ValueError):
        review.append_review(tmp_path, "panel", "a", "Bored")
    with pytest.raises(ValueError):
        review.append_review(tmp_path, "../etc", "a", "Focused")


def test_plan_apply_changes_adds_and_skips():
    labels = {"a": "Focused", "b": "TRASH", "c": "Distracted"}
    decisions = {"a": "Focused", "b": "Distracted", "d": "Focused",
                 "e": "Distracted", "c": review.SKIP}
    changes, skipped = review.plan_apply(labels, decisions, used_ids={"a", "b", "c", "d"})
    assert changes == [("b", "TRASH", "Distracted"), ("d", None, "Focused")]
    assert ("e", "window not used in training") in skipped
    assert ("c", "Skip") in skipped


def test_apply_rewrites_in_place_and_logs(tmp_path):
    ann = tmp_path / "annotations.csv"
    with open(ann, "w", newline="") as f:
        csv.writer(f).writerows([["window_id", "label"], ["a", "Focused"], ["b", "TRASH"]])
    log = tmp_path / "applied.csv"
    review.apply_changes(ann, [("b", "TRASH", "Distracted"), ("d", None, "Focused")], log, "panel")
    with open(ann, newline="") as f:
        rows = list(csv.reader(f))
    assert rows == [["window_id", "label"], ["a", "Focused"], ["b", "Distracted"], ["d", "Focused"]]
    with open(log, newline="") as f:
        logged = list(csv.DictReader(f))
    assert [(r["window_id"], r["old_label"], r["new_label"], r["reviewer"]) for r in logged] == [
        ("b", "TRASH", "Distracted", "panel"), ("d", "", "Focused", "panel")]


def test_suspicion_ranks_disagreement_and_low_confidence_within_each_label():
    labels = {"f1": "Focused", "f2": "Focused", "f3": "Focused", "d1": "Distracted", "d2": "Distracted"}
    p_dis = {"f1": 0.9, "f2": 0.1, "f3": 0.1, "d1": 0.9, "d2": 0.2}
    conf = {"f1": 0.8, "f2": 0.3, "f3": 0.8, "d1": 0.8, "d2": 0.8}
    s = review.suspicion_scores(labels, p_dis, conf)
    assert s["f1"]["p_other"] == pytest.approx(0.9)
    assert s["d2"]["p_other"] == pytest.approx(0.8)
    assert s["f3"]["score"] < s["f1"]["score"]
    assert s["f3"]["score"] < s["f2"]["score"]
    assert s["d2"]["score"] > s["d1"]["score"]


def test_suspicion_without_prediction_is_neutral():
    s = review.suspicion_scores({"a": "Focused", "b": "Focused"}, {"a": 0.9}, {"a": 0.5, "b": 0.5})
    assert s["b"]["p_other"] is None
    assert 0 < s["b"]["score"] < s["a"]["score"]


@pytest.mark.parametrize("header,expected", [
    (None, None),
    ("bytes=0-", (0, 999)),
    ("bytes=100-199", (100, 199)),
    ("bytes=900-5000", (900, 999)),
    ("bytes=-100", (900, 999)),
    ("bytes=2000-", "invalid"),
    ("items=0-1", "invalid"),
])
def test_parse_range(header, expected):
    assert review.parse_range(header, 1000) == expected


import common


def test_make_window_id_matches_legacy_format():
    w = {"video_name": "vidA", "track_id": 7, "start_frame": 45, "end_frame": 134}
    assert common.make_window_id(w) == "vidA_id_7_f45_f134"


def test_class_maps_are_binary():
    assert common.CLASS_NAMES == ["Focused", "Distracted"]
    assert common.CLASS_TO_IDX == {"Focused": 0, "Distracted": 1}


def test_load_labeled_data_applies_prenormalize(synthetic_dataset):
    d = common.load_labeled_data(
        pt_dir=synthetic_dataset["pt_dir"],
        annotations_csv=synthetic_dataset["annotations_csv"],
    )
    t = d["tensors"][0]
    assert t.shape == (3, 90, 17)
    # confident-joint x,y are bounded to ~[-1, 1] by prenormalize_2d
    conf = t[2] > 0.01
    assert float(t[0][conf].abs().max()) <= 1.0 + 1e-4
    assert float(t[1][conf].abs().max()) <= 1.0 + 1e-4


def test_normalize_window_is_removed():
    assert not hasattr(common, "normalize_window")

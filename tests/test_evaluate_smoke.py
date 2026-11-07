import common
import cv_utils
import evaluate


def test_run_cross_validation_produces_oof(synthetic_dataset):
    data = common.load_labeled_data(
        pt_dir=synthetic_dataset["pt_dir"],
        annotations_csv=synthetic_dataset["annotations_csv"],
    )
    result = evaluate.run_cross_validation(
        data, k=2, epochs=1, device="cpu", model_kind="stgcn")
    n = len(data["labels"])
    assert result["oof_pred"].shape[0] == n
    assert result["oof_prob"].shape == (n, 2)
    assert result["oof_true"].shape[0] == n
    assert len(result["order_window_ids"]) == n
    # OOF predictions must cover every sample exactly once (via disjoint test folds)
    assert sorted(result["order_window_ids"]) == sorted(data["window_ids"])


def test_cross_validation_holds_out_whole_videos(synthetic_dataset):
    """No source video may be tested by more than one fold.

    The fixture's first two clips come from one recording at adjacent indices, so
    they overlap by 1s of frames. Grouping by clip id would split them across the
    train/test boundary and make the metric near-duplicate matching.
    """
    data = common.load_labeled_data(
        pt_dir=synthetic_dataset["pt_dir"],
        annotations_csv=synthetic_dataset["annotations_csv"],
    )
    result = evaluate.run_cross_validation(
        data, k=2, epochs=1, device="cpu", model_kind="stgcn")

    videos = [cv_utils.cv_group_key(v) for v in data["video_names"]]
    assert len(set(videos)) == 2

    fold_of = result["fold_of"]
    assert len(fold_of) == len(videos)
    for video in set(videos):
        folds_touching = {fold_of[i] for i, v in enumerate(videos) if v == video}
        assert len(folds_touching) == 1, (
            f"video {video} was tested by folds {sorted(folds_touching)}; "
            "its windows are split across the train/test boundary")

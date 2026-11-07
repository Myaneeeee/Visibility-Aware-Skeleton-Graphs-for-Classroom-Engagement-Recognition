import numpy as np
import pytest

import cv_utils


def test_no_video_overlap_between_train_and_test():
    videos = np.array(["vidA"] * 10 + ["vidB"] * 10 + ["vidC"] * 10
                      + ["vidD"] * 10 + ["vidE"] * 10)
    folds = cv_utils.group_kfold_indices(videos, k=5)
    assert len(folds) == 5
    for train_idx, test_idx in folds:
        assert set(videos[train_idx]).isdisjoint(set(videos[test_idx]))


def test_every_sample_tested_once():
    videos = np.array(["vidA"] * 5 + ["vidB"] * 5 + ["vidC"] * 5)
    folds = cv_utils.group_kfold_indices(videos, k=3)
    tested = np.concatenate([test_idx for _, test_idx in folds])
    assert sorted(tested.tolist()) == list(range(len(videos)))


def test_cv_group_key_reduces_a_clip_id_to_its_source_video():
    assert cv_utils.cv_group_key("20230118_34_136") == "20230118_34"


def test_cv_group_key_is_idempotent_on_a_source_id():
    """Must not strip the second field: that would collapse every video recorded on
    one date into a single group."""
    assert cv_utils.cv_group_key("20230118_34") == "20230118_34"


def test_cv_group_key_separates_videos_sharing_the_nn_field():
    """20230116_44 and 20220716_44 are different schools. The NN field is not a
    room, so it must never merge two recordings."""
    assert (cv_utils.cv_group_key("20230116_44_10")
            != cv_utils.cv_group_key("20220716_44_10"))


def test_cv_group_key_separates_videos_sharing_a_date():
    assert (cv_utils.cv_group_key("20230118_34_1")
            != cv_utils.cv_group_key("20230118_21_1"))


def test_clips_from_one_video_never_straddle_a_fold():
    video_names = ["20221208_08_1", "20221208_08_2", "20221208_08_3",
                   "20230414_08_1", "20230414_08_2",
                   "20230101_21_1", "20230101_21_2",
                   "20230202_33_1", "20230202_33_2"]
    groups = [cv_utils.cv_group_key(v) for v in video_names]
    for train_idx, test_idx in cv_utils.group_kfold_indices(groups, k=3):
        assert not ({groups[i] for i in train_idx} & {groups[i] for i in test_idx})


def test_adjacent_overlapping_clips_stay_on_the_same_side():
    """Clip N spans [2N, 2N+3], so clips k and k+1 share 1s of frames."""
    video_names = ["20221208_08_1", "20221208_08_2",
                   "20230101_21_1", "20230101_21_2",
                   "20230202_33_1", "20230202_33_2"]
    groups = [cv_utils.cv_group_key(v) for v in video_names]
    for train_idx, test_idx in cv_utils.group_kfold_indices(groups, k=3):
        train_set, test_set = set(train_idx.tolist()), set(test_idx.tolist())
        for i in range(0, len(video_names), 2):
            pair = {i, i + 1}
            assert pair <= train_set or pair <= test_set


def test_grouping_report_names_videos_and_fold_counts():
    groups = [f"2023010{r % 10}_{r:02d}" for r in range(20) for _ in range(3)]
    text, ok = cv_utils.grouping_report(groups, k=5)
    assert "20" in text and "video" in text.lower()
    assert ok


def test_grouping_report_flags_too_few_videos_to_report():
    groups = [f"2023010{r}_{r:02d}" for r in range(6) for _ in range(3)]
    text, ok = cv_utils.grouping_report(groups, k=5)
    assert not ok
    assert "too few" in text.lower()


def test_cv_group_key_refuses_an_unrecognized_name():
    with pytest.raises(ValueError):
        cv_utils.cv_group_key("some_unexpected_name")

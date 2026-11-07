"""A SAV clip is a single 3-second unit, so make_windows emits one window per
person per clip rather than sliding a fixed 90-frame window across it.

The real footage is 18 fps, so a 3s clip is 54 frames -- fewer than the 90-frame
window the old slider required, which produced zero windows for every clip. These
tests pin the clip-span contract at the real frame counts.
"""
import json

import pytest

import make_windows

CLIP_FRAMES_18FPS = 54          # 3s at the 18 fps the SAV sources actually are
CLIP_FRAMES_30FPS = 90          # 3s at the fps the original design had assumed


def _write_timeline(path, n_frames, people):
    """Write an extract_poses-shaped timeline.

    people maps track_id -> set of frames the person is detected in, or None for
    "present in every frame". The x and y channels are set to the frame index so
    interpolation across a gap is checkable by value.
    """
    with open(path, "w") as f:
        for i in range(n_frames):
            record = {"frame_idx": i, "students": {}}
            for track_id, frames in people.items():
                if frames is None or i in frames:
                    record["students"][str(track_id)] = {
                        "keypoints": [[float(i), float(i), 0.9] for _ in range(17)],
                        "bbox": [100.0, 100.0, 50.0, 80.0],
                    }
            f.write(json.dumps(record) + "\n")
    return str(path)


def _timeline(tmp_path, n_frames, people, clip="20181016_05_7"):
    return _write_timeline(tmp_path / f"{clip}_timeline.jsonl", n_frames, people)


def test_18fps_clip_yields_one_window_per_person(tmp_path):
    """The bug this change fixes: 54 frames < 90 produced zero windows."""
    path = _timeline(tmp_path, CLIP_FRAMES_18FPS, {1: None, 2: None})

    windows = make_windows.preprocess_jsonl_to_windows(path)

    assert len(windows) == 2
    assert sorted(w["track_id"] for w in windows) == [1, 2]


def test_window_keeps_the_clips_native_frame_count(tmp_path):
    """Windows stay native-length on disk so the annotation UI's frame i lines up
    with video frame i. Length normalization to T=90 happens at load time."""
    path = _timeline(tmp_path, CLIP_FRAMES_18FPS, {1: None})

    w = make_windows.preprocess_jsonl_to_windows(path)[0]

    assert w["keypoints"].shape == (3, CLIP_FRAMES_18FPS, 17)
    assert len(w["bboxes"]) == CLIP_FRAMES_18FPS
    assert (w["start_frame"], w["end_frame"]) == (0, CLIP_FRAMES_18FPS - 1)


def test_window_is_marked_time_normalized_false_with_native_frame_count(tmp_path):
    """The loader needs to know the window is not yet at the model's T."""
    path = _timeline(tmp_path, CLIP_FRAMES_18FPS, {1: None})

    w = make_windows.preprocess_jsonl_to_windows(path)[0]

    assert w["n_native_frames"] == CLIP_FRAMES_18FPS


def test_video_name_is_the_clip_id(tmp_path):
    """video_name is the join key to SAV's action prior, so it must stay the clip id."""
    path = _timeline(tmp_path, CLIP_FRAMES_18FPS, {1: None}, clip="20230118_34_2")

    w = make_windows.preprocess_jsonl_to_windows(path)[0]

    assert w["video_name"] == "20230118_34_2"


def test_30fps_clip_still_yields_one_window(tmp_path):
    """Regression: the one case the old slider handled must keep working."""
    path = _timeline(tmp_path, CLIP_FRAMES_30FPS, {1: None})

    windows = make_windows.preprocess_jsonl_to_windows(path)

    assert len(windows) == 1
    assert windows[0]["keypoints"].shape == (3, CLIP_FRAMES_30FPS, 17)


def test_person_below_min_coverage_is_dropped(tmp_path):
    """Coverage is a fraction of the clip, not an absolute frame count."""
    half = set(range(CLIP_FRAMES_18FPS // 2))
    path = _timeline(tmp_path, CLIP_FRAMES_18FPS, {1: None, 2: half})

    windows = make_windows.preprocess_jsonl_to_windows(path, min_coverage=0.8)

    assert [w["track_id"] for w in windows] == [1]


def test_person_just_above_min_coverage_is_kept(tmp_path):
    """44/54 = 81.5% clears a 0.8 gate that an absolute 80-frame gate would fail."""
    path = _timeline(tmp_path, CLIP_FRAMES_18FPS, {1: set(range(44))})

    windows = make_windows.preprocess_jsonl_to_windows(path, min_coverage=0.8)

    assert len(windows) == 1


def test_interior_gap_is_interpolated(tmp_path):
    """x and y are the frame index by construction, so a filled gap equals its index."""
    present = set(range(CLIP_FRAMES_18FPS)) - {10}
    path = _timeline(tmp_path, CLIP_FRAMES_18FPS, {1: present})

    w = make_windows.preprocess_jsonl_to_windows(path)[0]

    assert w["keypoints"][0, 10, 0] == pytest.approx(10.0)
    assert w["keypoints"][1, 10, 0] == pytest.approx(10.0)


def test_clip_with_no_detections_yields_nothing(tmp_path):
    path = _timeline(tmp_path, CLIP_FRAMES_18FPS, {})

    assert make_windows.preprocess_jsonl_to_windows(path) == []


def test_source_video_length_timeline_is_rejected(tmp_path):
    """paths.py warns that pointing extract_poses at a 40-minute source breaks the
    clip-id join. Clip-span windowing would silently build one enormous window, so
    fail loudly instead."""
    path = _timeline(tmp_path, 500, {1: None})

    with pytest.raises(ValueError, match="clip"):
        make_windows.preprocess_jsonl_to_windows(path)

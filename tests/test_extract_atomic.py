import json
import os

import pytest

import extract_poses


def _read_lines(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_clean_exit_renames_into_place(tmp_path):
    final = str(tmp_path / "clip_timeline.jsonl")

    with extract_poses.atomic_jsonl(final) as f_out:
        # nothing is visible under the final name while writing
        assert not os.path.exists(final)
        for i in range(3):
            f_out.write(json.dumps({"frame_idx": i, "students": {}}) + "\n")

    assert os.path.exists(final)
    assert len(_read_lines(final)) == 3
    assert os.listdir(tmp_path) == ["clip_timeline.jsonl"]


def test_crash_leaves_no_file_under_the_final_name(tmp_path):
    final = str(tmp_path / "clip_timeline.jsonl")

    with pytest.raises(RuntimeError):
        with extract_poses.atomic_jsonl(final) as f_out:
            f_out.write(json.dumps({"frame_idx": 0, "students": {}}) + "\n")
            raise RuntimeError("GPU died mid-clip")

    # the whole point: a partial write must never look like a finished timeline
    assert not os.path.exists(final)
    assert os.listdir(tmp_path) == []


def test_interrupt_leaves_no_file_under_the_final_name(tmp_path):
    # the real failure mode is the operator killing a long run, which raises
    # KeyboardInterrupt -- a BaseException, not an Exception
    final = str(tmp_path / "clip_timeline.jsonl")

    with pytest.raises(KeyboardInterrupt):
        with extract_poses.atomic_jsonl(final) as f_out:
            f_out.write(json.dumps({"frame_idx": 0, "students": {}}) + "\n")
            raise KeyboardInterrupt

    assert not os.path.exists(final)
    assert os.listdir(tmp_path) == []


def test_partial_file_is_not_mistaken_for_a_finished_timeline(tmp_path):
    # simulate a SIGKILL: the temp file survives on disk with no chance to clean up
    final = str(tmp_path / "clip_timeline.jsonl")
    with open(final + extract_poses.PARTIAL_SUFFIX, "w") as f:
        f.write(json.dumps({"frame_idx": 0, "students": {}}) + "\n")

    assert not extract_poses.is_already_extracted(final)


def test_completed_timeline_is_skipped(tmp_path):
    final = str(tmp_path / "clip_timeline.jsonl")
    with extract_poses.atomic_jsonl(final) as f_out:
        f_out.write(json.dumps({"frame_idx": 0, "students": {}}) + "\n")

    assert extract_poses.is_already_extracted(final)


def test_rerun_after_crash_overwrites_the_stale_partial(tmp_path):
    final = str(tmp_path / "clip_timeline.jsonl")
    with open(final + extract_poses.PARTIAL_SUFFIX, "w") as f:
        f.write("garbage from a killed run\n")

    with extract_poses.atomic_jsonl(final) as f_out:
        f_out.write(json.dumps({"frame_idx": 0, "students": {}}) + "\n")

    assert _read_lines(final) == [{"frame_idx": 0, "students": {}}]
    assert os.listdir(tmp_path) == ["clip_timeline.jsonl"]

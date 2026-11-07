"""SAV names each annotated clip `{video}_{N}` and documents no mapping from N to a
time. Measured 2026-07-30 across all 60 downloaded videos: stride is 2s per index, so
clip N spans [2N, 2N+3] with its annotated keyframe at 2N+1 (which is why every AVA
row carries timestamp=1).

Cutting at the wrong offset would silently mis-cut all 462 clips, so the mapping and
the overrun guard are pinned here.
"""
import pytest

import cut_clips


def test_clip_span_starts_at_twice_the_index():
    assert cut_clips.clip_span(0) == (0.0, 3.0)
    assert cut_clips.clip_span(1) == (2.0, 5.0)
    assert cut_clips.clip_span(100) == (200.0, 203.0)


def test_adjacent_clips_overlap_one_second():
    """A 3s clip every 2s overlaps its neighbour by 1s -- this is exactly why clip id
    is the wrong CV grouping key and folds must group by room."""
    _, a_end = cut_clips.clip_span(10)
    b_start, _ = cut_clips.clip_span(11)
    assert a_end - b_start == pytest.approx(1.0)


def test_keyframe_is_one_second_into_the_clip():
    """SAV's AVA rows all carry timestamp=1, i.e. 1s into the clip."""
    start, _ = cut_clips.clip_span(50)
    assert cut_clips.keyframe_time(50) == pytest.approx(start + 1.0)


def test_parse_clip_id_splits_video_and_index():
    assert cut_clips.parse_clip_id("20181016_05_938") == ("20181016_05", 938)


def test_parse_clip_id_rejects_a_source_video_id():
    with pytest.raises(ValueError):
        cut_clips.parse_clip_id("20181016_05")


def test_plan_cuts_maps_clips_to_start_times():
    cuts, skipped = cut_clips.plan_cuts(["20181016_05_100"], {"20181016_05": 2401.8})

    assert skipped == []
    assert len(cuts) == 1
    assert cuts[0].clip_id == "20181016_05_100"
    assert cuts[0].video_id == "20181016_05"
    assert cuts[0].start_s == pytest.approx(200.0)


def test_plan_cuts_skips_clips_past_the_end_of_our_copy():
    """20230415_24's Youku upload is ~600s shorter than SAV's original, so 3 of its
    annotated clips do not exist in the footage we have."""
    cuts, skipped = cut_clips.plan_cuts(
        ["20230415_24_800", "20230415_24_1112"], {"20230415_24": 1620.2})

    assert [c.clip_id for c in cuts] == ["20230415_24_800"]
    assert len(skipped) == 1
    clip_id, reason = skipped[0]
    assert clip_id == "20230415_24_1112"
    assert "overrun" in reason


def test_plan_cuts_skips_clips_whose_source_is_not_downloaded():
    cuts, skipped = cut_clips.plan_cuts(["99999999_99_10"], {"20181016_05": 2401.8})

    assert cuts == []
    assert skipped[0][0] == "99999999_99_10"
    assert "source" in skipped[0][1]


def test_ffmpeg_args_reencode_and_do_not_force_an_fps():
    """Stream-copy would snap a 3s cut to the nearest keyframe and miss the clip.
    Forcing an fps would resample footage whose native rate is 18-30fps."""
    args = cut_clips.ffmpeg_cut_args("/src/v.mp4", 200.0, 3.0, "/out/v_100.mp4")

    assert "-c" not in args and "copy" not in args
    assert "-r" not in args
    assert "libx264" in args
    assert args[args.index("-ss") + 1] == "200.000"
    assert args[args.index("-t") + 1] == "3.000"
    assert args[-1] == "/out/v_100.mp4"


def test_ffmpeg_args_drop_audio():
    args = cut_clips.ffmpeg_cut_args("/src/v.mp4", 0.0, 3.0, "/out/x.mp4")
    assert "-an" in args

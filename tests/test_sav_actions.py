import sav_actions as sa


PBTXT = '''item {
  name: "standing"
  id: 1
}
item {
  name: "looking around"
  id: 2
}
item {
  name: "reading"
  id: 3
}
'''

# AVA rows: video_id,timestamp,x1,y1,x2,y2,action_id,person_id  (no header)
CSV = (
    "clipA_1,0,0.1,0.1,0.2,0.3,1,0\n"   # standing
    "clipA_1,0,0.1,0.1,0.2,0.3,2,0\n"   # looking around (same person, multi-label)
    "clipB_2,0,0.4,0.4,0.5,0.6,3,1\n"   # reading
)


def test_load_label_map(tmp_path):
    p = tmp_path / "labels.pbtxt"
    p.write_text(PBTXT)
    assert sa.load_label_map(str(p)) == {1: "standing", 2: "looking around", 3: "reading"}


# The real education_first_label.pbtxt is not uniformly formatted: entries open with
# `item{` (no space) and the "read" entry is `name:"read";` -- no space after the colon
# AND a stray trailing semicolon. A parser that assumes clean formatting silently drops
# that one action, which is exactly the most important focused-signal action.
REAL_PBTXT_QUIRKS = '''item{
  name: "sit"
  id: 1
}
item{
  name:"read";
  id: 5
}
item{
  name: "talk_with_others"
  id: 14
}
'''


def test_load_label_map_tolerates_real_sav_formatting(tmp_path):
    p = tmp_path / "labels.pbtxt"
    p.write_text(REAL_PBTXT_QUIRKS)
    assert sa.load_label_map(str(p)) == {1: "sit", 5: "read", 14: "talk_with_others"}


def test_load_clip_person_actions_groups_by_clip_then_person(tmp_path):
    pb = tmp_path / "labels.pbtxt"; pb.write_text(PBTXT)
    csv = tmp_path / "train.csv"; csv.write_text(CSV)
    label_map = sa.load_label_map(str(pb))
    clips = sa.load_clip_person_actions([str(csv)], label_map)
    # person 0 of clipA_1 is multi-label; the two rows collapse onto one person
    assert clips["clipA_1"] == {"0": {"standing", "looking around"}}
    assert clips["clipB_2"] == {"1": {"reading"}}


# --- the distracted score -------------------------------------------------
# A SAV clip is a 3-second shot of a WHOLE classroom (median 29 annotated people),
# so "does anybody look sideways" is true for 76.6% of real clips and selects
# almost nothing. The prior is therefore a fraction-of-people score, not a flag.

def _people(*distracted_counts):
    """Build {person_id: actions} where the first N people are distracted."""
    n_dis, n_total = distracted_counts
    people = {}
    for i in range(n_total):
        people[str(i)] = {"look_sideways"} if i < n_dis else {"sit", "look_forward"}
    return people


def test_clip_distracted_score_is_the_fraction_of_people():
    cpa = {"quiet": _people(1, 20),      # 1 of 20 glanced away
           "rowdy": _people(10, 20),     # half the room off-task
           "still": _people(0, 8)}
    assert sa.clip_distracted_score("quiet", cpa) == 0.05
    assert sa.clip_distracted_score("rowdy", cpa) == 0.5
    assert sa.clip_distracted_score("still", cpa) == 0.0


def test_clip_distracted_score_is_zero_for_unknown_clip():
    assert sa.clip_distracted_score("missing", {}) == 0.0


def test_score_separates_clips_the_boolean_flag_could_not():
    """Both clips contain a distracted person, so any boolean prior calls them
    equal. The score must rank the crowded-with-distraction one far higher."""
    cpa = {"one_glance": _people(1, 29), "half_the_room": _people(14, 29)}
    assert sa.clip_distracted_score("one_glance", cpa) < 0.05
    assert sa.clip_distracted_score("half_the_room", cpa) > 0.45


def test_clip_is_distracted_prior_uses_the_score_threshold():
    cpa = {"quiet": _people(1, 20), "rowdy": _people(10, 20)}
    assert sa.clip_is_distracted_prior("rowdy", cpa) is True
    assert sa.clip_is_distracted_prior("quiet", cpa) is False    # 5% < 25% default
    # threshold is tunable
    assert sa.clip_is_distracted_prior("quiet", cpa, threshold=0.04) is True
    assert sa.clip_is_distracted_prior("rowdy", cpa, threshold=0.9) is False


def test_default_threshold_is_documented_and_selective():
    assert 0.0 < sa.PRIOR_SCORE_THRESHOLD < 1.0


def test_clip_key_for_window_id():
    wid = "clipA_1_id_7_f0_f89"
    assert sa.clip_key_for_window_id(wid) == "clipA_1"


def test_source_video_id_strips_trailing_index():
    assert sa.source_video_id("classroom42_7") == "classroom42"
    assert sa.source_video_id("classroom42") == "classroom42"   # no index -> unchanged


def test_join_key_matches_real_sav_naming():
    # Real SAV clip ids look like <date>_<video>_<clipindex>, e.g. "20230118_34_136".
    # cut_clips.py writes each clip as "<clip_id>.mp4", so video_name == the SAV clip id.
    wid = "20230118_34_136_id_7_f0_f89"
    assert sa.clip_key_for_window_id(wid) == "20230118_34_136"
    assert sa.source_video_id("20230118_34_136") == "20230118_34"


def test_action_prior_covers_15_actions():
    assert len(sa.ACTION_PRIOR) == 15
    assert set(sa.ACTION_PRIOR.values()) <= {"distracted", "focused", "neutral"}
    # keys are the exact strings from the real education_first_label.pbtxt
    assert sa.ACTION_PRIOR["look_sideways"] == "distracted"
    assert sa.ACTION_PRIOR["read"] == "focused"
    assert sa.ACTION_PRIOR["stand"] == "neutral"


def test_action_prior_keys_match_the_real_label_file():
    """The prior is keyed by action NAME, so a spelling drift between the label file
    and ACTION_PRIOR silently makes every clip look non-distracted."""
    expected = {"sit", "stand", "look_forward", "look_sideways", "read", "flip_books",
                "touch_sth", "raise_hand", "hands_down", "take_notes", "applaud",
                "bend", "turn_around", "talk_with_others", "answer_questions"}
    assert set(sa.ACTION_PRIOR) == expected


def test_clip_is_distracted_prior_handles_missing_and_clean_clips():
    cpa = {"clean": {"0": {"read"}, "1": {"sit", "look_forward"}}}
    assert sa.clip_is_distracted_prior("clean", cpa) is False
    assert sa.clip_is_distracted_prior("missing", cpa) is False  # no annotation -> not prior


# --- source-video shortlist (which videos are worth downloading at all) ---

def _clip(n_people, n_distracted):
    """A synthetic clip: n_people annotated, the first n_distracted off-task."""
    return {str(i): ({"look_sideways"} if i < n_distracted else {"read"})
            for i in range(n_people)}


def test_rank_source_videos_orders_by_high_score_clip_count():
    cpa = {
        "20230101_01_1": _clip(4, 2),   # score 0.50 -> high
        "20230101_01_2": _clip(4, 2),   # score 0.50 -> high
        "20230101_01_3": _clip(4, 0),   # score 0.00
        "20230102_02_1": _clip(4, 1),   # score 0.25 -> high
        "20230102_02_2": _clip(4, 0),   # score 0.00
    }
    ranked = sa.rank_source_videos(cpa)
    assert [r["video"] for r in ranked] == ["20230101_01", "20230102_02"]
    assert ranked[0]["n_high"] == 2
    assert ranked[0]["n_clips"] == 3


def test_rank_source_videos_excludes_videos_with_no_high_score_clips():
    """A video with nothing above threshold cannot feed the distracted half of the
    worklist, so it is not worth any download bandwidth."""
    cpa = {"20230101_01_1": _clip(4, 2),
           "20230103_07_1": _clip(4, 0)}
    assert [r["video"] for r in sa.rank_source_videos(cpa)] == ["20230101_01"]


def test_select_source_videos_applies_an_explicit_date_cap():
    cpa = {
        "20230101_01_1": _clip(4, 2), "20230101_01_2": _clip(4, 2),
        "20230101_01_3": _clip(4, 2),
        "20230101_09_1": _clip(4, 2), "20230101_09_2": _clip(4, 2),
        "20230102_02_1": _clip(4, 2),
    }
    assert sa.select_source_videos(cpa, n=2, max_per_date=1) == [
        "20230101_01", "20230102_02"]


def test_select_source_videos_takes_rank_order_when_uncapped():
    cpa = {
        "20230101_01_1": _clip(4, 2), "20230101_01_2": _clip(4, 2),
        "20230101_01_3": _clip(4, 2),
        "20230101_09_1": _clip(4, 2), "20230101_09_2": _clip(4, 2),
        "20230102_02_1": _clip(4, 2),
    }
    assert sa.select_source_videos(cpa, n=2) == ["20230101_01", "20230101_09"]


def test_select_source_videos_caps_at_n():
    cpa = {"20230101_01_1": _clip(4, 2),
           "20230102_02_1": _clip(4, 2),
           "20230103_03_1": _clip(4, 2)}
    assert sa.select_source_videos(cpa, n=1) == ["20230101_01"]


def test_select_source_videos_caps_repeats_per_date():
    """Same-date videos are the pairs most likely to share a classroom, and folds
    group by source video, so a same-room pair split across folds is undetectable."""
    cpa = {
        "20230101_08_1": _clip(4, 2), "20230101_08_2": _clip(4, 2),
        "20230101_09_1": _clip(4, 2),
        "20230103_21_1": _clip(4, 2),
    }
    assert sa.select_source_videos(cpa, n=2, max_per_date=1) == [
        "20230101_08", "20230103_21"]


def test_select_source_videos_allows_more_per_date_when_raised():
    cpa = {"20230101_08_1": _clip(4, 2), "20230101_08_2": _clip(4, 2),
           "20230101_09_1": _clip(4, 2)}
    assert sa.select_source_videos(cpa, n=2, max_per_date=2) == [
        "20230101_08", "20230101_09"]


def test_source_video_room_is_gone():
    """The NN field is not a room; grouping must never depend on it again."""
    assert not hasattr(sa, "source_video_room")



def test_select_source_videos_has_no_date_cap_by_default():
    cpa = {"20230101_08_1": _clip(4, 2), "20230101_09_1": _clip(4, 2)}
    assert len(sa.select_source_videos(cpa, n=2)) == 2

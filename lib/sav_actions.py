"""Ingest SAV's AVA-format action annotations into per-clip action sets, and
derive a coarse engagement prior used ONLY to prioritize which windows a human
labels. SAV labels are never used as training/eval labels.
"""
import csv as _csv
import re


def load_label_map(pbtxt_path):
    """Parse education_first_label.pbtxt -> {action_id: action_name}.

    SAV's real label file is not uniformly formatted -- most entries are
    `name: "sit"` but one is `name:"read";` (no space, stray semicolon). We
    therefore allow arbitrary filler between the name and its id, bounded by
    `[^}]` so a match can never run past the end of its own `item{...}` block.
    """
    text = open(pbtxt_path, encoding="utf-8").read()
    items = re.findall(r'name:\s*"([^"]+)"[^}]*?id:\s*(\d+)', text)
    return {int(i): name for name, i in items}


def load_clip_person_actions(csv_paths, label_map):
    """AVA rows video_id,timestamp,x1,y1,x2,y2,action_id,person_id
    -> {clip: {person_id: {action_name}}}.

    Grouping by person (not just by clip) is what makes the prior usable: a SAV
    clip is a 3-second shot of a whole classroom, so a clip-level union of action
    names saturates and carries almost no signal. See clip_distracted_score.
    """
    clips = {}
    for path in csv_paths:
        with open(path, newline="", encoding="utf-8") as f:
            for row in _csv.reader(f):
                if len(row) < 8 or not row[6].strip().lstrip("-").isdigit():
                    continue                      # skip header / malformed rows
                clip, action_id, person_id = row[0], int(row[6]), row[7]
                name = label_map.get(action_id)
                if name is not None:
                    clips.setdefault(clip, {}).setdefault(person_id, set()).add(name)
    return clips


def clip_key_for_window_id(window_id):
    """Clip identity of a window: everything before the '_id_' track marker."""
    return window_id.split("_id_")[0]


def source_video_id(clip_key):
    """Source video id: clip key with a trailing '_<digits>' clip index removed."""
    return re.sub(r"_\d+$", "", clip_key)


# --- engagement prior (prioritization only; annotators make the real call) ---
# Keys are the EXACT name strings from SAV's education_first_label.pbtxt, verified
# against the real file on 2026-07-29 (ids 1-15). Buckets follow the design spec
# 2026-07-26-sav-gold-set-sampling-design.md Sec. 5. A spelling drift here is silent
# and total -- every clip would look non-distracted -- so tests pin the key set.
ACTION_PRIOR = {
    # pose
    "sit": "neutral",                    # id 1
    "stand": "neutral",                  # id 2
    # sight
    "look_forward": "focused",           # id 3
    "look_sideways": "distracted",       # id 4
    "read": "focused",                   # id 5
    # person-object
    "flip_books": "neutral",             # id 6
    "touch_sth": "neutral",              # id 7
    "raise_hand": "focused",             # id 8
    "hands_down": "neutral",             # id 9
    "take_notes": "focused",             # id 10
    "applaud": "neutral",                # id 11
    # body-motion
    "bend": "neutral",                   # id 12
    "turn_around": "distracted",         # id 13
    # person-person
    "talk_with_others": "distracted",    # id 14
    "answer_questions": "focused",       # id 15
}

_DISTRACTED_ACTIONS = {a for a, bucket in ACTION_PRIOR.items() if bucket == "distracted"}

# A clip counts as distracted-prior when at least this fraction of its annotated
# people show a distracted-signal action. Measured on the real SAV annotations
# (2026-07-29, 4,324 clips, median 29 people per clip):
#
#   threshold   clips kept   % of all   source videos
#        0%        3,313       76.6%        739     <- the old boolean; selects almost everything
#       25%        1,253       29.0%        513     <- default: selective, still 513 videos
#       50%          384        8.9%        240
#
# 0.25 is the knee: it cuts the candidate pool to under a third while leaving far
# more source videos than the 30-60 the study needs, so video spread stays free.
PRIOR_SCORE_THRESHOLD = 0.25


def clip_distracted_score(clip_key, clip_person_actions):
    """Fraction of the clip's annotated people showing a distracted-signal action.

    This is the prior's strength signal. A boolean "is anyone distracted?" is true
    for 76.6% of real SAV clips -- in a room of ~29 students somebody is always
    glancing away -- so it cannot prioritize anything. The fraction separates a
    clip where 1 of 29 glanced away (0.03) from one where 14 of 29 are off-task
    (0.48). Returns 0.0 for a clip with no annotations.
    """
    people = clip_person_actions.get(clip_key)
    if not people:
        return 0.0
    n_distracted = sum(1 for actions in people.values()
                       if actions & _DISTRACTED_ACTIONS)
    return n_distracted / len(people)


def clip_is_distracted_prior(clip_key, clip_person_actions,
                             threshold=PRIOR_SCORE_THRESHOLD):
    """True iff the clip's distracted score reaches `threshold`."""
    return clip_distracted_score(clip_key, clip_person_actions) >= threshold


# Mean `clip_distracted_score` over all 4,324 annotated SAV clips. The baseline a
# sampling prior has to beat: it is what you would get by picking clips at random.
SAV_WIDE_DISTRACTED_MEAN = 0.178


def rank_source_videos(clip_person_actions, threshold=PRIOR_SCORE_THRESHOLD):
    """Rank source videos by how many distracted-prior clips each one contains.

    Only ~60 of SAV's 758 source videos get downloaded, and a video whose clips are
    all low-score cannot feed the distracted half of the worklist -- it would cost
    ~12 minutes of download for windows we already have too many of. Videos with
    zero high-score clips are dropped entirely.

    Returns [{video, n_high, n_clips, mean_score}] sorted by n_high desc, then
    mean_score desc, then video id for determinism.
    """
    stats = {}
    for clip_key in clip_person_actions:
        vid = source_video_id(clip_key)
        s = stats.setdefault(vid, {"video": vid, "n_high": 0, "n_clips": 0,
                                   "_score_sum": 0.0})
        score = clip_distracted_score(clip_key, clip_person_actions)
        s["n_clips"] += 1
        s["_score_sum"] += score
        if score >= threshold:
            s["n_high"] += 1
    ranked = []
    for s in stats.values():
        if not s["n_high"]:
            continue
        s["mean_score"] = s.pop("_score_sum") / s["n_clips"]
        ranked.append(s)
    ranked.sort(key=lambda s: (-s["n_high"], -s["mean_score"], s["video"]))
    return ranked


def source_video_date(video_id):
    """Recording-date prefix of a source video id ('20230118_34' -> '20230118')."""
    return video_id.split("_")[0]


def select_source_videos(clip_person_actions, n=60,
                         threshold=PRIOR_SCORE_THRESHOLD, max_per_date=None):
    """Pick the `n` highest-ranked source videos.

    `max_per_date` caps videos per recording date; None means no cap.
    """
    chosen, per_date = [], {}
    for s in rank_source_videos(clip_person_actions, threshold):
        date = source_video_date(s["video"])
        if max_per_date is not None and per_date.get(date, 0) >= max_per_date:
            continue
        per_date[date] = per_date.get(date, 0) + 1
        chosen.append(s["video"])
        if len(chosen) == n:
            break
    return chosen

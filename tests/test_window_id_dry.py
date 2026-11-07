"""window_id is the join key for the whole project. Exactly one function may
construct it; anything else re-deriving the format string is a corruption risk."""
import os
import re

import common

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# matches a hand-rolled f-string like f"{w['video_name']}_id_{w['track_id']}_f..."
_HANDROLLED = re.compile(r"_id_\{.*?track_id")


def _source_files():
    for sub in ("lib", "steps"):
        d = os.path.join(_REPO, sub)
        for name in sorted(os.listdir(d)):
            if name.endswith(".py"):
                yield os.path.join(d, name)


def test_only_common_constructs_window_id():
    offenders = []
    for path in _source_files():
        if os.path.basename(path) == "common.py":
            continue                       # the one legitimate definition
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                if _HANDROLLED.search(line):
                    offenders.append(f"{os.path.relpath(path, _REPO)}:{lineno}")
    assert offenders == [], (
        "these files build window_id by hand instead of calling "
        f"common.make_window_id: {offenders}")


def test_make_window_id_is_the_documented_format():
    w = {"video_name": "vidA_1", "track_id": 7, "start_frame": 45, "end_frame": 134}
    assert common.make_window_id(w) == "vidA_1_id_7_f45_f134"

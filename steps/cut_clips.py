"""Step: cut SAV's annotated 3-second clips out of the downloaded source videos.

Reads the study's clip list ``datasets/sav/clip_list.txt`` and the sources in
``datasets/sav/videos_source/``, writes one ``<clip_id>.mp4`` per clip into
``datasets/sav/videos/`` plus a manifest recording exactly what was cut from where.

**The clip-index -> timestamp mapping.** SAV names each clip ``{video}_{N}`` and
documents no mapping from N to a time. Established empirically 2026-07-30 against all
60 downloaded videos:

* **Stride is 2s per index**, so clip N spans ``[2N, 2N+3]``. Evidence: sweeping
  candidate mappings ``t = m*N`` and scoring YOLO detections against SAV's own
  annotated person boxes, ``m=2`` beat every alternative ~3x (mean best IoU 0.303 vs
  0.08-0.12; 8.5 matched people per clip vs 1.6-2.8; won 16 of 24 clips).
* The keyframe sits at ``2N+1`` -- every AVA row carries ``timestamp=1``, i.e. 1s into
  its clip, and an IoU sweep over offsets peaks weakly at +1.0s.
* Adjacent indices therefore **overlap 1s**, which is why clip id is the wrong CV
  grouping key and folds must group by source video.

The phase is a best estimate, not a proof: over +/-2s the IoU curve is nearly flat
(peak/median 1.08) because seated students barely move in two seconds. A residual
sub-second offset is possible and costs sampling-prior precision only -- annotators
label the footage they are shown.
"""
import _bootstrap  # noqa: F401  -- puts ../lib on sys.path

import argparse
import collections
import csv
import json
import os
import re
import subprocess

from paths import (
    SAV_DIR,
    SAV_VIDEOS_DIR,
    SAV_VIDEOS_SOURCE_DIR,
)

CLIP_SECONDS = 3.0
STRIDE_S = 2.0          # seconds per clip index -- measured, see module docstring
KEYFRAME_OFFSET_S = 1.0  # AVA timestamp=1 is 1s into the clip

MANIFEST_FIELDS = ["clip_id", "video_id", "start_s", "end_s", "keyframe_s",
                   "fps", "frames", "width", "height", "bytes"]

# A SAV video id is <8-digit date>_<NN>, so a clip id has exactly one more numeric
# part. Matching a loose `.+_\d+` would read the *source* id "20181016_05" as clip 5 of
# video "20181016" and silently cut from a video that does not exist.
_CLIP_ID_RE = re.compile(r"^(?P<video>\d{8}_\d+)_(?P<index>\d+)$")


def clip_span(clip_index, clip_seconds=CLIP_SECONDS, stride=STRIDE_S):
    """(start, end) in seconds of clip ``clip_index`` within its source video."""
    start = stride * clip_index
    return (start, start + clip_seconds)


def keyframe_time(clip_index, stride=STRIDE_S, offset=KEYFRAME_OFFSET_S):
    """Absolute time of the frame SAV's action labels describe."""
    return stride * clip_index + offset


def parse_clip_id(clip_id):
    """``20181016_05_938`` -> ``("20181016_05", 938)``."""
    m = _CLIP_ID_RE.match(clip_id)
    if not m:
        raise ValueError(f"{clip_id!r} is not a clip id (expected <video>_<index>)")
    return m.group("video"), int(m.group("index"))


Cut = collections.namedtuple("Cut", "clip_id video_id start_s end_s")


def plan_cuts(clip_ids, source_durations, clip_seconds=CLIP_SECONDS, stride=STRIDE_S):
    """Split clips into cuttable ones and skipped ones with a reason.

    ``source_durations`` maps video_id -> duration in seconds. A clip is skipped when
    its source was not downloaded, or when it runs past the end of the copy we have
    (one shortlisted upload is ~600s shorter than SAV's original).
    """
    cuts, skipped = [], []
    for clip_id in clip_ids:
        try:
            video_id, index = parse_clip_id(clip_id)
        except ValueError as exc:
            skipped.append((clip_id, str(exc)))
            continue
        duration = source_durations.get(video_id)
        if duration is None:
            skipped.append((clip_id, f"source {video_id}.mp4 not downloaded"))
            continue
        start, end = clip_span(index, clip_seconds, stride)
        if end > duration:
            skipped.append((clip_id, f"overrun: needs {end:.1f}s of a {duration:.1f}s source"))
            continue
        cuts.append(Cut(clip_id, video_id, start, end))
    return cuts, skipped


def ffmpeg_cut_args(src, start_s, seconds, dest):
    """Frame-accurate cut, native fps preserved.

    Re-encodes deliberately: ``-c copy`` snaps the cut to the nearest keyframe, which
    on a 3-second clip can miss the annotated moment entirely. No ``-r``, so footage
    keeps its own frame rate (18-30fps across the shortlist) for the fps-aware
    downstream normalization.
    """
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start_s:.3f}",
        "-i", src,
        "-t", f"{seconds:.3f}",
        "-map", "0:v:0",
        "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        dest,
    ]


def probe_sources(source_dir):
    """{video_id: {duration, fps, width, height}} for every downloaded source."""
    out = {}
    for name in sorted(os.listdir(source_dir)):
        if not name.endswith(".mp4"):
            continue
        path = os.path.join(source_dir, name)
        done = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate,width,height",
             "-show_entries", "format=duration", "-of", "json", path],
            capture_output=True, text=True, check=True)
        d = json.loads(done.stdout)
        s = d["streams"][0]
        num, den = s["r_frame_rate"].split("/")
        out[name[:-4]] = {"duration": float(d["format"]["duration"]),
                          "fps": float(num) / float(den),
                          "width": s["width"], "height": s["height"]}
    return out


def read_clip_list(path):
    return [l.strip() for l in open(path, encoding="utf-8") if l.strip()]


def cut_one(cut, source_dir, output_dir, clip_seconds=CLIP_SECONDS):
    dest = os.path.join(output_dir, f"{cut.clip_id}.mp4")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    src = os.path.join(source_dir, f"{cut.video_id}.mp4")
    subprocess.run(ffmpeg_cut_args(src, cut.start_s, clip_seconds, dest), check=True)
    return dest


def main(clips_file, source_dir, output_dir, manifest_file, limit=None):
    os.makedirs(output_dir, exist_ok=True)

    sources = probe_sources(source_dir)
    clip_ids = read_clip_list(clips_file)
    cuts, skipped = plan_cuts(clip_ids, {v: m["duration"] for v, m in sources.items()})
    if limit is not None:
        cuts = cuts[:limit]

    not_ours = sum(1 for _, r in skipped if "not downloaded" in r)
    overruns = [(c, r) for c, r in skipped if "overrun" in r]
    print(f"{len(clip_ids)} clips in list | {len(cuts)} cuttable | "
          f"{not_ours} from videos not downloaded | {len(overruns)} overrun their source")
    for clip_id, reason in overruns:
        print(f"  skip {clip_id}: {reason}")

    rows = []
    for i, cut in enumerate(cuts, start=1):
        dest = cut_one(cut, source_dir, output_dir)
        meta = sources[cut.video_id]
        _, index = parse_clip_id(cut.clip_id)
        rows.append({
            "clip_id": cut.clip_id, "video_id": cut.video_id,
            "start_s": f"{cut.start_s:.3f}", "end_s": f"{cut.end_s:.3f}",
            "keyframe_s": f"{keyframe_time(index):.3f}",
            "fps": f"{meta['fps']:.3f}", "frames": round(meta["fps"] * CLIP_SECONDS),
            "width": meta["width"], "height": meta["height"],
            "bytes": os.path.getsize(dest),
        })
        if i % 50 == 0 or i == len(cuts):
            print(f"  cut {i}/{len(cuts)}")

    with open(manifest_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)

    by_fps = collections.Counter(r["fps"] for r in rows)
    print(f"\ncut {len(rows)} clips -> {output_dir}")
    print(f"fps spread: {dict(sorted(by_fps.items()))}")
    print(f"manifest -> {manifest_file}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", default=os.path.join(SAV_DIR, "clip_list.txt"))
    ap.add_argument("--source-dir", default=SAV_VIDEOS_SOURCE_DIR)
    ap.add_argument("--output-dir", default=SAV_VIDEOS_DIR)
    ap.add_argument("--manifest", default=os.path.join(SAV_DIR, "clip_manifest.csv"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    raise SystemExit(main(args.clips, args.source_dir, args.output_dir,
                          args.manifest, limit=args.limit))

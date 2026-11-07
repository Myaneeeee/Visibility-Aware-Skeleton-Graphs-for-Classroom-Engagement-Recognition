"""Fetch the shortlisted SAV source videos from Youku -- paced, resumable, verified.

SAV ships no video, only 758 Youku links in ``sav_dataset.txt``. Two facts shape this
script:

1. **Pace it.** The yt-dlp youku extractor first fetches "cna info" from
   ``log.mmstat.com`` (Alibaba tracking). Under the default user-agent that request
   times out, and hammering it in parallel gets the client throttled with error
   ``-6004``, whose message text reads like a copyright notice. A 2026-07-29 sweep
   misread exactly that and wrongly declared the entire dataset dead. So: browser
   user-agent, a referer, strictly sequential, with spacing between requests.

2. **Verify what you got.** ``sav_dataset.txt`` holds URLs with no ids, in sorted
   order of the source video ids. The mapping is therefore positional -- and a
   positional mapping that silently slips would mislabel every window downstream,
   because the SAV clip id is the join key to the action annotations. Every Youku
   title embeds its own id (``课堂教学视频20181016_05``), so each download checks the
   fetched title against the predicted id and **fails loudly** on mismatch.
"""
import _bootstrap  # noqa: F401  -- puts ../lib on sys.path

import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time

import paths
import sav_actions as sa

# Verified working 2026-07-30 (12/12 sampled URLs). The user-agent is not cosmetic:
# without it the cna-info step times out before any video API is reached.
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/140.0 Safari/537.36")
REFERER = "https://v.youku.com/"

# Take 720p: SAV's usable-skeleton rate is only 14-66%, and halving linear resolution
# costs keypoints on exactly the distant students that matter.
#
# Select by HEIGHT, never by format id. Youku's ids are per-video and inconsistent --
# three shortlisted videos offered {0, 1}, {0, h3, 2} and {0, 1, h3, h5}. A hardcoded
# `-f 1` gives 720p on the first, does not exist on the second (hard failure), and is
# one of four identical 720p entries on the third.
FORMAT_SELECTOR = "best[height<=720]/best"
FORMAT_SORT = "res:720"

# The media fragments come from a CDN, not from the API that rate-limits, so
# parallelising *within* one video is safe where parallelising across videos is not.
# Measured on a shortlisted video: 147 KiB/s serial vs 920 KiB/s at 8 fragments --
# the serial rate is a per-connection cap, not the available bandwidth.
CONCURRENT_FRAGMENTS = 8

SPACING_S = 3.0          # between videos; never fetch two videos at once
RETRIES = 3
BACKOFF_S = 15.0         # after a failure, wait this * attempt before retrying

MANIFEST_FIELDS = ["video_id", "url", "title", "duration_s", "format_id", "resolution",
                   "bytes", "sha256"]

_ID_RE = re.compile(r"\d{8}_\d+")


# --- shortlist / url resolution ---------------------------------------------

def read_shortlist(path):
    """Video ids from `video_shortlist.txt` (tab-separated, '#' comments)."""
    ids = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line.split("\t")[0])
    return ids


def sorted_source_ids(clips_to_keep_path):
    """The 758 source video ids, sorted -- the order `sav_dataset.txt` follows."""
    with open(clips_to_keep_path) as fh:
        clips = fh.read().split()
    return sorted({sa.source_video_id(c) for c in clips})


def read_urls(path):
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


def resolve_shortlist_urls(shortlist_ids, sorted_ids, urls):
    """Pair each shortlisted id with its URL, by position in the sorted id list.

    Returns pairs in sorted-id order (so the fetch walks the file top to bottom).
    """
    if len(sorted_ids) != len(urls):
        raise ValueError(
            f"length mismatch: {len(sorted_ids)} source ids vs {len(urls)} urls. "
            "The positional mapping between them is void; refusing to guess."
        )
    index = {vid: i for i, vid in enumerate(sorted_ids)}
    resolved = []
    for vid in shortlist_ids:
        if vid not in index:
            raise ValueError(f"{vid!r} is not in the source id list built from "
                             "clips_to_keep.txt")
        resolved.append((index[vid], vid))
    resolved.sort()
    return [(vid, urls[i]) for i, vid in resolved]


# --- the title check that guards the positional mapping ---------------------

def video_id_from_title(title):
    """Extract the SAV id embedded in a Youku title ('课堂教学视频20181016_05')."""
    match = _ID_RE.search(title or "")
    if not match:
        raise ValueError(f"no SAV video id (<8 digits>_<digits>) in title {title!r}")
    return match.group(0)


def verify_title(title, expected_id):
    """Raise unless the fetched title's embedded id is the one we predicted."""
    got = video_id_from_title(title)
    if got != expected_id:
        raise ValueError(
            f"title/id mismatch: expected {expected_id}, got {got} (title {title!r}). "
            "The url<->id mapping has slipped; stopping rather than mislabeling data."
        )


# --- resumability ----------------------------------------------------------

def output_path(source_dir, video_id):
    return os.path.join(source_dir, f"{video_id}.mp4")


def pending_downloads(pairs, source_dir):
    """Pairs whose output is not already a non-empty file on disk."""
    pending = []
    for vid, url in pairs:
        path = output_path(source_dir, vid)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue
        pending.append((vid, url))
    return pending


# --- manifest --------------------------------------------------------------

def _sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path):
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as fh:
        return {row["video_id"]: row for row in csv.DictReader(fh)}


def write_manifest(path, rows_by_id):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for vid in sorted(rows_by_id):
            writer.writerow(rows_by_id[vid])


# --- the network parts (exercised by the trial run, not by unit tests) ------

def yt_dlp_binary():
    """Absolute path to yt-dlp: the venv's copy first, then PATH.

    Resolving this up front turns a missing tool into an immediate clear error rather
    than one discovered partway through a multi-gigabyte fetch.
    """
    local = os.path.join(os.path.dirname(sys.executable), "yt-dlp")
    if os.access(local, os.X_OK):
        return local
    found = shutil.which("yt-dlp")
    if not found:
        raise RuntimeError("yt-dlp not found in the venv or on PATH; "
                           "install it with `pip install -U yt-dlp`")
    return os.path.abspath(found)


def _yt_dlp_base():
    return [yt_dlp_binary(), "--user-agent", USER_AGENT, "--referer", REFERER,
            "--socket-timeout", "30", "--no-warnings"]


def format_args():
    return ["-f", FORMAT_SELECTOR, "-S", FORMAT_SORT]


def probe(url):
    """Metadata for the format that will actually be fetched, without downloading.

    Returns (title, duration_s, format_id, resolution).
    """
    cmd = _yt_dlp_base() + format_args() + [
        "--skip-download",
        "--print", "%(title)s\t%(duration)s\t%(format_id)s\t%(resolution)s", url]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        tail = done.stderr.strip().splitlines()[-1] if done.stderr.strip() else "?"
        raise RuntimeError(f"probe failed: {tail}")
    line = [ln for ln in done.stdout.strip().splitlines() if "\t" in ln][-1]
    title, duration, format_id, resolution = line.split("\t")
    return title.strip(), duration.strip(), format_id.strip(), resolution.strip()


def fetch(url, dest, concurrent_fragments=CONCURRENT_FRAGMENTS):
    cmd = _yt_dlp_base() + format_args() + [
        "--merge-output-format", "mp4", "--continue",
        "--concurrent-fragments", str(concurrent_fragments),
        "-o", dest, url]
    done = subprocess.run(cmd)
    if done.returncode != 0:
        raise RuntimeError(f"download failed with exit {done.returncode}")


def download_one(video_id, url, source_dir, concurrent_fragments=CONCURRENT_FRAGMENTS):
    """Probe, verify the id, fetch, hash. Returns a manifest row."""
    title, duration, format_id, resolution = probe(url)
    verify_title(title, video_id)          # before spending ~350 MiB
    dest = output_path(source_dir, video_id)
    fetch(url, dest, concurrent_fragments)
    if not os.path.exists(dest) or os.path.getsize(dest) == 0:
        raise RuntimeError(f"{dest} missing or empty after a successful exit")
    return {"video_id": video_id, "url": url, "title": title, "duration_s": duration,
            "format_id": format_id, "resolution": resolution,
            "bytes": os.path.getsize(dest), "sha256": _sha256(dest)}


def main(shortlist_file, clips_file, urls_file, source_dir, manifest_file,
         limit=None, spacing=SPACING_S, retries=RETRIES,
         concurrent_fragments=CONCURRENT_FRAGMENTS):
    os.makedirs(source_dir, exist_ok=True)

    shortlist = read_shortlist(shortlist_file)
    pairs = resolve_shortlist_urls(shortlist, sorted_source_ids(clips_file),
                                   read_urls(urls_file))
    todo = pending_downloads(pairs, source_dir)
    already = len(pairs) - len(todo)
    if limit is not None:
        todo = todo[:limit]

    print(f"shortlist: {len(pairs)} videos | already on disk: {already} | "
          f"fetching now: {len(todo)}")
    if not todo:
        print("nothing to do.")
        return 0

    manifest = load_manifest(manifest_file)
    failures = []
    for i, (vid, url) in enumerate(todo, start=1):
        print(f"\n[{i}/{len(todo)}] {vid}")
        for attempt in range(1, retries + 1):
            try:
                manifest[vid] = download_one(vid, url, source_dir,
                                             concurrent_fragments)
                write_manifest(manifest_file, manifest)
                row = manifest[vid]
                print(f"  ok: {row['title']} | {row['duration_s']}s | "
                      f"{row['resolution']} (format {row['format_id']}) | "
                      f"{int(row['bytes']) / 2**20:.0f} MiB")
                break
            except ValueError:
                raise                      # id mismatch: never retry, never continue
            except Exception as exc:        # noqa: BLE001 -- report and retry
                print(f"  attempt {attempt}/{retries} failed: {exc}")
                if "-6004" in str(exc):
                    print("  (-6004 is rate limiting, not a takedown -- backing off)")
                if attempt == retries:
                    failures.append(vid)
                else:
                    time.sleep(BACKOFF_S * attempt)
        if i < len(todo):
            time.sleep(spacing)

    print(f"\ndone. fetched {len(todo) - len(failures)}/{len(todo)}, "
          f"manifest -> {manifest_file}")
    if failures:
        print(f"failed: {', '.join(failures)}")
    return len(failures)


if __name__ == "__main__":
    ann = os.path.join(paths.SAV_DIR, "annotations")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shortlist", default=os.path.join(ann, "video_shortlist.txt"))
    ap.add_argument("--clips-to-keep", default=os.path.join(ann, "clips_to_keep.txt"))
    ap.add_argument("--urls", default=os.path.join(paths.SAV_DIR, "sav_dataset.txt"))
    ap.add_argument("--source-dir", default=paths.SAV_VIDEOS_SOURCE_DIR)
    ap.add_argument("--manifest", default=paths.SAV_DOWNLOAD_MANIFEST_FILE)
    ap.add_argument("--limit", type=int, default=None,
                    help="fetch at most N videos this run (use for a trial)")
    ap.add_argument("--spacing", type=float, default=SPACING_S,
                    help="seconds between videos (default %(default)s); "
                         "lowering this risks the -6004 rate limit")
    ap.add_argument("--concurrent-fragments", type=int, default=CONCURRENT_FRAGMENTS,
                    help="parallel fragment downloads within one video (default "
                         "%(default)s). Fragments come from a CDN, not the rate-limited "
                         "API, so this is safe -- and worth ~6x")
    args = ap.parse_args()
    raise SystemExit(main(args.shortlist, args.clips_to_keep, args.urls,
                          args.source_dir, args.manifest, limit=args.limit,
                          spacing=args.spacing,
                          concurrent_fragments=args.concurrent_fragments))

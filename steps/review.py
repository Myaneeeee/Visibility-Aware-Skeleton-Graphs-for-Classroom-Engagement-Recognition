"""Review every SAV window in the browser: clip + skeleton, with label changes.

    python steps/review.py serve              # open http://127.0.0.1:8765
    python steps/review.py flag               # rank windows by how likely the label is wrong
    python steps/review.py apply --reviewer panel           # dry run
    python steps/review.py apply --reviewer panel --write   # change annotations.csv

Lists every window (labeled, unlabeled, dropped by the quality filter), every label whose
window does not exist, and every clip without a window. Decisions go to
datasets/sav/review/<reviewer>.csv; annotations.csv changes only on `apply --write`.
Ledgers, worklists and agreement files are never touched.
"""
import argparse
import bisect
import csv
import functools
import glob
import json
import os
import re
import threading
import time
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import _bootstrap  # noqa: F401
from common import make_window_id
from paths import (ANNOTATIONS_FILE, EXCLUDED_WINDOWS_FILE, SAV_LABEL_PROVENANCE_FILE,
                   SAV_REVIEW_DIR, SAV_VIDEOS_DIR, SAV_WINDOWS_DIR, SAV_WINDOWS_FILTERED_DIR,
                   report_file, runs_dir)

LABELS = ("Focused", "Distracted", "TRASH")
SKIP = "Skip"
SUSPICION_FILE = report_file("review_suspicion.csv")
UI_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "review_ui.html")
_REVIEWER = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def clip_of(window_id):
    return window_id.split("_id_")[0]


def build_index(windows, filtered_ids, clips, labels, excluded):
    """One item per window, per label without a window, and per clip without a window."""
    items, seen_clips = [], set()
    for w in windows:
        wid = make_window_id(w)
        seen_clips.add(w["video_name"])
        items.append({"id": wid, "clip": w["video_name"], "track": w["track_id"],
                      "start": w["start_frame"], "end": w["end_frame"],
                      "kind": "used" if wid in filtered_ids else "filtered_out",
                      "label": labels.get(wid), "excluded": wid in excluded})
    have = {i["id"] for i in items}
    for wid, label in labels.items():
        if wid not in have:
            items.append({"id": wid, "clip": clip_of(wid), "track": None, "start": None,
                          "end": None, "kind": "missing", "label": label,
                          "excluded": wid in excluded})
    for clip in clips:
        if clip not in seen_clips:
            items.append({"id": clip, "clip": clip, "track": None, "start": None, "end": None,
                          "kind": "no_window", "label": None, "excluded": False})
    items.sort(key=lambda i: (i["clip"], i["id"]))
    return items


def valid_reviewer(name):
    return bool(_REVIEWER.match(name or ""))


def append_review(review_dir, reviewer, window_id, label):
    if not valid_reviewer(reviewer):
        raise ValueError(f"bad reviewer name: {reviewer!r}")
    if label not in LABELS + (SKIP,):
        raise ValueError(f"bad label: {label!r}")
    os.makedirs(review_dir, exist_ok=True)
    path = os.path.join(review_dir, f"{reviewer}.csv")
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["window_id", "label", "time"])
        writer.writerow([window_id, label, datetime.now().isoformat(timespec="seconds")])


def read_reviews(review_dir):
    """{reviewer: {window_id: latest label}}."""
    out = {}
    for path in sorted(glob.glob(os.path.join(str(review_dir), "*.csv"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if not valid_reviewer(name):
            continue
        with open(path, newline="") as f:
            out[name] = {r["window_id"]: r["label"] for r in csv.DictReader(f)}
    return out


def read_labels(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        return {row[0]: row[1] for row in reader if len(row) == 2}


def read_excluded(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {r["window_id"] for r in csv.DictReader(f)}


def read_provenance(path):
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {r["window_id"]: f"{r['resolution']}: {r['judgements']}" for r in csv.DictReader(f)}


def plan_apply(labels, decisions, used_ids, excluded=frozenset()):
    """(changes [(id, old, new)], skipped [(id, reason)]) for one reviewer's decisions."""
    changes, skipped = [], []
    for wid, new in sorted(decisions.items()):
        old = labels.get(wid)
        if new == SKIP:
            skipped.append((wid, SKIP))
        elif wid not in used_ids:
            skipped.append((wid, "window not used in training"))
        elif wid in excluded:
            skipped.append((wid, "listed in excluded_windows.csv"))
        elif new != old:
            changes.append((wid, old, new))
    return changes, skipped


def apply_changes(annotations_csv, changes, log_path, reviewer):
    """Rewrite labels in place, append new ones, and log every change."""
    new_label = {wid: new for wid, _, new in changes}
    with open(annotations_csv, newline="") as f:
        rows = list(csv.reader(f))
    present = set()
    for row in rows[1:]:
        if len(row) == 2 and row[0] in new_label:
            row[1] = new_label[row[0]]
            present.add(row[0])
    rows += [[wid, new] for wid, _, new in changes if wid not in present]
    tmp = f"{annotations_csv}.tmp"
    with open(tmp, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    os.replace(tmp, annotations_csv)
    os.makedirs(os.path.dirname(str(log_path)) or ".", exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["window_id", "old_label", "new_label", "reviewer"])
        writer.writerows([wid, old or "", new, reviewer] for wid, old, new in changes)


def parse_range(header, size):
    """(start, end) inclusive for a `Range: bytes=` header, None without one, 'invalid' if unusable."""
    if not header:
        return None
    m = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
    if not m or (not m.group(1) and not m.group(2)):
        return "invalid"
    if not m.group(1):
        start, end = max(0, size - int(m.group(2))), size - 1
    else:
        start = int(m.group(1))
        end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
    if start >= size or start > end:
        return "invalid"
    return start, end


def pct_rank(values):
    """Percentile rank in (0, 1], ties averaged."""
    s = sorted(values)
    n = len(s)
    return [(bisect.bisect_left(s, v) + (bisect.bisect_right(s, v) - bisect.bisect_left(s, v) + 1) / 2) / n
            for v in values]


def suspicion_scores(labels, p_dis, conf):
    """Per Focused/Distracted window: mean of two within-label percentile ranks,
    the models' probability of the other class and low keypoint confidence."""
    out = {}
    for cls in ("Focused", "Distracted"):
        ids = [w for w, lab in labels.items() if lab == cls and w in conf]
        p_other = {w: (p_dis[w] if cls == "Focused" else 1 - p_dis[w]) if w in p_dis else None
                   for w in ids}
        r_model = pct_rank([0.5 if p_other[w] is None else p_other[w] for w in ids])
        r_conf = pct_rank([-conf[w] for w in ids])
        for w, a, b in zip(ids, r_model, r_conf):
            out[w] = {"label": cls, "score": (a + b) / 2, "p_other": p_other[w], "conf": conf[w]}
    return out


def mean_p_distracted(runs_dir):
    """{window_id: P(Distracted)} averaged over every saved oof_probs.pt; and the run count."""
    import torch
    paths = sorted(glob.glob(os.path.join(runs_dir, "*", "*", "oof_probs.pt")))
    acc = {}
    for p in paths:
        d = torch.load(p, weights_only=False)
        for wid, pr in zip(d["window_ids"], d["oof_prob"]):
            acc.setdefault(wid, []).append(float(pr[1]))
    return {w: sum(v) / len(v) for w, v in acc.items()}, len(paths)


def write_suspicion(scores, n_runs, path):
    rows = []
    for cls in ("Focused", "Distracted"):
        ids = sorted((w for w, v in scores.items() if v["label"] == cls), key=lambda w: -scores[w]["score"])
        for rank, w in enumerate(ids, 1):
            v = scores[w]
            rows.append([w, cls, rank, len(ids), f"{v['score']:.4f}",
                         "" if v["p_other"] is None else f"{v['p_other']:.4f}", f"{v['conf']:.4f}", n_runs])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["window_id", "label", "rank", "of", "score", "p_other_class", "mean_conf13", "n_runs"])
        writer.writerows(rows)


def read_suspicion(path):
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {r["window_id"]: {"label": r["label"], "rank": int(r["rank"]), "of": int(r["of"]), "score": float(r["score"]),
                                 "p_other": float(r["p_other_class"]) if r["p_other_class"] else None,
                                 "conf": float(r["mean_conf13"]), "n_runs": int(r["n_runs"])}
                for r in csv.DictReader(f)}


def load_windows(pt_dir):
    import torch
    out = []
    for pt in sorted(glob.glob(os.path.join(pt_dir, "*_windows.pt"))):
        out.extend(torch.load(pt, weights_only=False))
    return out


@functools.lru_cache(maxsize=None)
def clip_fps(video_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps or 25.0


def window_payload(w, fps):
    k = w["keypoints"].float().permute(1, 2, 0).tolist()
    return {"fps": fps, "start": w["start_frame"], "end": w["end_frame"],
            "kps": [[[round(x, 1), round(y, 1), round(c, 2)] for x, y, c in frame] for frame in k],
            "boxes": [[round(float(v), 1) for v in b] for b in w["bboxes"]]}


class App:
    def __init__(self, args):
        t0 = time.time()
        print("loading windows ...", flush=True)
        windows = load_windows(args.windows_dir)
        filtered = {make_window_id(w) for w in load_windows(args.filtered_dir)}
        self.videos_dir = args.videos_dir
        clips = sorted(os.path.splitext(p)[0] for p in os.listdir(args.videos_dir) if p.endswith(".mp4"))
        self.clips = set(clips)
        self.items = build_index(windows, filtered, clips, read_labels(args.annotations),
                                 read_excluded(args.excluded))
        self.windows = {make_window_id(w): w for w in windows}
        sus = read_suspicion(args.suspicion)
        for i in self.items:
            if i["id"] in sus and sus[i["id"]]["label"] == i["label"]:
                i["sus"] = sus[i["id"]]
        self.provenance = read_provenance(args.provenance)
        self.labelable = {i["id"] for i in self.items if i["kind"] in ("used", "filtered_out")}
        self.review_dir = args.review_dir
        self.lock = threading.Lock()
        kinds = {}
        for i in self.items:
            kinds[i["kind"]] = kinds.get(i["kind"], 0) + 1
        print(f"{len(self.items)} items {kinds} in {time.time() - t0:.0f}s", flush=True)

    def index(self):
        return {"items": self.items, "provenance": self.provenance,
                "reviews": read_reviews(self.review_dir), "labels": list(LABELS) + [SKIP]}

    def window(self, wid):
        w = self.windows[wid]
        return window_payload(w, clip_fps(os.path.join(self.videos_dir, f"{w['video_name']}.mp4")))

    def review(self, body):
        wid, reviewer, label = body.get("window_id"), body.get("reviewer"), body.get("label")
        if wid not in self.labelable:
            raise ValueError(f"not a reviewable window: {wid}")
        with self.lock:
            append_review(self.review_dir, reviewer, wid, label)


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, status, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                with open(UI_FILE, "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if url.path == "/api/index":
                return self._send(200, app.index())
            if url.path == "/api/window":
                wid = parse_qs(url.query).get("id", [""])[0]
                if wid not in app.windows:
                    return self._send(404, {"error": f"no window {wid}"})
                return self._send(200, app.window(wid))
            if url.path.startswith("/video/") and url.path.endswith(".mp4"):
                return self._video(url.path[len("/video/"):-len(".mp4")])
            self._send(404, {"error": "not found"})

        def _video(self, clip):
            if clip not in app.clips:
                return self._send(404, {"error": f"no clip {clip}"})
            path = os.path.join(app.videos_dir, f"{clip}.mp4")
            size = os.path.getsize(path)
            rng = parse_range(self.headers.get("Range"), size)
            if rng == "invalid":
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            start, end = rng or (0, size - 1)
            self.send_response(206 if rng else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                try:
                    self.wfile.write(f.read(end - start + 1))
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def do_POST(self):
            if urlparse(self.path).path != "/api/review":
                return self._send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                app.review(body)
            except (ValueError, json.JSONDecodeError) as e:
                return self._send(400, {"error": str(e)})
            self._send(200, {"ok": True})

    return Handler


def cmd_serve(args):
    app = App(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"review UI: http://{args.host}:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


def cmd_flag(args):
    labels = read_labels(args.annotations)
    conf = {make_window_id(w): float(w["keypoints"][2, :, :13].float().mean())
            for w in load_windows(args.filtered_dir)}
    p_dis, n_runs = mean_p_distracted(args.runs_dir)
    if not n_runs:
        raise SystemExit(f"no oof_probs.pt under {args.runs_dir}")
    scores = suspicion_scores(labels, p_dis, conf)
    write_suspicion(scores, n_runs, args.suspicion)
    n = {c: sum(v["label"] == c for v in scores.values()) for c in ("Focused", "Distracted")}
    print(f"{n_runs} runs averaged; scored {n['Focused']} Focused, {n['Distracted']} Distracted "
          f"-> {args.suspicion}")


def cmd_apply(args):
    decisions = read_reviews(args.review_dir).get(args.reviewer)
    if not decisions:
        raise SystemExit(f"no reviews for {args.reviewer!r} in {args.review_dir}")
    used = {make_window_id(w) for w in load_windows(args.filtered_dir)}
    labels = read_labels(args.annotations)
    changes, skipped = plan_apply(labels, decisions, used, read_excluded(args.excluded))
    print(f"{args.reviewer}: {len(decisions)} decisions, {len(changes)} label changes, "
          f"{len(skipped)} skipped")
    for wid, old, new in changes:
        print(f"  {wid}: {old or '(unlabeled)'} -> {new}")
    for wid, reason in skipped:
        print(f"  skip {wid}: {reason}")
    if not args.write:
        print("dry run: nothing written (add --write)")
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = os.path.join(args.review_dir, f"applied_{args.reviewer}_{stamp}.log.csv")
    apply_changes(args.annotations, changes, log, args.reviewer)
    print(f"annotations.csv updated; change log -> {log}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows-dir", default=SAV_WINDOWS_DIR)
    ap.add_argument("--filtered-dir", default=SAV_WINDOWS_FILTERED_DIR)
    ap.add_argument("--videos-dir", default=SAV_VIDEOS_DIR)
    ap.add_argument("--annotations", default=ANNOTATIONS_FILE)
    ap.add_argument("--excluded", default=EXCLUDED_WINDOWS_FILE)
    ap.add_argument("--provenance", default=SAV_LABEL_PROVENANCE_FILE)
    ap.add_argument("--review-dir", default=SAV_REVIEW_DIR)
    ap.add_argument("--suspicion", default=SUSPICION_FILE)
    ap.add_argument("--runs-dir", default=runs_dir())
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("serve")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    sub.add_parser("flag")
    p = sub.add_parser("apply")
    p.add_argument("--reviewer", required=True)
    p.add_argument("--write", action="store_true")
    args = ap.parse_args()
    {"serve": cmd_serve, "flag": cmd_flag, "apply": cmd_apply}[args.cmd](args)


if __name__ == "__main__":
    main()

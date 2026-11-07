"""Paper figures, regenerated from the data on disk.

  candidates      rank clips for Fig. 1a and write a contact sheet to choose from
  fig1 --clip ID  render Fig. 1a for one clip at full resolution, and Fig. 1 (a above b)
  fig2            per-joint visibility box plot (reads joint_visibility_samples.npy)
  fig3b           the full COCO-17 graph beside the 13-joint visible graph
  fig3            the architecture with the diagnostic ablation (a) beside fig3b (b)

Every frame is blurred whole before anything is drawn: the tracker recovers
about one student in five, so blurring only detected faces would leave the
rest recognizable. Check each render at 100% zoom before it goes in a paper.

Output: outputs/figures/paper/

Usage:  .venv/bin/python analysis/make_paper_figs.py candidates [--top 20]
        .venv/bin/python analysis/make_paper_figs.py fig1 --clip 20211203_04_594
        .venv/bin/python analysis/make_paper_figs.py fig2
        .venv/bin/python analysis/make_paper_figs.py fig3b
        .venv/bin/python analysis/make_paper_figs.py fig3
"""
import argparse
import collections
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path[:0] = ["lib", "steps"]
import paths  # noqa: E402
from model import COCO_EDGES, OBSERVED_JOINTS_13, get_coco_edges  # noqa: E402

VISIBLE = 0.35
LOWER = (13, 14, 15, 16)
OUT = os.path.join(paths.figures_dir(), "paper")
AGREEMENT = {"unanimous": 2, "majority": 1, "single": 0}
# Okabe-Ito blue, orange and reddish purple: distinct under the common colour-vision deficiencies.
ON_TASK, OFF_TASK, SKELETON, HIDDEN = (178, 114, 0), (0, 159, 230), (240, 240, 240), (167, 121, 204)
PURPLE, GREY = "#CC79A7", "#7F7F7F"
TEXT_WIDTH_IN = 12.2 / 2.54                       # LNCS text width
JOINT_LABELS = ["nose", "L eye", "R eye", "L ear", "R ear", "L shoulder", "R shoulder",
                "L elbow", "R elbow", "L wrist", "R wrist", "L hip", "R hip",
                "L knee", "R knee", "L ankle", "R ankle"]
# A front-facing COCO-17 layout; the subject's left is on the viewer's right.
LAYOUT = [(0, 1.72), (0.07, 1.79), (-0.07, 1.79), (0.16, 1.74), (-0.16, 1.74),
          (0.34, 1.42), (-0.34, 1.42), (0.46, 1.08), (-0.46, 1.08), (0.54, 0.76),
          (-0.54, 0.76), (0.2, 0.8), (-0.2, 0.8), (0.22, 0.4), (-0.22, 0.4),
          (0.22, 0.0), (-0.22, 0.0)]


def clip_of(window_id):
    return window_id.split("_id_")[0]


def track_of(window_id):
    return window_id.split("_id_")[1].split("_")[0]


def load_samples():
    """{clip: [(window_id, label, resolution)]} over every labeled window."""
    labels = {r["window_id"]: r["label"] for r in csv.DictReader(open(paths.ANNOTATIONS_FILE))}
    prov = {r["window_id"]: r["resolution"] for r in csv.DictReader(
        open(os.path.join(paths.SAV_DIR, "label_provenance.csv")))}
    by_clip = collections.defaultdict(list)
    for w, lab in labels.items():
        by_clip[clip_of(w)].append((w, lab, prov.get(w, "single")))
    return by_clip


def load_timeline(clip):
    with open(os.path.join(paths.SAV_POSE_TRACKS_DIR, f"{clip}_timeline.jsonl")) as f:
        return [json.loads(line) for line in f]


def pick(samples, label):
    """The best-agreed sample with this label, or None."""
    cands = [s for s in samples if s[1] == label]
    return max(cands, key=lambda s: AGREEMENT.get(s[2], 0)) if cands else None


def best_frame(timeline, tracks):
    """The frame nearest the middle in which every highlighted track is present."""
    mid = len(timeline) // 2
    order = sorted(range(len(timeline)), key=lambda i: abs(i - mid))
    for i in order:
        if all(t in timeline[i]["students"] for t in tracks):
            return i
    return mid


def lower_body_confidence(timeline, tracks):
    vals = [timeline[i]["students"][t]["keypoints"][j][2]
            for i in range(len(timeline)) for t in tracks
            if t in timeline[i]["students"] for j in LOWER]
    return float(np.mean(vals)) if vals else 1.0


def rank_candidates(by_clip, min_students=6):
    rows = []
    for clip, samples in by_clip.items():
        on, off = pick(samples, "Focused"), pick(samples, "Distracted")
        if len(samples) < min_students or on is None or off is None:
            continue
        timeline = load_timeline(clip)
        tracks = [track_of(s[0]) for s in samples]
        rows.append({"clip": clip, "students": len(samples), "on": on, "off": off,
                     "agreement": AGREEMENT[on[2]] + AGREEMENT[off[2]],
                     "lower_conf": lower_body_confidence(timeline, tracks)})
    rows.sort(key=lambda r: (-r["agreement"], -r["students"], r["lower_conf"]))
    return rows


def blur(frame, strength):
    k = int(strength) * 2 + 1
    return cv2.GaussianBlur(frame, (k, k), 0)


def draw_skeleton(img, kps, scale):
    pts = [(int(x), int(y), c) for x, y, c in kps]
    for a, b in COCO_EDGES:
        pa, pb = pts[a], pts[b]
        seen = pa[2] >= VISIBLE and pb[2] >= VISIBLE
        if seen:
            cv2.line(img, pa[:2], pb[:2], SKELETON, max(1, 2 * scale), cv2.LINE_AA)
        else:
            _dashed(img, pa[:2], pb[:2], HIDDEN, 2 * scale)
    for x, y, c in pts:
        r = 3 * scale
        if c >= VISIBLE:
            cv2.circle(img, (x, y), r, SKELETON, -1, cv2.LINE_AA)
        else:
            cv2.circle(img, (x, y), r + scale, HIDDEN, 2 * scale, cv2.LINE_AA)


def _dashed(img, p, q, color, thickness, dash=6):
    p, q = np.array(p, float), np.array(q, float)
    n = max(1, int(np.linalg.norm(q - p) // dash))
    for i in range(0, n, 2):
        a = p + (q - p) * i / n
        b = p + (q - p) * min(i + 1, n) / n
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), color, thickness, cv2.LINE_AA)


def draw_box(img, bbox, color, text, scale):
    cx, cy, w, h = bbox
    x1, y1, x2, y2 = int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2 * scale, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, scale)
    cv2.rectangle(img, (x1, y1 - th - 8 * scale), (x1 + tw + 6 * scale, y1), color, -1)
    cv2.putText(img, text, (x1 + 3 * scale, y1 - 5 * scale), cv2.FONT_HERSHEY_SIMPLEX,
                0.6 * scale, (255, 255, 255), scale, cv2.LINE_AA)


def render(clip, samples, on, off, blur_strength=24, scale=1):
    timeline = load_timeline(clip)
    highlight = {track_of(on[0]): (ON_TASK, "on-task"), track_of(off[0]): (OFF_TASK, "off-task")}
    idx = best_frame(timeline, list(highlight))
    cap = cv2.VideoCapture(os.path.join(paths.SAV_VIDEOS_DIR, f"{clip}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, timeline[idx]["frame_idx"])
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"{clip}: could not read frame {idx}")
    img = blur(frame, blur_strength)
    if scale > 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    students = timeline[idx]["students"]
    trainable = {track_of(s[0]) for s in samples if s[1] in ("Focused", "Distracted")}
    for t in trainable & set(students):
        kps = [(x * scale, y * scale, c) for x, y, c in students[t]["keypoints"]]
        draw_skeleton(img, kps, scale)
    for t, (color, text) in highlight.items():
        if t in students:
            draw_box(img, [v * scale for v in students[t]["bbox"]], color, text, scale)
    return img


def candidates(args):
    os.makedirs(os.path.join(OUT, "fig1_candidates"), exist_ok=True)
    by_clip = load_samples()
    rows = rank_candidates(by_clip)[:args.top]
    tiles = []
    print(f"{'#':>3}  {'clip':<20}{'students':>9}  {'on-task label':<16}"
          f"{'off-task label':<16}{'lower-body conf':>16}")
    for n, r in enumerate(rows, 1):
        img = render(r["clip"], by_clip[r["clip"]], r["on"], r["off"])
        cv2.imwrite(os.path.join(OUT, "fig1_candidates", f"{n:02d}_{r['clip']}.png"), img)
        tile = cv2.resize(img, (640, 360))
        cv2.rectangle(tile, (0, 0), (640, 28), (0, 0, 0), -1)
        cv2.putText(tile, f"#{n}  {r['clip']}  on:{r['on'][2]} off:{r['off'][2]}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
        print(f"{n:>3}  {r['clip']:<20}{r['students']:>9}  {r['on'][2]:<16}"
              f"{r['off'][2]:<16}{r['lower_conf']:>16.3f}")
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    out = os.path.join(OUT, "fig1_candidates.png")
    cv2.imwrite(out, sheet)
    print(f"\ncontact sheet: {os.path.relpath(out, paths.REPO_ROOT)}")
    print(f"full-size renders: {os.path.relpath(os.path.join(OUT, 'fig1_candidates'), paths.REPO_ROOT)}/")


def fig1(args):
    os.makedirs(OUT, exist_ok=True)
    by_clip = load_samples()
    samples = by_clip[args.clip]
    on = next(s for s in samples if s[0] == args.on) if args.on else pick(samples, "Focused")
    off = next(s for s in samples if s[0] == args.off) if args.off else pick(samples, "Distracted")
    img = render(args.clip, samples, on, off, blur_strength=args.blur, scale=2)
    out = os.path.join(OUT, "fig1a.png")
    cv2.imwrite(out, img)
    print(f"on-task  {on[0]} ({on[2]})\noff-task {off[0]} ({off[2]})")
    print(f"wrote {os.path.relpath(out, paths.REPO_ROOT)}")
    _fig1_composite(args.clip, samples, off, img, args.blur)


def _student_track(clip, window_id):
    """(T, 17, 3) keypoints of one student across the clip, pixel coordinates."""
    track = track_of(window_id)
    return np.array([f["students"][track]["keypoints"] for f in load_timeline(clip)
                     if track in f["students"]], dtype=float)


def _fig1_composite(clip, samples, student, fig1a, blur_strength):
    """Fig. 1: (a) the annotated frame above (b) the pipeline strip."""
    plt = _paper_style()
    timeline = load_timeline(clip)
    cap = cv2.VideoCapture(os.path.join(paths.SAV_VIDEOS_DIR, f"{clip}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, timeline[len(timeline) // 2]["frame_idx"])
    ok, raw = cap.read()
    cap.release()
    raw = cv2.cvtColor(blur(raw, blur_strength), cv2.COLOR_BGR2RGB)
    kps = _student_track(clip, student[0])

    fig = plt.figure(figsize=(TEXT_WIDTH_IN, 4.4))
    grid = fig.add_gridspec(2, 4, height_ratios=[2.6, 1], hspace=0.28, wspace=0.14)
    top = fig.add_subplot(grid[0, :])
    top.imshow(cv2.cvtColor(fig1a, cv2.COLOR_BGR2RGB))
    top.set_title("(a) a classroom clip, blurred, with the trainable students' skeletons",
                  loc="left", pad=3)
    top.axis("off")

    tiles = [fig.add_subplot(grid[1, k]) for k in range(4)]
    tiles[0].imshow(raw)
    tiles[0].set_title("3-s clip", pad=2, fontsize=7)
    tiles[1].imshow(cv2.cvtColor(render(clip, samples, student, student, blur_strength),
                                 cv2.COLOR_BGR2RGB))
    tiles[1].set_title("pose + tracking", pad=2, fontsize=7)
    ax = tiles[2]
    idx = np.linspace(0, len(kps) - 1, 3).astype(int)
    for n, t in enumerate(idx):
        pts = kps[t]
        for a, b in COCO_EDGES:
            seen = pts[a, 2] >= VISIBLE and pts[b, 2] >= VISIBLE
            ax.plot(pts[[a, b], 0], pts[[a, b], 1], color="black" if seen else PURPLE,
                    linestyle="-" if seen else (0, (2, 1.5)), linewidth=0.8,
                    alpha=0.35 + 0.3 * n)
    ax.invert_yaxis()
    ax.set_box_aspect(raw.shape[0] / raw.shape[1])
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("one student", pad=2, fontsize=7)
    tiles[3].set_box_aspect(raw.shape[0] / raw.shape[1])
    tiles[3].text(0.5, 0.5, "on-task\noff-task\nrejected", ha="center", va="center",
                  fontsize=7, linespacing=1.3,
                  bbox={"boxstyle": "round,pad=0.4", "facecolor": "#EFEFEF", "lw": 0.6})
    tiles[3].set_title("label", pad=2, fontsize=7)
    for t in tiles:
        t.axis("off")
    for k in range(3):
        fig.add_artist(plt.matplotlib.patches.ConnectionPatch(
            (1.02, 0.5), (-0.02, 0.5), "axes fraction", "axes fraction",
            axesA=tiles[k], axesB=tiles[k + 1], arrowstyle="-|>", lw=0.7))
    tiles[0].text(0.0, 1.38, "(b) the pipeline", transform=tiles[0].transAxes, fontsize=8)
    _save(fig, "fig1")


def _paper_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    return plt


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    print(f"wrote {os.path.relpath(os.path.join(OUT, name), paths.REPO_ROOT)}.pdf / .png")


def fig2(args):
    plt = _paper_style()
    conf = np.load(os.path.join(paths.diagnostics_dir(), "joint_visibility_samples.npy"))
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.3))
    box = ax.boxplot([conf[:, j] for j in range(17)], whis=(5, 95), showfliers=False,
                     showmeans=True, patch_artist=True, widths=0.6,
                     medianprops={"color": "black", "linewidth": 1},
                     meanprops={"marker": "o", "markersize": 3, "markerfacecolor": "white",
                                "markeredgecolor": "black", "markeredgewidth": 0.6},
                     whiskerprops={"linewidth": 0.8}, capprops={"linewidth": 0.8})
    for j, patch in enumerate(box["boxes"]):
        hidden = j not in OBSERVED_JOINTS_13
        patch.set(facecolor=PURPLE if hidden else "#D9D9D9",
                  edgecolor="black", linewidth=0.6, alpha=0.9)
    threshold = ax.axhline(VISIBLE, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(range(1, 18), JOINT_LABELS, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("detector confidence\n(per-sample mean)")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0.4, 17.6)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="#D9D9D9", edgecolor="black", lw=0.6),
               plt.Rectangle((0, 0), 1, 1, facecolor=PURPLE, edgecolor="black", lw=0.6),
               threshold]
    # The space under the elbows-to-hips whiskers is empty (all whiskers end above 0.36).
    ax.legend(handles, ["visible (13)", "invisible (4)", "visibility threshold (0.35)"],
              loc="lower center", bbox_to_anchor=(0.54, 0.0), frameon=False, ncol=1)
    _save(fig, "fig2")


def _draw_graphs(plt, axes):
    """The full COCO-17 skeleton beside the 13-joint visible graph, on two axes."""
    visible = set(OBSERVED_JOINTS_13)
    panels = [("full skeleton\n(COCO, 17 joints)", list(range(17)), COCO_EDGES),
              ("visible-joint graph\n(13 joints)", OBSERVED_JOINTS_13,
               [(OBSERVED_JOINTS_13[a], OBSERVED_JOINTS_13[b])
                for a, b in get_coco_edges(OBSERVED_JOINTS_13)])]
    for ax, (title, nodes, edges) in zip(axes, panels):
        for a, b in edges:
            (xa, ya), (xb, yb) = LAYOUT[a], LAYOUT[b]
            seen = a in visible and b in visible
            ax.plot([xa, xb], [ya, yb], color="black" if seen else PURPLE,
                    linestyle="-" if seen else (0, (2.5, 1.5)), linewidth=1.1, zorder=1)
        for j in nodes:
            x, y = LAYOUT[j]
            if j in visible:
                ax.scatter(x, y, s=10, color="black", zorder=2)
            else:
                ax.scatter(x, y, s=16, facecolor="white", edgecolor=PURPLE,
                           linewidth=1.1, zorder=2)
        ax.set_title(title, pad=4)
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.12, 1.92)
        ax.set_aspect("equal")
        ax.axis("off")
    return [plt.Line2D([], [], color="black", marker="o", markersize=3, lw=1.1),
            plt.Line2D([], [], color=PURPLE, marker="o", markersize=4, lw=1.1,
                       linestyle=(0, (2.5, 1.5)), markerfacecolor="white")]


GRAPH_LEGEND = ["visible joint and limb", "invisible joint (knee, ankle: confidence < 0.35)"]

# The diagnostic ablation, placed under the layer each intervention acts on:
# (stage, label, macro-F1 difference vs the full-skeleton ST-GCN).
STAGES = ["input\nx, y, confidence\n90 × 17 joints", "input\nbatch norm",
          "3 × (graph +\ntemporal conv)", "global avg.\npooling", "linear →\non/off-task"]
INTERVENTIONS = [(0, "visible-joint\ngraph", "E1"),
                 (1, "input\nmasking", "E6"), (1, "per-joint\nnormalization", "E7"),
                 (2, "confidence-weighted\nadjacency", "E2"),
                 (3, "visible-joint\npooling", "E5")]
VERDICT_COLOR = {"helps": "#0072B2", "hurts": "#D55E00", "no effect": GREY}


def macro_f1_vs_c(arm):
    """Macro-F1 of an arm minus the full-skeleton C, same estimator as Table 2."""
    from e_ladder_table import arm_stats
    from paired_ci import ARM
    return arm_stats(ARM[arm])["macro_f1"][0] - arm_stats(ARM["C"])["macro_f1"][0]


def _verdict(diff):
    return "helps" if diff > 0.006 else "hurts" if diff < -0.006 else "no effect"


def _draw_architecture(plt, ax):
    from matplotlib.patches import FancyBboxPatch
    xs = np.linspace(0.1, 0.9, len(STAGES))
    w, h, y = 0.17, 0.3, 0.83
    for k, (x, text) in enumerate(zip(xs, STAGES)):
        hot = k == 1
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.005,rounding_size=0.02",
                                    facecolor="#F2D4E4" if hot else "#EFEFEF",
                                    edgecolor=PURPLE if hot else "black",
                                    linewidth=1.2 if hot else 0.6))
        ax.text(x, y, text, ha="center", va="center", fontsize=6, linespacing=1.1)
        if k:
            ax.annotate("", xy=(x - w / 2, y), xytext=(xs[k - 1] + w / 2 + 0.002, y),
                        arrowprops={"arrowstyle": "-|>", "lw": 0.7, "color": "black"})
    below = {}
    for stage, label, arm in INTERVENTIONS:
        diff = macro_f1_vs_c(arm)
        verdict = _verdict(diff)
        ty = y - h / 2 - 0.06 - below.get(stage, 0)
        below[stage] = below.get(stage, 0) + 0.3
        number = f"{diff:+.4f}".replace("-", "\u2212")
        text = f"{label}\n{number}\n{verdict}"
        ax.text(xs[stage], ty, text, ha="center", va="top", fontsize=5.8,
                color=VERDICT_COLOR[verdict], fontweight="bold" if stage == 0 else "normal",
                linespacing=1.05)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def fig3b(args):
    plt = _paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN * 0.62, 2.3))
    handles = _draw_graphs(plt, axes)
    fig.legend(handles, GRAPH_LEGEND, loc="lower center", ncol=1, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    _save(fig, "fig3b")


def fig3(args):
    plt = _paper_style()
    fig = plt.figure(figsize=(TEXT_WIDTH_IN, 3.4))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.5, 1], width_ratios=[1, 1, 1.25],
                            hspace=0.12, wspace=0.0)
    arch = fig.add_subplot(grid[0, :])
    _draw_architecture(plt, arch)
    arch.set_title("(a) where the invisible joints could act; macro-F1 change vs the full skeleton",
                   loc="left", pad=2, fontsize=7.5)
    graphs = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    handles = _draw_graphs(plt, graphs)
    key = fig.add_subplot(grid[1, 2])
    key.axis("off")
    key.legend(handles, ["visible joint\nand limb", "invisible joint\n(knee, ankle:\n"
                         "confidence < 0.35)"], loc="center left", frameon=False,
               fontsize=6.5, labelspacing=1.2)
    graphs[0].set_title("(b) full skeleton\n(COCO, 17 joints)", pad=4)
    _save(fig, "fig3")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--top", type=int, default=20)
    f = sub.add_parser("fig1")
    f.add_argument("--clip", required=True)
    f.add_argument("--on", help="window id of the on-task student (default: best agreed)")
    f.add_argument("--off", help="window id of the off-task student (default: best agreed)")
    f.add_argument("--blur", type=int, default=24, help="Gaussian blur radius in pixels")
    sub.add_parser("fig2")
    sub.add_parser("fig3b")
    sub.add_parser("fig3")
    args = ap.parse_args()
    {"candidates": candidates, "fig1": fig1, "fig2": fig2, "fig3b": fig3b,
     "fig3": fig3}[args.cmd](args)


if __name__ == "__main__":
    main()

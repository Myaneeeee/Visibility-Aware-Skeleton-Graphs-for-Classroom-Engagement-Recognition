import os
import glob
import cv2
import torch
import csv

import _bootstrap  # noqa: F401  -- puts ../lib on sys.path
import common

# ==========================================
# 1. PATHS & CONFIGURATION
# ==========================================
from paths import (
    SAV_WINDOWS_FILTERED_DIR as UNLABELED_DIR,
    SAV_VIDEOS_DIR as VIDEO_DIR,
    ANNOTATIONS_FILE,
)

def resolve_paths(unlabeled_dir=None, video_dir=None, annotations_file=None):
    return (unlabeled_dir or UNLABELED_DIR,
            video_dir or VIDEO_DIR,
            annotations_file or ANNOTATIONS_FILE)

# Streamlined binary classification map according to lecturer feedback
CLASS_MAP = {
    ord('1'): "Focused",
    ord('2'): "Distracted",
    ord('8'): "TRASH", # Will be ignored during training
    ord('9'): "QUIT"
}

# Written to the ledger when the annotator declines to decide. It is a *record*,
# not a class: LABELING_GUIDELINES §9 Phase C adjudicates every Skip, which needs
# a list of them, and a resumed session should not re-offer a window whose
# verdict was already "I cannot tell". Excluded from training (common.CLASS_TO_IDX)
# and from agreement (annotation_report._VALID).
SKIP_LABEL = "Skip"


def resolve_key(key):
    """Map a raw ``cv2.waitKey`` code to an action, so key handling is testable
    without a GUI. Letter keys are case-insensitive -- the on-screen legend says
    'D', and a caps-locked annotator pressing it must still skip."""
    if key < 0 or key > 127:            # -1 / 255 = "no key pressed"
        return None
    if key == 32:
        return "toggle"
    lowered = ord(chr(key).lower())
    if lowered == ord('d'):
        return "skip"
    return CLASS_MAP.get(lowered)

# ==========================================
# 2. STATE MANAGEMENT & I/O
# ==========================================
def load_existing_annotations(annotations_file):
    """Loads already judged windows so we don't start from scratch if the app
    crashes. 'Judged' includes Skips -- see SKIP_LABEL."""
    labeled_ids = set()
    if os.path.exists(annotations_file):
        with open(annotations_file, mode='r', newline='') as f:
            reader = csv.reader(f)
            next(reader, None)          # header; save_annotation guarantees one
            for row in reader:
                if row:
                    labeled_ids.add(row[0]) # row[0] is the unique window ID
    return labeled_ids

def save_annotation(window_id, label, annotations_file):
    """Instantly appends the label to disk. Crash-proof."""
    parent = os.path.dirname(annotations_file)
    if parent:
        # Per-annotator ledgers live in their own directory, which may not
        # not exist yet. Creating it here means a forgotten mkdir costs nothing;
        # without it the annotator's *first* keypress raises FileNotFoundError.
        os.makedirs(parent, exist_ok=True)
    # Size, not existence: an existing-but-empty file (interrupted run, `touch`)
    # would otherwise never get a header, and every reader in the repo skips row
    # 1 -- so the first real label would vanish silently.
    needs_header = (not os.path.exists(annotations_file)
                    or os.path.getsize(annotations_file) == 0)
    with open(annotations_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if needs_header:
            writer.writerow(["window_id", "label"]) # Write header
        writer.writerow([window_id, label])

def load_worklist_ids(path):
    """Load window_ids (one per line) restricting the queue, or None.
    Returns None if no path is given, so callers can distinguish "no filter"
    from "filter to an empty set"."""
    if path is None:
        return None
    ids = set()
    with open(path, mode='r') as f:
        for line in f:
            wid = line.strip()
            if wid:
                ids.add(wid)
    return ids

# ==========================================
# 3. PREFLIGHT
# ==========================================
def preflight(unlabeled_dir=None, video_dir=None, annotations_file=None,
              worklist_file=None):
    """Verify a labeling setup without opening a window. Returns (ok, problems).

    The clips and windows are ~377 MB copied to each annotator's machine by hand,
    so the realistic failure is a missing directory or a wrong relative path.
    Catching that in one command beats discovering it ten windows into a session.
    """
    UNLABELED_DIR, VIDEO_DIR, ANNOTATIONS_FILE = resolve_paths(
        unlabeled_dir, video_dir, annotations_file)
    problems = []

    pt_files = sorted(glob.glob(os.path.join(UNLABELED_DIR, "*_windows.pt")))
    if not os.path.isdir(UNLABELED_DIR):
        problems.append(f"windows dir does not exist: {UNLABELED_DIR}")
    elif not pt_files:
        problems.append(f"no *_windows.pt in {UNLABELED_DIR}")
    if not os.path.isdir(VIDEO_DIR):
        problems.append(f"video dir does not exist: {VIDEO_DIR}")

    on_disk = {}
    for pt_path in pt_files:
        for w in torch.load(pt_path, weights_only=False):
            on_disk[common.make_window_id(w)] = w

    wanted = load_worklist_ids(worklist_file) if worklist_file else None
    if worklist_file and not os.path.exists(worklist_file):
        problems.append(f"list file does not exist: {worklist_file}")
        wanted = None

    queue_ids = sorted(wanted & set(on_disk)) if wanted is not None else sorted(on_disk)
    if wanted is not None:
        for missing in sorted(wanted - set(on_disk)):
            problems.append(f"listed id has no window on disk: {missing}")

    for wid in queue_ids:
        clip = on_disk[wid]["video_name"]
        if not os.path.exists(os.path.join(VIDEO_DIR, f"{clip}.mp4")):
            problems.append(f"missing clip video: {clip}.mp4")

    # Writability: the ledger's directory must be creatable, or the first label
    # an annotator presses is lost.
    parent = os.path.dirname(ANNOTATIONS_FILE) or "."
    try:
        os.makedirs(parent, exist_ok=True)
    except OSError as e:
        problems.append(f"cannot create the label directory {parent}: {e}")

    already = load_existing_annotations(ANNOTATIONS_FILE)
    print(f"windows available : {len(on_disk)} in {UNLABELED_DIR}")
    print(f"clip videos       : {VIDEO_DIR}")
    print(f"writing labels to : {ANNOTATIONS_FILE}")
    if wanted is not None:
        print(f"queue restricted  : {len(wanted)} ids from {worklist_file}")
    print(f"already judged    : {len(already & set(queue_ids))} of {len(queue_ids)}")
    print(f"left to label     : {len(set(queue_ids) - already)}")

    # De-duplicate while keeping order; a missing clip repeats once per window.
    problems = list(dict.fromkeys(problems))
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
    else:
        print("\nAll good -- you are ready to label.")
    return not problems, problems


# ==========================================
# 4. ANNOTATION APP LOGIC
# ==========================================
def run_annotator(unlabeled_dir=None, video_dir=None, annotations_file=None, worklist_file=None):
    UNLABELED_DIR, VIDEO_DIR, ANNOTATIONS_FILE = resolve_paths(unlabeled_dir, video_dir, annotations_file)

    pt_files = sorted(glob.glob(os.path.join(UNLABELED_DIR, "*_windows.pt")))

    if not pt_files:
        print(f"No aggregated window files found in {UNLABELED_DIR}")
        return

    existing_labels = load_existing_annotations(ANNOTATIONS_FILE)
    worklist_ids = load_worklist_ids(worklist_file)

    # Flatten all windows from all videos into a single queue
    annotation_queue = []
    for pt_path in pt_files:
        windows = torch.load(pt_path, weights_only=False)
        for w in windows:
            # Create a unique ID for this specific clip-span snippet
            window_id = common.make_window_id(w)
            if window_id not in existing_labels:
                if worklist_ids is not None and window_id not in worklist_ids:
                    continue
                w["window_id"] = window_id
                annotation_queue.append(w)

    total_pending = len(annotation_queue)
    if total_pending == 0:
        print("All extracted windows have already been annotated! You are ready for Phase 4.")
        return

    print(f"Found {total_pending} windows waiting for annotation.\n")
    # Echoed because writing to the wrong ledger is the one mistake that ruins the
    # calibration round: three annotators sharing one CSV overwrite each other and
    # kappa cannot be computed. Seeing the path before the timer runs out is the
    # last chance to catch it.
    print(f"Writing labels to: {ANNOTATIONS_FILE}")
    if worklist_ids is not None:
        print(f"Queue restricted to {len(worklist_ids)} ids from: {worklist_file}")
    print("\nCONTROLS:")
    for key, name in CLASS_MAP.items():
        print(f"  Press '{chr(key)}' -> {name}")
    print("  Press 'SPACE' -> Toggle Skeleton ON/OFF")
    print("  Press 'D'     -> Skip (recorded as 'Skip' for adjudication)")
    print("\nStarting in 3 seconds...")

    cv2.namedWindow("Annotator", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Annotator", 1280, 720)
    cv2.waitKey(3000)

    show_skeleton = True

    for idx, item in enumerate(annotation_queue):
        video_path = os.path.join(VIDEO_DIR, f"{item['video_name']}.mp4")
        if not os.path.exists(video_path):
            print(f"Missing video: {video_path}")
            continue

        # Extract tensors and shapes
        keypoints = item["keypoints"].numpy() # Shape: (3, Time, Nodes)
        bboxes = item["bboxes"] # List of [cx, cy, w, h]
        num_frames = keypoints.shape[1]
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, item["start_frame"])
        
        raw_drawn = []
        raw_clean = []
        
        # 1. Cache the window's frames into memory for looping (num_frames is the
        #    clip's native count, so keypoint index i is video frame start_frame + i)
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret: break
                
            clean_frame = frame.copy()
            kps_frame = keypoints[:, i, :] # Shape: (3, 17)
            
            # Draw the Bounding Box to isolate the target student
            cx, cy, w, h = bboxes[i]
            x1, y1 = int(cx - w/2), int(cy - h/2)
            x2, y2 = int(cx + w/2), int(cy + h/2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2) # Orange Box
            cv2.putText(frame, "TARGET", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            # Draw the 17 Keypoints
            for node in range(17):
                x, y, conf = int(kps_frame[0, node]), int(kps_frame[1, node]), kps_frame[2, node]
                if conf > 0.3: 
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1) 
            
            # Draw updated 2-Class UI elements on both views
            ui_text1 = f"Clip: {idx + 1} / {total_pending} | ID: {item['track_id']}"
            ui_text2 = "1 = Focused | 2 = Distracted | 8 = Trash | 9 = Quit"
            
            for f_target in [frame, clean_frame]:
                cv2.putText(f_target, ui_text1, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(f_target, ui_text2, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            raw_drawn.append(frame)
            raw_clean.append(clean_frame)
            
        cap.release()

        if not raw_drawn:
            continue

        # 2. Infinite Playback Loop (UX Improvement)
        action_flag = None
        while action_flag is None:
            for i in range(len(raw_drawn)):
                display_frame = raw_drawn[i] if show_skeleton else raw_clean[i]
                cv2.imshow("Annotator", display_frame)

                action = resolve_key(cv2.waitKey(33) & 0xFF)

                if action is None:
                    continue
                if action == "toggle":
                    show_skeleton = not show_skeleton
                elif action == "skip":
                    # Recorded, not dropped: Phase C adjudicates every Skip.
                    save_annotation(item["window_id"], SKIP_LABEL, ANNOTATIONS_FILE)
                    print(f"⏭️  Skipped [{SKIP_LABEL}] for {item['window_id']} "
                          f"-- goes to adjudication")
                    action_flag = "skip"
                    break
                elif action == "QUIT":
                    action_flag = "quit"
                    break
                else:
                    # INSTANT DISK SAVE
                    save_annotation(item["window_id"], action, ANNOTATIONS_FILE)
                    print(f"✅ Saved [{action}] for {item['window_id']}")
                    action_flag = "labeled"
                    break

            if action_flag is not None:
                break

        if action_flag == "quit":
            print("\nSession Terminated. Progress has been saved locally.")
            break

    cv2.destroyAllWindows()
    print("\nAnnotation phase completed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-dir", default=None)
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--worklist", default=None)
    parser.add_argument("--check", action="store_true",
                        help="verify the setup (data present, paths valid, ledger "
                             "writable) and exit without opening the GUI")
    args = parser.parse_args()
    if args.check:
        ok, _ = preflight(args.windows_dir, args.video_dir, args.annotations,
                          args.worklist)
        raise SystemExit(0 if ok else 1)
    run_annotator(args.windows_dir, args.video_dir, args.annotations, args.worklist)
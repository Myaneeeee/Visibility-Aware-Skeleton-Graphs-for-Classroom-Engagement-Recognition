import os
import glob
import cv2
import json
import torch
import contextlib
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

import _bootstrap  # noqa: F401  -- puts ../lib on sys.path
from paths import SAV_VIDEOS_DIR, SAV_POSE_TRACKS_DIR

# ==========================================
# 1. GLOBAL CONFIGURATION
# ==========================================
ENVIRONMENT = "local" # Change to "cloud" when moving to heavier hardware

if ENVIRONMENT == "local":
    # --- Local Environment Settings ---
    YOLO_MODEL_NAME = "yolo11n-pose.pt"
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    INPUT_VIDEO_DIR = SAV_VIDEOS_DIR
    OUTPUT_TRACKS_DIR = SAV_POSE_TRACKS_DIR
    CONFIDENCE_THRESHOLD = 0.5

elif ENVIRONMENT == "cloud":
    # --- Cloud Environment Settings ---
    YOLO_MODEL_NAME = "yolo11x-pose.pt"
    DEVICE = "cuda:0"
    INPUT_VIDEO_DIR = SAV_VIDEOS_DIR
    OUTPUT_TRACKS_DIR = SAV_POSE_TRACKS_DIR
    CONFIDENCE_THRESHOLD = 0.65

# ==========================================
# 1.5 PARAMETRIZATION HELPER
# ==========================================
def resolve_dirs(input_dir=None, output_dir=None):
    """Return (input_video_dir, output_tracks_dir); CLI args override defaults."""
    inp = input_dir if input_dir else INPUT_VIDEO_DIR
    out = output_dir if output_dir else OUTPUT_TRACKS_DIR
    return inp, out

print(f"\n=== ENVIRONMENT: {ENVIRONMENT.upper()} ===")
print(f"Using Device: {DEVICE}")
print(f"Looking for videos in: {INPUT_VIDEO_DIR}")

# ==========================================
# 1.6 CRASH-SAFE TIMELINE WRITES
# ==========================================
PARTIAL_SUFFIX = ".partial"

@contextlib.contextmanager
def atomic_jsonl(final_path):
    """Write to a sibling .partial file, renamed into place only on clean exit.

    The skip logic below tests for *existence*, so a timeline left half-written by
    an interrupted run would be silently accepted as finished -- yielding a window
    with the wrong frame count and wrong coverage. Writing under a name the skip
    logic ignores makes that impossible: the final name only ever appears once the
    last frame is on disk.
    """
    partial_path = final_path + PARTIAL_SUFFIX
    try:
        with open(partial_path, "w") as f_out:
            yield f_out
        os.replace(partial_path, final_path)
    except BaseException:
        # BaseException, not Exception: the common case is the operator killing a
        # long run, which raises KeyboardInterrupt.
        if os.path.exists(partial_path):
            os.remove(partial_path)
        raise

def is_already_extracted(final_path):
    """True only for a timeline that was written all the way through."""
    return os.path.exists(final_path)

def order_videos(video_files, order_file):
    """Return ``video_files`` in the clip order listed in ``order_file``.

    The tracker persists across clips, so track ids depend on processing order;
    the order file must name exactly the clips present.
    """
    by_clip = {os.path.basename(p)[:-len(".mp4")]: p for p in video_files}
    with open(order_file) as f:
        order = [line.strip() for line in f if line.strip()]
    missing = [c for c in order if c not in by_clip]
    unlisted = sorted(set(by_clip) - set(order))
    if missing or unlisted:
        raise ValueError(f"{order_file} does not match the videos on disk: "
                         f"{len(missing)} listed but absent, {len(unlisted)} present but unlisted")
    return [by_clip[c] for c in order]

# ==========================================
# 2. POSE & BBOX TRACKING LOGIC
# ==========================================
def extract_pose_from_video(video_path, output_dir, model, conf_thresh):
    """
    Processes uncut video, tracking students, extracting 17 keypoints AND Bounding Boxes.
    Streams output frame-by-frame to a JSONL file to prevent RAM exhaustion and data loss.
    """
    video_name = os.path.basename(video_path).replace(".mp4", "")
    output_jsonl_path = os.path.join(output_dir, f"{video_name}_timeline.jsonl")
    
    # Crash Recovery / Skip Logic: only a fully written timeline carries the final
    # name, so its existence is enough to call this clip done.
    # (If you want to re-run, just delete the file from the folder)
    if is_already_extracted(output_jsonl_path):
        print(f"  -> Skipping {video_name} (Timeline already exists. Delete file to re-run).")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  -> Skipping {video_name} (Cannot open file)")
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    unique_students = set()
    
    # Data goes straight to disk, under a temporary name until the last frame lands.
    with atomic_jsonl(output_jsonl_path) as f_out:
        with tqdm(total=total_frames, desc=f"Extracting {video_name}", leave=False) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Run YOLO tracking with BoT-SORT
                results = model.track(
                    frame, 
                    persist=True, 
                    classes=0, 
                    conf=conf_thresh, 
                    tracker="botsort.yaml", 
                    verbose=False,
                    max_det=100
                )
                
                # Prepare the container for this specific frame
                frame_record = {
                    "frame_idx": frame_idx,
                    "students": {}
                }
                
                # Check for robust detections
                if results[0].boxes is not None and results[0].boxes.id is not None and results[0].keypoints is not None:
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    boxes_data = results[0].boxes.xywh.cpu().numpy() 
                    keypoints_data = results[0].keypoints.data.cpu().numpy()
                    
                    if len(track_ids) == len(boxes_data) == len(keypoints_data):
                        for i, track_id in enumerate(track_ids):
                            str_id = str(track_id)
                            unique_students.add(str_id)
                            
                            # Vectorized rounding
                            person_keypoints = np.round(keypoints_data[i], 3).tolist()
                            person_bbox = np.round(boxes_data[i], 3).tolist()
                            
                            # Attach to this frame's record
                            frame_record["students"][str_id] = {
                                "keypoints": person_keypoints,
                                "bbox": person_bbox
                            }
                
                # Flush the frame record to disk immediately as a JSON string
                f_out.write(json.dumps(frame_record) + "\n")
                
                frame_idx += 1
                pbar.update(1)
            
    cap.release()
    print(f"\n  -> Saved {video_name}: Tracked {len(unique_students)} unique students.")

# ==========================================
# 3. EXECUTE PIPELINE
# ==========================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--order", default=None,
                        help="file of clip ids, one per line, giving the processing order")
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N clips (trial runs)")
    args = parser.parse_args()
    input_dir, output_dir = resolve_dirs(args.input_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    try:
        print(f"\nLoading {YOLO_MODEL_NAME} into VRAM...")
        yolo_model = YOLO(YOLO_MODEL_NAME)
        yolo_model.to(DEVICE)
        video_files = glob.glob(os.path.join(input_dir, "**", "*.mp4"), recursive=True)
        if args.order:
            video_files = order_videos(video_files, args.order)
        if args.limit is not None:
            video_files = video_files[:args.limit]
        print(f"Found {len(video_files)} videos to process.\n")
        for video in video_files:
            extract_pose_from_video(video, output_dir, yolo_model, CONFIDENCE_THRESHOLD)
        print("\n✅ Extraction complete! Timelines saved to:", output_dir)
    except Exception as e:
        print(f"Error during execution: {e}")
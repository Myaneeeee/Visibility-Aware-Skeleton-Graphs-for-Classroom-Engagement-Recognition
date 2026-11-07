"""Step: turn pose-track timelines into one ST-GCN window per person per clip.

Reads ``datasets/sav/pose_tracks/*_timeline.jsonl`` and writes
``datasets/sav/windows/<video>_windows.pt``. Applies linear interpolation
across minor tracking gaps.

A SAV clip is a single 3-second unit, and the source videos run 18-30 fps (modal 25,
measured across all 60), so a clip is 54-90 frames -- most often 75, and only 28% of
videos reach 90. Sliding a fixed 90-frame window across that produced zero windows
for every clip, so each person's track is taken across the clip's actual span
instead, gated on a *fraction* of the clip rather than an absolute frame count.

Windows are stored at the clip's native frame count, not resampled to 90 here, so
that keypoint index i still lines up with video frame i -- annotate.py and
review.py seek the video by ``start_frame`` and overlay skeletons per frame, and
resampling on disk would drift the overlay off the footage. ``common.TARGET_T``
normalization happens in the model-facing loader (``common.load_labeled_data``).
"""
import _bootstrap  # noqa: F401  -- puts ../lib on sys.path

import os
import glob
import json
import numpy as np
import torch

from paths import SAV_POSE_TRACKS_DIR, SAV_WINDOWS_DIR


MIN_COVERAGE = 0.8      # fraction of the clip a person must be tracked in
MAX_CLIP_FRAMES = 300   # ~10s at 30fps; a real clip is 3s, so anything longer is a source video


def preprocess_jsonl_to_windows(jsonl_path, min_coverage=MIN_COVERAGE,
                                max_clip_frames=MAX_CLIP_FRAMES):
    """
    Builds one window per tracked person spanning the clip's own frame range.
    Drops people tracked in less than ``min_coverage`` of the clip, and applies
    linear interpolation across the remaining gaps.
    """
    video_name = os.path.basename(jsonl_path).replace("_timeline.jsonl", "")

    timeline = {}
    max_frame_in_video = -1

    # Reconstruct timeline from JSONL and find the absolute maximum frame
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            f_idx = data["frame_idx"]
            if f_idx > max_frame_in_video:
                max_frame_in_video = f_idx

            for track_id, s_data in data["students"].items():
                if track_id not in timeline:
                    timeline[track_id] = {}
                timeline[track_id][f_idx] = s_data

    n_frames = max_frame_in_video + 1
    if n_frames > max_clip_frames:
        raise ValueError(
            f"{video_name} has {n_frames} frames, over the {max_clip_frames}-frame ceiling "
            f"for a 3-second clip. This step expects one cut clip per timeline -- a full "
            f"source video would collapse into a single enormous window and its video_name "
            f"would not be a clip id, breaking the join to SAV's action annotations."
        )

    all_windows = []

    for track_id, frames_dict in sorted(timeline.items(), key=lambda kv: int(kv[0])):
        present_frames = sorted(frames_dict)

        # Gate on a fraction of the clip: an absolute frame count would reject every
        # person in an 18fps clip, which has fewer frames than one 90-frame window.
        if len(present_frames) < min_coverage * n_frames:
            continue

        full_kps = np.zeros((n_frames, 17, 3), dtype=np.float32)
        full_bboxes = np.zeros((n_frames, 4), dtype=np.float32)
        mask = np.zeros(n_frames, dtype=bool)

        for f_idx in present_frames:
            full_kps[f_idx] = np.array(frames_dict[f_idx]["keypoints"])
            full_bboxes[f_idx] = np.array(frames_dict[f_idx]["bbox"])
            mask[f_idx] = True

        # Interpolate minor gaps
        if not np.all(mask):
            x_range = np.arange(n_frames)
            known_indices = x_range[mask]

            for j in range(17):
                for c in range(3):
                    full_kps[:, j, c] = np.interp(x_range, known_indices, full_kps[known_indices, j, c])
            for b in range(4):
                full_bboxes[:, b] = np.interp(x_range, known_indices, full_bboxes[known_indices, b])

        window_kp_transposed = np.transpose(full_kps, (2, 0, 1))

        all_windows.append({
            "video_name": video_name,
            "track_id": int(track_id),
            "start_frame": 0,
            "end_frame": n_frames - 1,
            "keypoints": torch.tensor(window_kp_transposed, dtype=torch.float32),
            "bboxes": full_bboxes.tolist(),
            # marks this as a clip-span window: the loader stretches it to TARGET_T
            # rather than applying an fps ratio to it
            "n_native_frames": n_frames,
        })

    return all_windows


def process_jsonl_to_tensors(input_tracks_dir=None, output_tensor_dir=None):
    INPUT_TRACKS_DIR = input_tracks_dir or SAV_POSE_TRACKS_DIR
    OUTPUT_TENSOR_DIR = output_tensor_dir or SAV_WINDOWS_DIR
    os.makedirs(OUTPUT_TENSOR_DIR, exist_ok=True)

    jsonl_files = glob.glob(os.path.join(INPUT_TRACKS_DIR, "*.jsonl"))
    print(f"Found {len(jsonl_files)} timeline files.\n")

    total_saved = 0
    for jsonl_path in jsonl_files:
        video_name = os.path.basename(jsonl_path).replace("_timeline.jsonl", "")
        print(f"Processing {video_name}...")

        windows = preprocess_jsonl_to_windows(jsonl_path)

        if not windows:
            print(f"  -> No person tracked through enough of {video_name}.")
            continue

        save_path = os.path.join(OUTPUT_TENSOR_DIR, f"{video_name}_windows.pt")
        torch.save(windows, save_path)
        total_saved += len(windows)

    print(f"\n✅ Windowing complete! Saved {total_saved} total windows aggregated by video to {OUTPUT_TENSOR_DIR}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    process_jsonl_to_tensors(args.input_dir, args.output_dir)

import csv
import numpy as np
import torch
import pytest


def _make_window(video_name, track_id, start_frame):
    # keypoints: (3, 90, 17) = (x, y, conf, time, node)
    rng = np.random.default_rng(track_id + start_frame)
    kps = rng.uniform(0, 500, size=(3, 90, 17)).astype(np.float32)
    kps[2] = rng.uniform(0.3, 0.95, size=(90, 17)).astype(np.float32)  # conf channel
    bboxes = rng.uniform(50, 400, size=(90, 4)).astype(np.float32).tolist()
    return {
        "video_name": video_name,
        "track_id": int(track_id),
        "start_frame": int(start_frame),
        "end_frame": int(start_frame + 89),
        "keypoints": torch.from_numpy(kps),
        "bboxes": bboxes,
    }


@pytest.fixture
def synthetic_dataset(tmp_path):
    pt_dir = tmp_path / "unlabeled_windows"
    pt_dir.mkdir()

    # 3 clip files, each with a few tracks/windows. Names follow the real SAV shape
    # <date>_<room>_<clipindex>, because that is what extract_poses writes: each 3s
    # clip is its own mp4, so video_name is a CLIP id, not a recording. The first two
    # clips are the SAME room recorded on the same day (adjacent indices, so they
    # physically overlap by 1s); the third is a different room. Fixtures with invented
    # names like "vidA" cannot exercise room-level fold separation at all.
    plan = {"20230118_34_1": [(1, 0), (1, 45), (2, 0)],
            "20230118_34_2": [(5, 0), (5, 45)],
            "20230202_21_1": [(9, 0), (9, 45), (10, 0)]}

    labels = {}  # window_id -> label
    for video, specs in plan.items():
        windows = []
        for track_id, start in specs:
            w = _make_window(video, track_id, start)
            windows.append(w)
            wid = f"{video}_id_{track_id}_f{start}_f{start + 89}"
            # deterministic label assignment: alternate, leave one unlabeled + one trash
            labels[wid] = "Focused" if (track_id + start) % 2 == 0 else "Distracted"
        torch.save(windows, pt_dir / f"{video}_windows.pt")

    # Force one TRASH and one None for realism
    all_ids = list(labels.keys())
    labels[all_ids[0]] = "TRASH"
    unlabeled_id = all_ids[1]
    del labels[unlabeled_id]

    ann = tmp_path / "annotations.csv"
    with open(ann, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["window_id", "label"])
        for wid, lab in labels.items():
            wr.writerow([wid, lab])

    usable = sum(1 for v in labels.values() if v in ("Focused", "Distracted"))
    return {
        "pt_dir": str(pt_dir),
        "annotations_csv": str(ann),
        "expected": {"usable": usable, "videos": 3},
    }

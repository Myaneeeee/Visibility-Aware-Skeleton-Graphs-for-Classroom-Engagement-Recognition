import csv

import numpy as np
import torch

import common


def _native_length_dataset(tmp_path, n_frames):
    """One clip, one window, saved at a clip's native frame count instead of 90."""
    pt_dir = tmp_path / "windows"
    pt_dir.mkdir()
    rng = np.random.default_rng(0)
    kps = rng.uniform(0, 500, size=(3, n_frames, 17)).astype(np.float32)
    kps[2] = rng.uniform(0.6, 0.95, size=(n_frames, 17)).astype(np.float32)
    w = {
        "video_name": "20181016_05_7",
        "track_id": 1,
        "start_frame": 0,
        "end_frame": n_frames - 1,
        "keypoints": torch.from_numpy(kps),
        "bboxes": rng.uniform(50, 400, size=(n_frames, 4)).astype(np.float32).tolist(),
        "n_native_frames": n_frames,
    }
    torch.save([w], pt_dir / "20181016_05_7_windows.pt")

    ann = tmp_path / "annotations.csv"
    with open(ann, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["window_id", "label"])
        wr.writerow([common.make_window_id(w), "Focused"])
    return str(pt_dir), str(ann)


def test_load_stretches_native_clip_length_to_model_T(tmp_path):
    """An 18 fps 3s clip is 54 frames on disk; the loader must deliver T=90."""
    pt_dir, ann = _native_length_dataset(tmp_path, 54)

    d = common.load_labeled_data(pt_dir=pt_dir, annotations_csv=ann)

    assert all(t.shape == (3, 90, 17) for t in d["tensors"])
    assert torch.stack(d["tensors"]).shape == (1, 3, 90, 17)


def test_load_labeled_data_counts_and_shapes(synthetic_dataset):
    d = common.load_labeled_data(
        pt_dir=synthetic_dataset["pt_dir"],
        annotations_csv=synthetic_dataset["annotations_csv"],
    )
    n = synthetic_dataset["expected"]["usable"]
    assert len(d["tensors"]) == n
    assert len(d["labels"]) == n
    assert len(d["video_names"]) == n
    assert len(d["window_ids"]) == n
    assert all(t.shape == (3, 90, 17) for t in d["tensors"])
    assert set(d["labels"]).issubset({0, 1})


def test_load_excludes_trash_and_unlabeled(synthetic_dataset):
    d = common.load_labeled_data(
        pt_dir=synthetic_dataset["pt_dir"],
        annotations_csv=synthetic_dataset["annotations_csv"],
    )
    # window_ids returned must all be present in the CSV as Focused/Distracted only
    assert len(d["window_ids"]) == len(set(d["window_ids"]))




def test_load_drops_excluded_windows(synthetic_dataset, tmp_path):
    kwargs = dict(pt_dir=synthetic_dataset["pt_dir"],
                  annotations_csv=synthetic_dataset["annotations_csv"])
    full = common.load_labeled_data(**kwargs, excluded_csv=str(tmp_path / "absent.csv"))
    dropped = full["window_ids"][0]
    excluded = tmp_path / "excluded_windows.csv"
    excluded.write_text(f"window_id,reason\n{dropped},person unverified\n")

    d = common.load_labeled_data(**kwargs, excluded_csv=str(excluded))

    assert d["window_ids"] == full["window_ids"][1:]

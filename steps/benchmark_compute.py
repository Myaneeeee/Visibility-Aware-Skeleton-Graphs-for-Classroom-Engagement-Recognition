"""Compute cost of the MLP, the visible-joint ST-GCN (E1) and their ensemble.

Trains each model once on fold 0 with the sweep's own settings (40 epochs,
batch 16, E1 warm-started and augmented), then times prediction on all
labeled windows on GPU and CPU. Reports parameters, FLOPs per window,
training time, inference time and peak GPU memory.

Writes outputs/diagnostics/compute_benchmark.json.

Usage:  .venv/bin/python -u steps/benchmark_compute.py [--repeats 5]
"""
import argparse
import json
import os
import platform
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

import _bootstrap  # noqa: F401  -- puts ../lib on sys.path
import common
import cv_utils
import evaluate
import paths
import representations
from augment import SkeletonAugment, DEFAULT_AUGMENT, flip_perm_for

SEED = 51
STGCN_CONFIG = "E1"
BATCH = 256
LATENCY_WINDOWS = 100


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def timed(fn, device, repeats):
    """Median wall time of fn() over `repeats` runs, after one warm-up run."""
    fn()
    sync(device)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def cpu_name():
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def flops_per_window(model, x1):
    model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x1)
    return fc.get_total_flops()


def predict(model, X, device):
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            F.softmax(model(X[i:i + BATCH].to(device)), dim=1).cpu()


def predict_mc(model, X, device):
    for i in range(0, len(X), BATCH):
        evaluate.mc_dropout_probs(model, X[i:i + BATCH], device)


def latency_ms(model, X, device, mc=False):
    """Median time for one window, batch size 1."""
    model.eval()
    xs = [X[i:i + 1] for i in range(LATENCY_WINDOWS)]

    def one(x):
        if mc:
            evaluate.mc_dropout_probs(model, x, device)
        else:
            with torch.no_grad():
                F.softmax(model(x.to(device)), dim=1).cpu()
    for x in xs[:10]:
        one(x)
    times = []
    for x in xs:
        sync(device)
        t0 = time.perf_counter()
        one(x)
        sync(device)
        times.append(time.perf_counter() - t0)
    return 1000 * statistics.median(times)


def train_timed(kind, X, y, device, epochs, augment=None, **model_kw):
    evaluate.set_determinism(SEED)
    model = evaluate._make_model(kind, len(common.CLASS_NAMES), **model_kw)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    sync(device)
    t0 = time.perf_counter()
    evaluate._train_fold(model, X, y, epochs, device, augment=augment)
    sync(device)
    seconds = time.perf_counter() - t0
    peak = (torch.cuda.max_memory_allocated() / 2**20
            if device.startswith("cuda") else None)
    return model.eval(), seconds, peak


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--cpu-epochs", type=int, default=1,
                    help="epochs timed for ST-GCN training on CPU, scaled to 40")
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU: the sweep trained on one")
    gpu = "cuda:0"

    data = common.load_labeled_data(pt_dir=paths.SAV_WINDOWS_FILTERED_DIR,
                                    annotations_csv=paths.ANNOTATIONS_FILE)
    labels = torch.tensor(data["labels"], dtype=torch.long)
    windows = torch.stack(data["tensors"])
    joints = evaluate.joints_for(STGCN_CONFIG)
    X_graph = windows[:, :, :, joints]
    n = len(labels)

    t0 = time.perf_counter()
    feats = np.stack([representations.features("marg85", t) for t in data["tensors"]])
    feature_seconds = time.perf_counter() - t0
    X_mlp = torch.tensor(feats, dtype=torch.float32)

    train_idx, _ = cv_utils.source_video_folds(data["video_names"], k=5)[0]
    n_train = len(train_idx)
    print(f"{n} windows | fold 0 trains on {n_train} | seed {SEED}")
    print(f"GPU {torch.cuda.get_device_name(0)} | CPU {cpu_name()}"
          f" | {torch.get_num_threads()} threads | torch {torch.__version__}")

    stgcn_kw = dict(backbone_path=evaluate.backbone_path_for(STGCN_CONFIG),
                    joints=joints, weights=evaluate.weights_for(STGCN_CONFIG),
                    readout=evaluate.readout_for(STGCN_CONFIG),
                    observed=evaluate.observed_for(STGCN_CONFIG),
                    joint_bn=evaluate.joint_bn_for(STGCN_CONFIG))

    def stgcn_augment():
        return SkeletonAugment(seed=SEED * 1000, flip_perm=flip_perm_for(joints),
                               **DEFAULT_AUGMENT)

    res = {"windows": n, "fold0_train_windows": n_train, "seed": SEED,
           "stgcn_config": STGCN_CONFIG, "mc_passes": evaluate.MC_PASSES,
           "batch": BATCH, "repeats": args.repeats,
           "gpu": torch.cuda.get_device_name(0), "cpu": cpu_name(),
           "cpu_threads": torch.get_num_threads(),
           "torch": torch.__version__,
           "mlp_feature_extraction_s": feature_seconds}

    print("\n[1] training, fold 0, 40 epochs")
    mlp_gpu, s, peak = train_timed("mlp", X_mlp[train_idx], labels[train_idx], gpu, 40,
                                   in_dim=X_mlp.shape[1])
    res["mlp_train_gpu_s"], res["mlp_train_gpu_peak_mib"] = s, peak
    print(f"  MLP     GPU {s:8.1f} s   peak {peak:.0f} MiB")
    _, s, _ = train_timed("mlp", X_mlp[train_idx], labels[train_idx], "cpu", 40,
                          in_dim=X_mlp.shape[1])
    res["mlp_train_cpu_s"] = s
    print(f"  MLP     CPU {s:8.1f} s")
    stgcn_gpu, s, peak = train_timed("stgcn", X_graph[train_idx], labels[train_idx], gpu,
                                     40, augment=stgcn_augment(), **stgcn_kw)
    res["stgcn_train_gpu_s"], res["stgcn_train_gpu_peak_mib"] = s, peak
    print(f"  ST-GCN  GPU {s:8.1f} s   peak {peak:.0f} MiB")
    _, s, _ = train_timed("stgcn", X_graph[train_idx], labels[train_idx], "cpu",
                          args.cpu_epochs, augment=stgcn_augment(), **stgcn_kw)
    res["stgcn_train_cpu_s_est40"] = s * 40 / args.cpu_epochs
    res["stgcn_train_cpu_epochs_timed"] = args.cpu_epochs
    print(f"  ST-GCN  CPU {s * 40 / args.cpu_epochs:8.1f} s  "
          f"(timed {args.cpu_epochs} epoch(s), scaled to 40)")

    mlp_cpu = mlp_gpu.to("cpu")
    stgcn_cpu = stgcn_gpu.to("cpu")
    res["mlp_params"] = n_params(mlp_cpu)
    res["stgcn_params"] = n_params(stgcn_cpu)
    res["mlp_flops"] = flops_per_window(mlp_cpu, X_mlp[:1])
    res["stgcn_flops"] = flops_per_window(stgcn_cpu, X_graph[:1])
    print(f"\n[2] size   MLP {res['mlp_params']:,} params, {res['mlp_flops']:,} FLOPs/window"
          f" | ST-GCN {res['stgcn_params']:,} params, {res['stgcn_flops']:,} FLOPs/window")

    print(f"\n[3] inference on all {n} windows (median of {args.repeats}), "
          f"and batch-1 latency")
    for dev in (gpu, "cpu"):
        tag = "gpu" if dev == gpu else "cpu"
        mlp = mlp_cpu.to(dev)
        stgcn = stgcn_cpu.to(dev)
        if dev == gpu:
            torch.cuda.reset_peak_memory_stats()
        rows = {
            "mlp": timed(lambda: predict(mlp, X_mlp, dev), dev, args.repeats),
            "stgcn": timed(lambda: predict(stgcn, X_graph, dev), dev, args.repeats),
            "stgcn_mc": timed(lambda: predict_mc(stgcn, X_graph, dev), dev, args.repeats),
        }
        if dev == gpu:
            res["infer_gpu_peak_mib"] = torch.cuda.max_memory_allocated() / 2**20
        lat = {
            "mlp": latency_ms(mlp, X_mlp, dev),
            "stgcn": latency_ms(stgcn, X_graph, dev),
            "stgcn_mc": latency_ms(stgcn, X_graph, dev, mc=True),
        }
        for k in rows:
            res[f"infer_{tag}_{k}_s"] = rows[k]
            res[f"latency_{tag}_{k}_ms"] = lat[k]
            print(f"  {tag.upper()} {k:9s} all {rows[k]:8.3f} s   one window {lat[k]:8.2f} ms")
        mlp_cpu, stgcn_cpu = mlp.to("cpu"), stgcn.to("cpu")

    feat_ms = 1000 * feature_seconds / n
    print(f"\n[4] summary (MLP features: {feature_seconds:.2f} s for all, "
          f"{feat_ms:.2f} ms/window, CPU, counted once per arm)")
    arms = {
        "MLP": dict(params=res["mlp_params"], flops=res["mlp_flops"],
                    train_gpu=res["mlp_train_gpu_s"], train_cpu=res["mlp_train_cpu_s"],
                    keys=[("mlp", 1)]),
        "MLP x2 (control)": dict(params=2 * res["mlp_params"], flops=2 * res["mlp_flops"],
                                 train_gpu=2 * res["mlp_train_gpu_s"],
                                 train_cpu=2 * res["mlp_train_cpu_s"],
                                 keys=[("mlp", 2)]),
        "ST-GCN-13": dict(params=res["stgcn_params"], flops=res["stgcn_flops"],
                          train_gpu=res["stgcn_train_gpu_s"],
                          train_cpu=res["stgcn_train_cpu_s_est40"],
                          keys=[("stgcn", 1)]),
        "ST-GCN-13 + MC": dict(params=res["stgcn_params"],
                               flops=evaluate.MC_PASSES * res["stgcn_flops"],
                               train_gpu=res["stgcn_train_gpu_s"],
                               train_cpu=res["stgcn_train_cpu_s_est40"],
                               keys=[("stgcn_mc", 1)]),
        "MLP + ST-GCN-13 + MC": dict(
            params=res["mlp_params"] + res["stgcn_params"],
            flops=res["mlp_flops"] + evaluate.MC_PASSES * res["stgcn_flops"],
            train_gpu=res["mlp_train_gpu_s"] + res["stgcn_train_gpu_s"],
            train_cpu=res["mlp_train_cpu_s"] + res["stgcn_train_cpu_s_est40"],
            keys=[("mlp", 1), ("stgcn_mc", 1)]),
    }
    summary = {}
    hdr = (f"| arm | params | FLOPs/window | train/fold GPU (s) | train/fold CPU (s) "
           f"| infer all GPU (s) | infer all CPU (s) | 1 window GPU (ms) | 1 window CPU (ms) |")
    print(hdr)
    print("|" + "---|" * 9)
    for name, a in arms.items():
        uses_mlp = any(k == "mlp" for k, _ in a["keys"])
        extra_s = feature_seconds if uses_mlp else 0.0
        extra_ms = feat_ms if uses_mlp else 0.0
        row = dict(params=a["params"], flops=a["flops"], train_gpu_s=a["train_gpu"],
                   train_cpu_s=a["train_cpu"])
        for tag in ("gpu", "cpu"):
            row[f"infer_{tag}_s"] = extra_s + sum(
                m * res[f"infer_{tag}_{k}_s"] for k, m in a["keys"])
            row[f"latency_{tag}_ms"] = extra_ms + sum(
                m * res[f"latency_{tag}_{k}_ms"] for k, m in a["keys"])
        summary[name] = row
        print(f"| {name} | {row['params']:,} | {row['flops'] / 1e6:,.2f} M "
              f"| {row['train_gpu_s']:.1f} | {row['train_cpu_s']:.1f} "
              f"| {row['infer_gpu_s']:.3f} | {row['infer_cpu_s']:.3f} "
              f"| {row['latency_gpu_ms']:.2f} | {row['latency_cpu_ms']:.2f} |")
    res["summary"] = summary

    out = os.path.join(paths.diagnostics_dir(), "compute_benchmark.json")
    os.makedirs(paths.diagnostics_dir(), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

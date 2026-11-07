import argparse
import json
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import _bootstrap  # noqa: F401  -- puts ../lib on sys.path
import common
import cv_utils
import metrics as metrics_mod
import baselines as bl
import transforms
import transfer
import paths
import representations
import losses
from model import (STGCNClassifier, OBSERVED_JOINTS_13,
                   COCO_JOINT_CONFIDENCE, bone_parents)
from paths import NTU_BACKBONE_FILE
from augment import (AugmentedTensorDataset, SkeletonAugment, DEFAULT_AUGMENT,
                     flip_perm_for)

SEED = 42
MC_PASSES = 20

# Appended to the run label so a weighted run can never overwrite the
# pre-registered unweighted one it is meant to be compared against. "ce" maps to
# the empty string, which keeps every existing A_seed51-style path byte-identical.
LOSS_SUFFIX = {"ce": "", "weighted": "+w", "focal": "+f"}

# The ablation ladder, PREREGISTRATION.md Sec. 5b.3. Each rung adds exactly one thing to
# the rung below it, so a difference in the table is attributable to that one thing.
# A, B and C are training runs. D is not: it is C's saved run plus post-hoc
# calibration, so its classifier is bit-identical to C's by construction and
# training it again would burn a full GPU run to reproduce C under another name.
CONFIGS = {
    "A": {"pretrained": False, "augment": False, "desc": "from-scratch ST-GCN"},
    "B": {"pretrained": True, "augment": False, "desc": "A + NTU-60 warm start"},
    "C": {"pretrained": True, "augment": True, "desc": "B + augmentation"},
    # The observability ladder. E1 is C with the graph and the node set rebuilt
    # over the joints the detector actually sees; knees and ankles sit at 0.29
    # and 0.19 mean confidence because students are behind desks.
    "E1": {"pretrained": True, "augment": True, "joints": OBSERVED_JOINTS_13,
           "desc": "C on the 13 observed joints (ST-GCN-13)"},
    # E2 makes the same prior continuous instead of binary: every joint stays,
    # but each edge is scaled by how well its endpoints are actually seen.
    "E2": {"pretrained": True, "augment": True, "weights": COCO_JOINT_CONFIDENCE,
           "desc": "C with an observability-weighted adjacency"},
    # E3 is E1 with the input representation swapped for bone vectors, which
    # Sec. 3.3 measured as the strongest of twelve feature families. It replaces
    # the coordinates rather than extending them: a wider input would change
    # in_channels and break the NTU warm start, i.e. two things at once. The
    # joint+bone two-stream model is recovered by averaging E1 and E3.
    "E3": {"pretrained": True, "augment": True, "joints": OBSERVED_JOINTS_13,
           "stream": "bone",
           "desc": "E1 on bone vectors instead of joint coordinates"},
    # E5 is C with the classifier's pool restricted to the 13 observed joints.
    # Graph, node set and input stay C's, so E1 - E5 is the topology's share.
    "E5": {"pretrained": True, "augment": True, "readout": OBSERVED_JOINTS_13,
           "desc": "C with the readout pooled over the 13 observed joints"},
    # E6 is C with the 4 unobserved joints blocked at the input: out of data_bn's
    # statistics, then zeroed. The 17-node graph and readout stay C's, so E6 - C
    # is the share of E1's gain that is input contamination, not topology.
    "E6": {"pretrained": True, "augment": True, "observed": OBSERVED_JOINTS_13,
           "desc": "C with the 4 unobserved joints blocked at the input"},
    # E7 is C with data_bn built as the reference ST-GCN builds it: one set of
    # statistics per (joint, channel) instead of one per channel pooled over
    # all joints. Nothing is masked, so E7 - C separates the pooled statistics
    # from the unobserved joints' values, which E6 changed together.
    "E7": {"pretrained": True, "augment": True, "joint_bn": True,
           "desc": "C with per-joint input normalization"},
}

# Kept in the CLI's --config choices on purpose: `--config D` is a reasonable
# thing to try, and refusing it with directions beats argparse's generic
# "invalid choice" or, worse, silently filing C's numbers under D's name.
POST_HOC_CONFIGS = {
    "D": ("C + calibrated confidence. Produced post-hoc from C's saved run by "
          "steps/analyze_confidence.py, which retrains nothing"),
}

MODEL_KINDS = ["stgcn", "gru", "mlp"]


def resolve_config(name):
    """Look up a *trainable* ablation config by letter.

    Raises for D (which is post-hoc, not a training run) and for unknown names.
    """
    if name in CONFIGS:
        return CONFIGS[name]
    if name in POST_HOC_CONFIGS:
        raise ValueError(
            f"config {name} is not a training run: {POST_HOC_CONFIGS[name]}.\n"
            f"  Run it in two steps instead:\n"
            f"    steps/evaluate.py --config C --seed <n>\n"
            f"    steps/analyze_confidence.py outputs/runs/C_seed<n>/stgcn/oof_probs.pt")
    raise ValueError(
        f"unknown config {name!r}; expected one of {sorted(CONFIGS) + sorted(POST_HOC_CONFIGS)}")


def run_label(config, seed, loss="ce"):
    """Identifies one (config, seed, loss) run, keeping outputs apart on disk.

    The loss is part of the identity because class weighting is an off-ladder
    arm (PREREGISTRATION.md Sec. 5b.8), not a replacement for the rung it is
    compared with: C+w must sit beside C, not on top of it.
    """
    if loss not in LOSS_SUFFIX:
        raise ValueError(
            f"unknown loss {loss!r}; expected one of {sorted(LOSS_SUFFIX)}")
    return f"{config}{LOSS_SUFFIX[loss]}_seed{seed}"


def set_determinism(seed):
    """Seed every RNG and pin the deterministic kernels.

    CUBLAS_WORKSPACE_CONFIG must be set before CUDA initializes, so call this before
    touching the GPU. warn_only=True because a few ops have no deterministic CUDA
    implementation: those warn instead of aborting a multi-hour run, and the warning
    names the op so the residual nondeterminism is documented rather than silent.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Cross-video CV for one rung of the ablation ladder.")
    p.add_argument("--config", default="A",
                   choices=sorted(CONFIGS) + sorted(POST_HOC_CONFIGS),
                   help="ablation config to train (default: A, from-scratch). "
                        "D is post-hoc over C -- see steps/analyze_confidence.py")
    p.add_argument("--seed", type=int, default=SEED,
                   help=f"RNG seed; also names the output files (default: {SEED})")
    p.add_argument("--out-prefix", default=None,
                   help="override the output filename prefix (default: <config>_seed<seed>)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--models", nargs="+", default=list(MODEL_KINDS),
                   choices=list(MODEL_KINDS),
                   help="which models to run (default: all three). The E rungs "
                        "only move the ST-GCN, so --models stgcn reruns nothing "
                        "the matching C seed already has")
    p.add_argument("--loss", default="ce", choices=list(losses.LOSS_CHOICES),
                   help="training objective (default: ce, the pre-registered "
                        "one for configs A/B/C). 'weighted' and 'focal' are the "
                        "off-ladder imbalance arm; they write to <config>+w / "
                        "<config>+f run labels so they cannot overwrite the "
                        "reported unweighted runs")
    return p


def backbone_path_for(config):
    """The checkpoint a config warm-starts from, or None to train from scratch."""
    return NTU_BACKBONE_FILE if resolve_config(config)["pretrained"] else None


def augment_for(config):
    """Whether a config augments its training folds."""
    return resolve_config(config)["augment"]


def joints_for(config):
    """The COCO-17 joints a config's graph runs on, or None for all of them."""
    return resolve_config(config).get("joints")


def weights_for(config):
    """Per-joint observability weights on the config's edges, or None."""
    return resolve_config(config).get("weights")


def stream_for(config):
    """The ST-GCN's input representation: joint coordinates or bone vectors."""
    return resolve_config(config).get("stream", "joint")


def readout_for(config):
    """The node indices the ST-GCN's readout pools over, or None for all."""
    return resolve_config(config).get("readout")


def observed_for(config):
    """The node indices that reach the ST-GCN's input, or None for all."""
    return resolve_config(config).get("observed")


def joint_bn_for(config):
    """Whether the ST-GCN's data_bn keeps separate statistics per joint."""
    return resolve_config(config).get("joint_bn", False)


def _make_model(kind, num_classes, backbone_path=None, joints=None, weights=None,
                readout=None, observed=None, joint_bn=False,
                in_dim=bl.POSE_STAT_DIM):
    """Build one model. `joints`, `weights`, `readout`, `observed` and `joint_bn`
    shape the ST-GCN; the baselines are the comparison rather than the treatment
    and always see the full skeleton. `in_dim` is the MLP's feature width."""
    if kind == "stgcn":
        model = STGCNClassifier(in_channels=3, hidden_dim=128,
                                num_classes=num_classes, joints=joints,
                                weights=weights, readout=readout,
                                observed=observed, joint_bn=joint_bn)
        if backbone_path:
            # Backbone only: the NTU head predicts 60 action classes, ours predicts 2,
            # so model.fc keeps its fresh init. transfer asserts on any mismatch.
            transfer.load_pretrained_backbone(model, backbone_path)
        return model
    if backbone_path:
        raise ValueError(
            f"no pretrained backbone exists for model_kind={kind!r}; "
            "the NTU checkpoint is ST-GCN-shaped")
    if kind == "gru":
        return bl.KeypointGRU(num_classes=num_classes)
    if kind == "mlp":
        return bl.PoseStatsMLP(in_dim=in_dim, num_classes=num_classes)
    raise ValueError(kind)


def has_dropout(model):
    """Whether MC-dropout can produce anything but a copy of the deterministic pass."""
    return any(isinstance(m, torch.nn.Dropout) for m in model.modules())


def enable_mc_dropout(model):
    """Put ONLY the dropout layers into training mode.

    model.train() would also switch BatchNorm to batch statistics and mutate its
    running stats, corrupting both the predictions and the model. That is the
    classic MC-dropout bug and it fails silently.
    """
    if not has_dropout(model):
        raise ValueError(
            f"{type(model).__name__} has no nn.Dropout modules, so MC-dropout "
            "would return the deterministic prediction 20 times over and read as "
            "'uncertainty sampling changed nothing'. Check has_dropout() first.")
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def mc_dropout_probs(model, X, device, passes=MC_PASSES):
    """Average `passes` stochastic forward passes. Returns (N, num_classes).

    Unlike temperature scaling, this can change which class wins, which is what
    makes config D a genuinely different predictor rather than only a relabeled
    confidence.

    The dropout masks are drawn from a private generator-backed RNG state rather
    than the global one. Dropout has no `generator=` argument, so the global
    state is saved and restored around the passes instead. Without that, D's
    sampling would advance the same stream that seeds fold i+1's weight init and
    DataLoader shuffle -- so how many MC passes D takes would change the
    classifier A/B/C train, and the ladder's one-thing-per-rung logic would break.
    """
    cpu_state = torch.get_rng_state()
    cuda_states = (torch.cuda.get_rng_state_all()
                   if torch.cuda.is_available() else None)
    try:
        enable_mc_dropout(model)
        total = None
        with torch.no_grad():
            for _ in range(passes):
                prob = F.softmax(model(X.to(device)), dim=1)
                total = prob if total is None else total + prob
        model.eval()
        return (total / passes).cpu().numpy()
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _train_fold(model, X, y, epochs, device, batch_size=16, augment=None,
                loss_history=None, loss_fn=None, transform=None):
    """Train one fold. Appends this fold's per-epoch mean loss to loss_history.

    `loss_fn` defaults to plain cross-entropy, which is what configs A/B/C are
    pre-registered on. A weighted objective must be built from THIS fold's
    training labels only -- see run_cross_validation.
    """
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=5e-4)
    dataset = AugmentedTensorDataset(X, y, augment=augment, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = loss_fn if loss_fn is not None else F.cross_entropy
    epoch_losses = []
    for _ in range(epochs):
        total, n_batches = 0.0, 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            n_batches += 1
        epoch_losses.append(total / max(n_batches, 1))
    if loss_history is not None:
        loss_history.append(epoch_losses)
    return model


def run_cross_validation(data, k=5, epochs=40, device="cpu", model_kind="stgcn",
                         seed=SEED, backbone_path=None, augment=False,
                         loss="ce", joints=None, weights=None,
                         stream="joint", readout=None, observed=None,
                         joint_bn=False, features="marg85"):
    set_determinism(seed)
    num_classes = len(common.CLASS_NAMES)

    tensors = torch.stack(data["tensors"])          # (N, 3, 90, 17)
    labels = torch.tensor(data["labels"], dtype=torch.long)
    if model_kind == "mlp":
        feats = np.stack([representations.features(features, t)
                          for t in data["tensors"]])
        tensors = torch.tensor(feats, dtype=torch.float32)
    elif joints is not None and model_kind == "stgcn":
        # Sliced after prenormalize_2d, which ran at load time over all 17 joints.
        # Renormalizing over the subset would be a second change on the same rung.
        tensors = tensors[:, :, :, joints]

    # Bone vectors are derived AFTER augmentation, so the training and test folds
    # see differences of the same coordinates. Deriving them once up front would
    # hand rotation and cropping quantities that are no longer joint positions.
    to_bone = None
    if stream == "bone" and model_kind == "stgcn":
        parents = bone_parents(joints=joints)
        to_bone = lambda w: transforms.to_bone(w, parents)   # noqa: E731

    n = len(labels)
    oof_pred = np.full(n, -1, dtype=int)
    oof_prob = np.zeros((n, num_classes), dtype=np.float32)
    mc_prob = np.zeros((n, num_classes), dtype=np.float32)
    fold_of = np.full(n, -1, dtype=int)
    fold_metrics = []
    loss_history = []

    # Group by source video, not by video_name: video_name is a clip id, and adjacent
    # clips overlap by 1 s, so clip grouping would put identical frames on both sides
    # of a fold. PREREGISTRATION.md Sec. 5b.2 promises generalization to unseen video.
    for fold_i, (train_idx, test_idx) in enumerate(
            cv_utils.source_video_folds(data["video_names"], k=k)):
        fold_of[test_idx] = fold_i
        model = _make_model(model_kind, num_classes, backbone_path=backbone_path,
                            joints=joints, weights=weights, readout=readout,
                            observed=observed, joint_bn=joint_bn,
                            in_dim=tensors.shape[1] if model_kind == "mlp"
                            else bl.POSE_STAT_DIM)
        # Augmentation is train-fold only, and ST-GCN only: the MLP baseline's
        # tensors were replaced by summary features before this loop, and the
        # transforms are defined on (3, T, V) skeletons.
        fold_augment = None
        if augment and model_kind == "stgcn":
            # Seeded per (seed, fold) so multi-seed runs reproduce exactly.
            fold_augment = SkeletonAugment(
                seed=seed * 1000 + fold_i,
                flip_perm=flip_perm_for(joints) if joints is not None else None,
                **DEFAULT_AUGMENT)
        # Built per fold from the TRAINING labels only. Weights derived from the
        # full labeled set would carry the held-out fold's class prior into the
        # objective -- a small leak, but the entire protocol is cross-video
        # honesty, and a leak in the imbalance arm would be the first thing
        # questioned about it.
        fold_loss = losses.make_loss_fn(loss, labels[train_idx].numpy(),
                                        num_classes, device=device)
        model = _train_fold(model, tensors[train_idx], labels[train_idx], epochs,
                            device, augment=fold_augment,
                            loss_history=loss_history, loss_fn=fold_loss,
                            transform=to_bone)
        model.eval()
        X_test = tensors[test_idx]
        if to_bone is not None:
            X_test = torch.stack([to_bone(w) for w in X_test])
        with torch.no_grad():
            logits = model(X_test.to(device))
            prob = F.softmax(logits, dim=1).cpu().numpy()
        pred = prob.argmax(axis=1)
        oof_pred[test_idx] = pred
        oof_prob[test_idx] = prob
        # Sampled here, while this fold's trained model is still in memory: D
        # then needs no retraining and no saved checkpoints. Skipped for models
        # with no dropout (the GRU baseline) -- there it would cost 20 extra
        # forward passes to reproduce oof_prob exactly, and a saved copy would
        # later read as "MC-dropout changed nothing". NaN instead, so it can
        # never be mistaken for a real distribution.
        if has_dropout(model):
            mc_prob[test_idx] = mc_dropout_probs(model, X_test, device)
        else:
            mc_prob[test_idx] = np.nan
        fold_metrics.append(
            metrics_mod.compute_metrics(labels[test_idx].numpy(), pred, num_classes))

    return {
        "fold_metrics": fold_metrics,
        "oof_pred": oof_pred,
        "oof_prob": oof_prob,
        "mc_prob": mc_prob,
        "oof_true": labels.numpy(),
        "order_window_ids": data["window_ids"],
        "fold_of": fold_of,
        "loss_history": loss_history,
        "model_kind": model_kind,
        "loss": loss,
    }


def save_run_outputs(result, label, model="stgcn"):
    """Persist one model's out-of-fold predictions, metrics and loss trace.

    `fold_of` is saved alongside the probabilities because config D needs it twice
    over: temperature must be fit per fold (fitting it on all the out-of-fold
    predictions at once is fitting on the evaluation set), and configs are compared
    with a PAIRED test across matched folds. It is free to record now and costs a
    full CV re-run to recover later.

    Called for every model_kind, not only the ST-GCN. The MLP baseline scores
    higher than any ST-GCN rung, and while its predictions were being discarded
    the best-performing model in the study was the one model that could not be
    calibrated, plotted, or error-analysed.
    """
    os.makedirs(paths.run_dir(label, model), exist_ok=True)
    out = paths.oof_probs_file(label, model)
    torch.save(
        {"oof_prob": result["oof_prob"], "mc_prob": result["mc_prob"],
         "oof_true": result["oof_true"],
         "window_ids": result["order_window_ids"], "fold_of": result["fold_of"],
         "run_label": label, "model": model},
        out)
    with open(paths.fold_metrics_file(label, model), "w") as f:
        json.dump({"run_label": label, "model": model,
                   "fold_metrics": result["fold_metrics"],
                   "mean": _mean_metrics(result["fold_metrics"])}, f, indent=2)
    with open(paths.train_loss_file(label, model), "w") as f:
        json.dump({"run_label": label, "model": model,
                   "per_fold_epoch_loss": result.get("loss_history", [])}, f)
    return out


def _mean_metrics(fold_metrics):
    """Fold-averaged scalars only; per-class lists are left to the raw records."""
    keys = ["accuracy", "macro_f1", "balanced_acc", "mcc"]
    return {k: float(np.mean([m[k] for m in fold_metrics]))
            for k in keys if k in fold_metrics[0]}


def _summarize(fold_metrics):
    keys = ["macro_f1", "balanced_acc", "mcc"]
    print("\n=== Cross-Video CV (mean +/- std) ===")
    for key in keys:
        vals = np.array([m[key] for m in fold_metrics])
        print(f"  {key:14s}: {vals.mean():.4f} +/- {vals.std():.4f}")


def _save_confusion(oof_true, oof_pred, path, subtitle=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cm = confusion_matrix(oof_true, oof_pred, labels=range(len(common.CLASS_NAMES)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=common.CLASS_NAMES, yticklabels=common.CLASS_NAMES)
    plt.title("Out-of-fold confusion matrix, 5-fold CV grouped by source video"
              + (f"\n{subtitle}" if subtitle else ""))
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    spec = resolve_config(args.config)           # raises for D before any training
    label = args.out_prefix or run_label(args.config, args.seed, args.loss)
    backbone = backbone_path_for(args.config)

    print(f"=== Config {args.config}: {spec['desc']} | seed {args.seed} "
          f"| loss {args.loss} | outputs -> {label}_* ===")
    if args.loss != "ce":
        print("  NOTE: this is the off-ladder imbalance arm (PREREGISTRATION.md "
              "Sec. 5b.8), not a pre-registered rung. Compare it against "
              f"{run_label(args.config, args.seed)}, do not substitute it.")
    if backbone and not os.path.exists(backbone):
        raise FileNotFoundError(
            f"config {args.config} warm-starts from {backbone}, which is missing. "
            "Run steps/pretrain_ntu.py first, or use --config A.")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    data = common.load_labeled_data(
        pt_dir=paths.SAV_WINDOWS_FILTERED_DIR,
        annotations_csv=paths.ANNOTATIONS_FILE,
    )
    print(f"Loaded {len(data['labels'])} labeled windows "
          f"across {len(set(data['video_names']))} clips.")

    report, ok = cv_utils.grouping_report(
        [cv_utils.cv_group_key(v) for v in data["video_names"]], k=args.folds)
    print(report)
    if not ok:
        print("  -> Results from this few source videos should not be reported as final.")

    for kind in args.models:
        print(f"\n### Model: {kind} ###")
        # Only the ST-GCN takes the warm start; the GRU/MLP baselines are a different
        # architecture and always train from scratch, in every config.
        result = run_cross_validation(
            data, k=args.folds, epochs=args.epochs, device=device, model_kind=kind,
            seed=args.seed, backbone_path=backbone if kind == "stgcn" else None,
            augment=augment_for(args.config), loss=args.loss,
            joints=joints_for(args.config), weights=weights_for(args.config),
            stream=stream_for(args.config), readout=readout_for(args.config),
            observed=observed_for(args.config),
            joint_bn=joint_bn_for(args.config))
        _summarize(result["fold_metrics"])
        out = save_run_outputs(result, label, model=kind)
        print(f"  wrote {os.path.dirname(out)}/")
        _save_confusion(result["oof_true"], result["oof_pred"],
                        paths.cv_confusion_matrix_file(label, kind),
                        subtitle=f"{label} | model={kind}")

    # Majority baseline (no training)
    maj = bl.MajorityBaseline().fit(data["labels"])
    preds = maj.predict(len(data["labels"]))
    print("\n### Model: majority ###")
    print("  ", metrics_mod.compute_metrics(data["labels"], preds, len(common.CLASS_NAMES)))


if __name__ == "__main__":
    main()

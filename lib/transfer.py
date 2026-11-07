from collections import OrderedDict

import torch


def is_topology_buffer(key):
    """Whether a state_dict key holds the graph itself rather than a weight.

    The adjacency is rebuilt deterministically from the model's joint set, so
    it is never transferred: keeping it would pin a warm-started model to the
    donor's topology and make rung E1's 13-joint graph unloadable.
    """
    return key.endswith("gcn.A")


def _tile_pooled_data_bn(ckpt, target):
    """Copy a per-channel data_bn onto every node of a per-joint one (rung E7).

    The per-joint layout is feature v*C + c, so repeating the C channel values
    once per node gives each joint the donor's statistics: before fine-tuning
    the two normalizations compute the same function.
    """
    for key in ("data_bn.weight", "data_bn.bias",
                "data_bn.running_mean", "data_bn.running_var"):
        if key in ckpt and key in target and ckpt[key].shape != target[key].shape:
            reps, rem = divmod(target[key].numel(), ckpt[key].numel())
            assert rem == 0, f"{key}: cannot tile {ckpt[key].shape} to {target[key].shape}"
            ckpt[key] = ckpt[key].repeat(reps)
    return ckpt


def load_pretrained_backbone(model, ckpt_path):
    """Load a backbone-only checkpoint (no fc.*) into `model`, leaving model.fc
    at its current (random) init. Returns {"loaded", "skipped"}."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    ckpt = OrderedDict((k, v) for k, v in ckpt.items() if not is_topology_buffer(k))
    fc_in_ckpt = [k for k in ckpt if k.startswith("fc.")]
    assert not fc_in_ckpt, f"checkpoint unexpectedly contains head weights: {fc_in_ckpt}"
    ckpt = _tile_pooled_data_bn(ckpt, model.state_dict())

    result = model.load_state_dict(ckpt, strict=False)
    unexpected = list(result.unexpected_keys)
    assert not unexpected, f"checkpoint has keys not in model: {unexpected}"
    missing = [k for k in result.missing_keys if not is_topology_buffer(k)]
    assert all(k.startswith("fc.") for k in missing), \
        f"backbone keys failed to load: {[k for k in missing if not k.startswith('fc.')]}"

    return {"loaded": list(ckpt.keys()), "skipped": missing}

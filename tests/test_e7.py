"""Rung E7 (per-joint input normalization).

E7 is config C with one change: data_bn normalizes every (joint, channel) pair
with its own statistics, as the reference ST-GCN implementation does
(BatchNorm1d over V*C), instead of pooling one set of statistics per channel
over all joints. The graph, node set, readout and warm start stay C's.
"""
import copy

import numpy as np
import torch

import common
import evaluate
import model as model_mod
import transfer
from pretrain_ntu import export_backbone


DEAD = [13, 14, 15, 16]
SEEN = model_mod.OBSERVED_JOINTS_13


def _train_forward(m, x, seed=7):
    torch.manual_seed(seed)
    with torch.no_grad():
        return m(x)


def _layer1_input(m, x, seed=7):
    seen = {}
    h = m.layer1.register_forward_pre_hook(
        lambda mod, inp: seen.setdefault("x", inp[0]))
    _train_forward(m, x, seed)
    h.remove()
    return seen["x"]


def _load(synthetic_dataset):
    return common.load_labeled_data(
        pt_dir=synthetic_dataset["pt_dir"],
        annotations_csv=synthetic_dataset["annotations_csv"],
    )


def _pooled_backbone(tmp_path):
    torch.manual_seed(0)
    src = model_mod.STGCNClassifier(num_classes=60)
    with torch.no_grad():
        src.data_bn.weight.uniform_(0.5, 1.5)
        src.data_bn.bias.uniform_(-0.5, 0.5)
        src.data_bn.running_mean.uniform_(-1, 1)
        src.data_bn.running_var.uniform_(0.5, 2)
    p = tmp_path / "bb.pth"
    torch.save(export_backbone(src.state_dict()), p)
    return src, str(p)


class TestE7Config:
    def test_e7_is_c_plus_per_joint_input_normalization(self):
        spec = evaluate.resolve_config("E7")
        assert spec["pretrained"] is True and spec["augment"] is True
        assert evaluate.joint_bn_for("E7") is True

    def test_e7_keeps_cs_graph_node_set_stream_readout_and_input(self):
        assert evaluate.joints_for("E7") is None
        assert evaluate.weights_for("E7") is None
        assert evaluate.readout_for("E7") is None
        assert evaluate.observed_for("E7") is None
        assert evaluate.stream_for("E7") == "joint"

    def test_the_other_rungs_pool_over_joints(self):
        for config in ("A", "B", "C", "E1", "E2", "E3", "E5", "E6"):
            assert evaluate.joint_bn_for(config) is False

    def test_cli_accepts_e7(self):
        assert evaluate.build_arg_parser().parse_args(
            ["--config", "E7"]).config == "E7"


class TestPerJointNormalization:
    def test_data_bn_has_one_feature_per_joint_and_channel(self):
        m = model_mod.STGCNClassifier(joint_bn=True)
        assert isinstance(m.data_bn, torch.nn.BatchNorm1d)
        assert m.data_bn.num_features == 3 * 17

    def test_everything_but_data_bn_matches_c(self):
        c = model_mod.STGCNClassifier().state_dict()
        e7 = model_mod.STGCNClassifier(joint_bn=True).state_dict()
        assert c.keys() == e7.keys()
        for k in c:
            if not k.startswith("data_bn."):
                assert c[k].shape == e7[k].shape, k

    def test_each_joint_and_channel_is_normalized_over_itself(self):
        torch.manual_seed(0)
        m = model_mod.STGCNClassifier(joint_bn=True).train()
        x = torch.randn(4, 3, 90, 17) * 3 + 1
        got = _layer1_input(m, x)
        mean = x.mean(dim=(0, 2), keepdim=True)
        var = x.var(dim=(0, 2), unbiased=False, keepdim=True)
        expected = (x - mean) / torch.sqrt(var + m.data_bn.eps)
        torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)

    def test_the_unseen_joints_cannot_move_the_seen_joints_statistics(self):
        torch.manual_seed(0)
        a = model_mod.STGCNClassifier(joint_bn=True).train()
        b = copy.deepcopy(a)
        x = torch.randn(4, 3, 90, 17)
        y = x.clone()
        y[:, :, :, DEAD] = 50.0
        torch.testing.assert_close(_layer1_input(a, x)[..., SEEN],
                                   _layer1_input(b, y)[..., SEEN])

    def test_c_lets_the_unseen_joints_move_the_seen_joints_statistics(self):
        """Guards the test above: pooled normalization does let them bite."""
        torch.manual_seed(0)
        a = model_mod.STGCNClassifier().train()
        b = copy.deepcopy(a)
        x = torch.randn(4, 3, 90, 17)
        y = x.clone()
        y[:, :, :, DEAD] = 50.0
        assert not torch.allclose(_layer1_input(a, x)[..., SEEN],
                                  _layer1_input(b, y)[..., SEEN])

    def test_the_unseen_joints_still_reach_the_backbone(self):
        """Unlike E6, nothing is zeroed: only the statistics are separated."""
        torch.manual_seed(0)
        m = model_mod.STGCNClassifier(joint_bn=True).train()
        got = _layer1_input(m, torch.randn(4, 3, 90, 17))
        assert torch.count_nonzero(got[..., DEAD]) > 0

    def test_joint_bn_and_the_input_mask_are_separate_rungs(self):
        try:
            model_mod.STGCNClassifier(joint_bn=True, observed=SEEN)
        except ValueError:
            return
        raise AssertionError("joint_bn with observed should be refused")

    def test_make_model_passes_joint_bn_to_the_stgcn(self):
        m = evaluate._make_model("stgcn", 2, joint_bn=True)
        assert m.data_bn.num_features == 3 * 17
        assert m.layer1.gcn.A.shape == (3, 17, 17)


class TestWarmStart:
    def test_pooled_statistics_are_copied_to_every_joint(self, tmp_path):
        src, ckpt = _pooled_backbone(tmp_path)
        m = model_mod.STGCNClassifier(joint_bn=True)
        transfer.load_pretrained_backbone(m, ckpt)
        for name in ("weight", "bias", "running_mean", "running_var"):
            got = getattr(m.data_bn, name).view(17, 3)
            want = getattr(src.data_bn, name).expand(17, 3)
            torch.testing.assert_close(got, want)
        torch.testing.assert_close(m.layer1.gcn.conv.weight,
                                   src.layer1.gcn.conv.weight)

    def test_warm_started_e7_equals_warm_started_c_in_eval(self, tmp_path):
        """Before any fine-tuning the two normalizations are the same function."""
        _, ckpt = _pooled_backbone(tmp_path)
        c = model_mod.STGCNClassifier()
        e7 = model_mod.STGCNClassifier(joint_bn=True)
        transfer.load_pretrained_backbone(c, ckpt)
        transfer.load_pretrained_backbone(e7, ckpt)
        e7.fc.load_state_dict(c.fc.state_dict())
        x = torch.randn(4, 3, 90, 17)
        with torch.no_grad():
            torch.testing.assert_close(c.eval()(x), e7.eval()(x))

    def test_c_still_loads_the_pooled_backbone_unchanged(self, tmp_path):
        src, ckpt = _pooled_backbone(tmp_path)
        m = model_mod.STGCNClassifier()
        transfer.load_pretrained_backbone(m, ckpt)
        torch.testing.assert_close(m.data_bn.weight, src.data_bn.weight)


class TestJointBnInCV:
    def test_joint_bn_changes_the_stgcn(self, synthetic_dataset):
        data = _load(synthetic_dataset)
        c = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="stgcn", seed=3)
        e7 = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="stgcn", seed=3,
            joint_bn=True)
        assert not np.array_equal(c["oof_prob"], e7["oof_prob"])

    def test_c_is_unchanged_by_the_new_parameter(self, synthetic_dataset):
        data = _load(synthetic_dataset)
        a = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="stgcn", seed=11)
        b = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="stgcn", seed=11,
            joint_bn=False)
        np.testing.assert_array_equal(a["oof_prob"], b["oof_prob"])

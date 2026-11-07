"""Rung E6 (input-contamination control).

E6 is config C with one change: the four unobserved joints are blocked at the
input. They are excluded from data_bn's statistics and zeroed afterwards, so
they reach no later layer carrying signal. The graph, node set, readout and warm
start stay C's, so E6 - C is the share of E1's gain that is input contamination
rather than the topology.
"""
import copy

import numpy as np
import torch

import common
import evaluate
import model as model_mod


DEAD = [13, 14, 15, 16]


def _train_forward(m, x, seed=7):
    """One training-mode forward, with dropout's draws pinned so two calls differ
    only where their inputs do."""
    torch.manual_seed(seed)
    with torch.no_grad():
        return m(x)


def _load(synthetic_dataset):
    return common.load_labeled_data(
        pt_dir=synthetic_dataset["pt_dir"],
        annotations_csv=synthetic_dataset["annotations_csv"],
    )


class TestE6Config:
    def test_e6_is_c_plus_a_13_joint_input_mask(self):
        spec = evaluate.resolve_config("E6")
        assert spec["pretrained"] is True and spec["augment"] is True
        assert evaluate.observed_for("E6") == model_mod.OBSERVED_JOINTS_13

    def test_e6_keeps_cs_graph_node_set_stream_and_readout(self):
        assert evaluate.joints_for("E6") is None
        assert evaluate.weights_for("E6") is None
        assert evaluate.readout_for("E6") is None
        assert evaluate.stream_for("E6") == "joint"

    def test_the_other_rungs_mask_nothing(self):
        for config in ("A", "B", "C", "E1", "E2", "E3", "E5"):
            assert evaluate.observed_for(config) is None

    def test_cli_accepts_e6(self):
        assert evaluate.build_arg_parser().parse_args(
            ["--config", "E6"]).config == "E6"


class TestObservedInputModel:
    def test_masking_adds_no_parameters_or_buffers(self):
        """The NTU warm start must load into E6 exactly as it loads into C."""
        c = model_mod.STGCNClassifier()
        e6 = model_mod.STGCNClassifier(observed=model_mod.OBSERVED_JOINTS_13)
        assert ({k: v.shape for k, v in c.state_dict().items()}
                == {k: v.shape for k, v in e6.state_dict().items()})

    def test_observing_every_joint_equals_the_default_normalization(self):
        torch.manual_seed(0)
        c = model_mod.STGCNClassifier().eval()
        full = model_mod.STGCNClassifier(observed=list(range(17))).eval()
        full.load_state_dict(c.state_dict())
        x = torch.randn(4, 3, 90, 17)
        with torch.no_grad():
            torch.testing.assert_close(c(x), full(x))

    def test_the_unobserved_joints_reach_the_backbone_as_zeros(self):
        torch.manual_seed(0)
        m = model_mod.STGCNClassifier(observed=model_mod.OBSERVED_JOINTS_13).eval()
        seen = {}
        m.layer1.register_forward_pre_hook(
            lambda mod, inp: seen.setdefault("x", inp[0]))
        with torch.no_grad():
            m(torch.randn(2, 3, 90, 17))
        assert torch.count_nonzero(seen["x"][:, :, :, DEAD]) == 0

    def test_the_observed_joints_are_normalized_over_themselves_only(self):
        torch.manual_seed(0)
        m = model_mod.STGCNClassifier(observed=model_mod.OBSERVED_JOINTS_13).eval()
        seen = {}
        m.layer1.register_forward_pre_hook(
            lambda mod, inp: seen.setdefault("x", inp[0]))
        x = torch.randn(2, 3, 90, 17)
        with torch.no_grad():
            m(x)
            expected = m.data_bn(x[:, :, :, model_mod.OBSERVED_JOINTS_13])
        torch.testing.assert_close(
            seen["x"][:, :, :, model_mod.OBSERVED_JOINTS_13], expected)

    def test_the_unobserved_joints_cannot_change_the_output_in_eval(self):
        torch.manual_seed(0)
        m = model_mod.STGCNClassifier(observed=model_mod.OBSERVED_JOINTS_13).eval()
        x = torch.randn(4, 3, 90, 17)
        y = x.clone()
        y[:, :, :, DEAD] = 50.0
        with torch.no_grad():
            torch.testing.assert_close(m(x), m(y))

    def test_the_unobserved_joints_cannot_change_the_batch_statistics(self):
        """The point of the rung: in train mode data_bn pools over B, T and V,
        so in C the dead joints move the statistics every later layer sees."""
        torch.manual_seed(0)
        a = model_mod.STGCNClassifier(observed=model_mod.OBSERVED_JOINTS_13).train()
        b = copy.deepcopy(a)
        x = torch.randn(4, 3, 90, 17)
        y = x.clone()
        y[:, :, :, DEAD] = 50.0
        torch.testing.assert_close(_train_forward(a, x), _train_forward(b, y))

    def test_c_lets_the_unobserved_joints_move_the_batch_statistics(self):
        """Guards the test above: without the mask the perturbation does bite."""
        torch.manual_seed(0)
        a = model_mod.STGCNClassifier().train()
        b = copy.deepcopy(a)
        x = torch.randn(4, 3, 90, 17)
        y = x.clone()
        y[:, :, :, DEAD] = 50.0
        assert not torch.allclose(_train_forward(a, x), _train_forward(b, y))

    def test_masking_keeps_gradients_flowing_to_the_observed_joints(self):
        torch.manual_seed(0)
        m = model_mod.STGCNClassifier(observed=model_mod.OBSERVED_JOINTS_13)
        x = torch.randn(2, 3, 90, 17, requires_grad=True)
        m(x).sum().backward()
        assert torch.count_nonzero(x.grad[:, :, :, model_mod.OBSERVED_JOINTS_13]) > 0
        assert torch.count_nonzero(x.grad[:, :, :, DEAD]) == 0

    def test_make_model_passes_the_mask_to_the_stgcn(self):
        m = evaluate._make_model("stgcn", 2, observed=model_mod.OBSERVED_JOINTS_13)
        assert m.observed == model_mod.OBSERVED_JOINTS_13
        assert m.layer1.gcn.A.shape == (3, 17, 17)


class TestObservedInCV:
    def test_masking_changes_the_stgcn(self, synthetic_dataset):
        data = _load(synthetic_dataset)
        c = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="stgcn", seed=3)
        e6 = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="stgcn", seed=3,
            observed=model_mod.OBSERVED_JOINTS_13)
        assert not np.array_equal(c["oof_prob"], e6["oof_prob"])

    def test_the_mlp_baseline_ignores_the_mask(self, synthetic_dataset):
        data = _load(synthetic_dataset)
        a = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="mlp", seed=3)
        b = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="mlp", seed=3,
            observed=model_mod.OBSERVED_JOINTS_13)
        np.testing.assert_array_equal(a["oof_prob"], b["oof_prob"])

    def test_c_is_unchanged_by_the_new_parameter(self, synthetic_dataset):
        data = _load(synthetic_dataset)
        a = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="stgcn", seed=11)
        b = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="stgcn", seed=11,
            observed=None)
        np.testing.assert_array_equal(a["oof_prob"], b["oof_prob"])

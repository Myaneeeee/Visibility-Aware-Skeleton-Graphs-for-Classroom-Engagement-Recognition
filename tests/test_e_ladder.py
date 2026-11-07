"""Rung E1 of the observability ladder: the ST-GCN on the 13 observed joints.

E1 is config C with exactly one thing changed -- the graph, and the node set the
graph runs on. Everything else (NTU warm start, augmentation, folds, seeding,
and both baselines) must be bit-identical to C, or the E-vs-C difference stops
being attributable to the topology.
"""
import numpy as np
import torch

import common
import evaluate
import model as model_mod


def _load(synthetic_dataset):
    return common.load_labeled_data(
        pt_dir=synthetic_dataset["pt_dir"],
        annotations_csv=synthetic_dataset["annotations_csv"],
    )


class TestConfigRegistration:
    def test_e1_is_config_c_plus_the_observed_joint_graph(self):
        spec = evaluate.resolve_config("E1")
        assert spec["pretrained"] is True
        assert spec["augment"] is True
        assert spec["joints"] == model_mod.OBSERVED_JOINTS_13

    def test_the_preregistered_ladder_stays_on_the_full_skeleton(self):
        """A/B/C must keep every joint: they are already run and reported."""
        for config in ("A", "B", "C"):
            assert evaluate.joints_for(config) is None

    def test_joints_for_returns_the_subset_for_e1(self):
        assert evaluate.joints_for("E1") == model_mod.OBSERVED_JOINTS_13

    def test_e1_writes_to_its_own_run_label(self):
        assert evaluate.run_label("E1", 51) == "E1_seed51"

    def test_label_still_parses_into_config_and_seed(self):
        config, seed = evaluate.run_label("E1", 51).split("_seed", 1)
        assert config == "E1" and int(seed) == 51

    def test_cli_accepts_e1(self):
        assert evaluate.build_arg_parser().parse_args(
            ["--config", "E1"]).config == "E1"


class TestModelConstruction:
    def test_stgcn_gets_the_subset_graph(self):
        m = evaluate._make_model("stgcn", 2, joints=model_mod.OBSERVED_JOINTS_13)
        assert m.layer1.gcn.A.shape == (3, 13, 13)

    def test_stgcn_defaults_to_the_full_graph(self):
        assert evaluate._make_model("stgcn", 2).layer1.gcn.A.shape == (3, 17, 17)

    def test_baselines_ignore_the_subset(self):
        """The MLP and GRU are the comparison, not the treatment: they keep all
        17 joints under every rung so C's numbers carry over unchanged."""
        import baselines as bl
        mlp = evaluate._make_model("mlp", 2, joints=model_mod.OBSERVED_JOINTS_13)
        assert mlp.net[0].in_features == bl.POSE_STAT_DIM


class TestCrossValidationOnTheSubset:
    def test_stgcn_runs_on_the_subset_and_covers_every_window(self, synthetic_dataset):
        data = _load(synthetic_dataset)
        result = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="stgcn",
            joints=model_mod.OBSERVED_JOINTS_13)
        assert result["oof_prob"].shape == (len(data["labels"]), 2)
        assert sorted(result["order_window_ids"]) == sorted(data["window_ids"])

    def test_augmentation_survives_the_subset(self, synthetic_dataset):
        """The COCO-17 flip permutation would index off the end of a 13-node
        window. E1 augments exactly as C does, so this path must work."""
        data = _load(synthetic_dataset)
        result = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="stgcn", augment=True,
            joints=model_mod.OBSERVED_JOINTS_13)
        assert np.isfinite(result["oof_prob"]).all()

    def test_the_mlp_baseline_is_untouched_by_the_subset(self, synthetic_dataset):
        """Pairing guarantee: E1's MLP column must equal C's at the same seed."""
        data = _load(synthetic_dataset)
        full = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="mlp", seed=3)
        under_e1 = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="mlp", seed=3,
            joints=model_mod.OBSERVED_JOINTS_13)
        np.testing.assert_array_equal(full["oof_prob"], under_e1["oof_prob"])

    def test_the_stgcn_changes_under_the_subset(self, synthetic_dataset):
        """If the joint subset never reached the tensors, E1 would silently be a
        rerun of C and the whole rung would report a null result."""
        data = _load(synthetic_dataset)
        full = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="stgcn", seed=3)
        subset = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="stgcn", seed=3,
            joints=model_mod.OBSERVED_JOINTS_13)
        assert not np.array_equal(full["oof_prob"], subset["oof_prob"])

    def test_the_full_skeleton_run_is_unchanged_by_the_new_parameter(
            self, synthetic_dataset):
        """E0-reproduces-C at the CV level: joints=None must be the old path."""
        data = _load(synthetic_dataset)
        a = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="stgcn", seed=11)
        b = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="stgcn", seed=11,
            joints=None)
        np.testing.assert_array_equal(a["oof_prob"], b["oof_prob"])


class TestModelSelection:
    def test_defaults_to_the_full_set_of_models(self):
        assert evaluate.build_arg_parser().parse_args([]).models == \
            ["stgcn", "gru", "mlp"]

    def test_can_be_restricted_to_the_stgcn(self):
        """E-rungs only move the ST-GCN; rerunning the baselines would spend GPU
        time reproducing numbers C already has."""
        args = evaluate.build_arg_parser().parse_args(["--models", "stgcn"])
        assert args.models == ["stgcn"]

    def test_rejects_a_model_that_does_not_exist(self):
        import pytest
        with pytest.raises(SystemExit):
            evaluate.build_arg_parser().parse_args(["--models", "transformer"])


class TestE2ObservabilityWeighted:
    """E2 keeps all 17 joints and weights the edges instead of cutting them."""

    def test_e2_keeps_every_joint(self):
        assert evaluate.joints_for("E2") is None

    def test_e2_carries_the_measured_confidences(self):
        assert evaluate.weights_for("E2") == model_mod.COCO_JOINT_CONFIDENCE

    def test_the_other_rungs_are_unweighted(self):
        for config in ("A", "B", "C", "E1"):
            assert evaluate.weights_for(config) is None

    def test_e2_model_has_a_full_but_reweighted_graph(self):
        m = evaluate._make_model("stgcn", 2,
                                 weights=model_mod.COCO_JOINT_CONFIDENCE)
        assert m.layer1.gcn.A.shape == (3, 17, 17)
        assert not torch.equal(m.layer1.gcn.A,
                               evaluate._make_model("stgcn", 2).layer1.gcn.A)

    def test_cli_accepts_e2(self):
        assert evaluate.build_arg_parser().parse_args(
            ["--config", "E2"]).config == "E2"

    def test_e2_changes_the_predictions(self, synthetic_dataset):
        data = _load(synthetic_dataset)
        plain = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="stgcn", seed=3)
        weighted = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="stgcn", seed=3,
            weights=model_mod.COCO_JOINT_CONFIDENCE)
        assert not np.array_equal(plain["oof_prob"], weighted["oof_prob"])


class TestE3BoneStream:
    """E3 is E1 with one thing changed: the input is bone vectors."""

    def test_e3_runs_on_the_observed_joints(self):
        assert evaluate.joints_for("E3") == model_mod.OBSERVED_JOINTS_13

    def test_e3_feeds_the_bone_stream(self):
        assert evaluate.stream_for("E3") == "bone"

    def test_the_other_rungs_feed_joint_coordinates(self):
        for config in ("A", "B", "C", "E1", "E2"):
            assert evaluate.stream_for(config) == "joint"

    def test_e3_keeps_cs_warm_start_and_augmentation(self):
        """Changing in_channels would break the NTU transfer, which is why the
        bone stream replaces the coordinates rather than being concatenated."""
        spec = evaluate.resolve_config("E3")
        assert spec["pretrained"] is True and spec["augment"] is True

    def test_cli_accepts_e3(self):
        assert evaluate.build_arg_parser().parse_args(
            ["--config", "E3"]).config == "E3"

    def test_bone_stream_changes_the_predictions(self, synthetic_dataset):
        data = _load(synthetic_dataset)
        joint = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="stgcn", seed=3,
            joints=model_mod.OBSERVED_JOINTS_13)
        bone = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="stgcn", seed=3,
            joints=model_mod.OBSERVED_JOINTS_13, stream="bone")
        assert not np.array_equal(joint["oof_prob"], bone["oof_prob"])

    def test_bone_stream_survives_augmentation(self, synthetic_dataset):
        data = _load(synthetic_dataset)
        result = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="stgcn", augment=True,
            joints=model_mod.OBSERVED_JOINTS_13, stream="bone")
        assert np.isfinite(result["oof_prob"]).all()

    def test_the_mlp_baseline_ignores_the_stream(self, synthetic_dataset):
        data = _load(synthetic_dataset)
        a = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="mlp", seed=3)
        b = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="mlp", seed=3,
            stream="bone")
        np.testing.assert_array_equal(a["oof_prob"], b["oof_prob"])

    def test_the_joint_stream_is_unchanged_by_the_new_parameter(
            self, synthetic_dataset):
        data = _load(synthetic_dataset)
        a = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="stgcn", seed=11)
        b = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="stgcn", seed=11,
            stream="joint")
        np.testing.assert_array_equal(a["oof_prob"], b["oof_prob"])

"""Rung E5 (readout-only control), source-video folds, and the MLP feature probe.

E5 is config C with one change: the classifier's global average pool runs over
the 13 observed joints only. The graph, node set, input and warm start stay C's,
so E1 - E5 is the share of E1's gain that needs the topology change.
"""
import numpy as np
import pytest
import torch

import baselines as bl
import common
import cv_utils
import evaluate
import model as model_mod
import representations as rep


def _load(synthetic_dataset):
    return common.load_labeled_data(
        pt_dir=synthetic_dataset["pt_dir"],
        annotations_csv=synthetic_dataset["annotations_csv"],
    )


def _fold_of(folds, n):
    out = np.full(n, -1)
    for i, (_, te) in enumerate(folds):
        out[te] = i
    return out


class TestSourceVideoFolds:
    NAMES = ["20221208_08_1", "20221208_08_2", "20221208_08_3",
             "20230414_08_1", "20230414_08_2", "20230101_21_1",
             "20230101_21_2", "20230202_33_1", "20230202_33_2"]

    def test_matches_keying_then_splitting(self):
        keyed = [cv_utils.cv_group_key(v) for v in self.NAMES]
        n = len(self.NAMES)
        assert np.array_equal(
            _fold_of(cv_utils.source_video_folds(self.NAMES, k=3), n),
            _fold_of(cv_utils.group_kfold_indices(keyed, k=3), n))

    def test_clips_of_one_source_video_never_straddle_a_fold(self):
        names = np.array(self.NAMES)
        keys = np.array([cv_utils.cv_group_key(v) for v in names])
        for train_idx, test_idx in cv_utils.source_video_folds(names, k=3):
            assert set(keys[train_idx]).isdisjoint(keys[test_idx])

    def test_rejects_names_it_cannot_group(self):
        with pytest.raises(ValueError):
            cv_utils.source_video_folds(["vidA", "vidB", "vidC"], k=3)


class TestE5Config:
    def test_e5_is_c_plus_a_13_joint_readout(self):
        spec = evaluate.resolve_config("E5")
        assert spec["pretrained"] is True and spec["augment"] is True
        assert evaluate.readout_for("E5") == model_mod.OBSERVED_JOINTS_13

    def test_e5_keeps_cs_graph_input_and_node_set(self):
        assert evaluate.joints_for("E5") is None
        assert evaluate.weights_for("E5") is None
        assert evaluate.stream_for("E5") == "joint"

    def test_the_other_rungs_pool_over_every_node(self):
        for config in ("A", "B", "C", "E1", "E2", "E3"):
            assert evaluate.readout_for(config) is None

    def test_cli_accepts_e5(self):
        assert evaluate.build_arg_parser().parse_args(
            ["--config", "E5"]).config == "E5"


class TestReadoutModel:
    def test_readout_adds_no_parameters_or_buffers(self):
        """The NTU warm start must load into E5 exactly as it loads into C."""
        c = model_mod.STGCNClassifier()
        e5 = model_mod.STGCNClassifier(readout=model_mod.OBSERVED_JOINTS_13)
        assert ({k: v.shape for k, v in c.state_dict().items()}
                == {k: v.shape for k, v in e5.state_dict().items()})

    def test_readout_over_every_joint_equals_the_default_pool(self):
        torch.manual_seed(0)
        c = model_mod.STGCNClassifier().eval()
        full = model_mod.STGCNClassifier(readout=list(range(17))).eval()
        full.load_state_dict(c.state_dict())
        x = torch.randn(4, 3, 90, 17)
        with torch.no_grad():
            torch.testing.assert_close(c(x), full(x))

    def test_readout_ignores_the_excluded_nodes_features(self):
        torch.manual_seed(0)
        m = model_mod.STGCNClassifier(readout=model_mod.OBSERVED_JOINTS_13).eval()
        feats = {}
        m.layer3.register_forward_hook(lambda mod, i, o: feats.setdefault("h", o))
        x = torch.randn(2, 3, 90, 17)
        with torch.no_grad():
            out = m(x)
        pooled = feats["h"][:, :, :, model_mod.OBSERVED_JOINTS_13].mean(dim=(2, 3))
        torch.testing.assert_close(out, m.fc(pooled))

    def test_make_model_passes_the_readout_to_the_stgcn(self):
        m = evaluate._make_model("stgcn", 2, readout=model_mod.OBSERVED_JOINTS_13)
        assert m.readout == model_mod.OBSERVED_JOINTS_13
        assert m.layer1.gcn.A.shape == (3, 17, 17)


class TestReadoutInCV:
    def test_readout_changes_the_stgcn(self, synthetic_dataset):
        data = _load(synthetic_dataset)
        c = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="stgcn", seed=3)
        e5 = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="stgcn", seed=3,
            readout=model_mod.OBSERVED_JOINTS_13)
        assert not np.array_equal(c["oof_prob"], e5["oof_prob"])

    def test_the_mlp_baseline_ignores_the_readout(self, synthetic_dataset):
        data = _load(synthetic_dataset)
        a = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="mlp", seed=3)
        b = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="mlp", seed=3,
            readout=model_mod.OBSERVED_JOINTS_13)
        np.testing.assert_array_equal(a["oof_prob"], b["oof_prob"])

    def test_c_is_unchanged_by_the_new_parameter(self, synthetic_dataset):
        data = _load(synthetic_dataset)
        a = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="stgcn", seed=11)
        b = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="stgcn", seed=11,
            readout=None)
        np.testing.assert_array_equal(a["oof_prob"], b["oof_prob"])


class TestRepresentations:
    def _window(self, seed=0):
        g = np.random.default_rng(seed)
        return torch.tensor(g.uniform(-1, 1, (3, 90, 17)), dtype=torch.float32)

    def test_every_family_has_its_documented_width(self):
        x = self._window()
        for name, dim in rep.FEATURE_DIMS.items():
            assert rep.features(name, x).shape == (dim,)

    def test_marg85_is_the_mlp_baselines_vector(self):
        x = self._window()
        np.testing.assert_array_equal(rep.features("marg85", x),
                                      bl.pose_stat_features(x))

    def test_marg65_is_marg85_restricted_to_the_observed_joints(self):
        x = self._window(1)
        full = rep.features("marg85", x)
        o = np.array(model_mod.OBSERVED_JOINTS_13)
        cols = np.concatenate([o, 17 + o, 34 + o, 51 + o, 68 + o])
        np.testing.assert_allclose(rep.features("marg65", x), full[cols], atol=1e-6)

    def test_dist156_is_dist272_restricted_to_observed_pairs(self):
        x = self._window(2)
        iu17 = np.triu_indices(17, k=1)
        keep = np.array([i < 13 and j < 13 for i, j in zip(*iu17)])
        full = rep.features("dist272", x)
        np.testing.assert_allclose(rep.features("dist156", x),
                                   np.concatenate([full[:136][keep], full[136:][keep]]),
                                   atol=1e-5)

    def test_unknown_family_is_rejected(self):
        with pytest.raises(ValueError):
            rep.features("nope", self._window())


class TestFeatureProbeInCV:
    def test_default_features_are_marg85(self, synthetic_dataset):
        data = _load(synthetic_dataset)
        a = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="mlp", seed=5)
        b = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="mlp", seed=5,
            features="marg85")
        np.testing.assert_array_equal(a["oof_prob"], b["oof_prob"])

    def test_a_narrower_family_trains_and_differs(self, synthetic_dataset):
        data = _load(synthetic_dataset)
        a = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="mlp", seed=5)
        b = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="mlp", seed=5,
            features="marg65")
        assert np.isfinite(b["oof_prob"]).all()
        assert not np.array_equal(a["oof_prob"], b["oof_prob"])

    def test_the_stgcn_ignores_the_feature_family(self, synthetic_dataset):
        data = _load(synthetic_dataset)
        a = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="stgcn", seed=5)
        b = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="stgcn", seed=5,
            features="bone144")
        np.testing.assert_array_equal(a["oof_prob"], b["oof_prob"])

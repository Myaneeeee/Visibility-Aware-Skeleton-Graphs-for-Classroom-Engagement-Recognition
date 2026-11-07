"""The MC-dropout analyses: mc_threshold.py, ensemble_without_mc.py, and the
paired_ci.py helpers they share. Runs on synthetic saved predictions."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))

import paths  # noqa: E402
import paired_ci  # noqa: E402
import mc_threshold  # noqa: E402
import ensemble_without_mc  # noqa: E402
from e_ladder_table import SEEDS, fold_rows  # noqa: E402

N_VIDEOS, PER_VIDEO = 20, 12


def _ids_and_folds():
    ids, fold_of = [], []
    for v in range(N_VIDEOS):
        for w in range(PER_VIDEO):
            ids.append(f"2021010{v // 10}_{v:02d}_{w}_id_{w}_f0_f74")
            fold_of.append(v % 5)
    return ids, np.array(fold_of)


def _write_run(label, model, seed, true, ids, fold_of, rng):
    signal = np.where(true == 1, 0.65, 0.35) + rng.normal(0, 0.2, len(true))
    p1 = np.clip(signal, 0.01, 0.99)
    mc1 = np.clip(p1 + 0.08, 0.01, 0.99)
    blob = {"oof_prob": np.stack([1 - p1, p1], 1), "mc_prob": np.stack([1 - mc1, mc1], 1),
            "oof_true": true, "fold_of": fold_of, "window_ids": ids}
    path = paths.oof_probs_file(f"{label}_seed{seed}", model)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(blob, path)


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", str(tmp_path))
    ids, fold_of = _ids_and_folds()
    rng = np.random.default_rng(0)
    true = (rng.random(len(ids)) < 0.25).astype(int)
    for seed in SEEDS:
        for label in ("C", "E1", "E6"):
            _write_run(label, "stgcn", seed, true, ids, fold_of, rng)
        _write_run("C", "mlp", seed, true, ids, fold_of, rng)
    return true, fold_of


class TestPairedCiHelpers:
    def test_predictions_is_the_argmax_of_probabilities(self, runs):
        spec = [("E1", "stgcn", "mc_prob")]
        prob, *_ = paired_ci.probabilities(spec, 51)
        pred, *_ = paired_ci.predictions(spec, 51)
        assert np.array_equal(pred, prob.argmax(1))

    def test_prepare_matches_the_table_estimator(self, runs):
        arms = {"E1": [("E1", "stgcn", "oof_prob")],
                "E1+mc & MLP": [("E1", "stgcn", "mc_prob"), ("C", "mlp", "oof_prob")]}
        point, boot = paired_ci.prepare(arms, 50, np.random.default_rng(0))
        for name, specs in arms.items():
            table = [np.mean([r["macro_f1"] for r in fold_rows(specs, s)]) for s in SEEDS]
            assert np.allclose(point[name], table)
            assert boot[name].shape == (50, len(SEEDS))

    def test_missing_runs_drop_the_arm(self, runs):
        arms = {"E1": [("E1", "stgcn", "oof_prob")], "E7": [("E7", "stgcn", "oof_prob")]}
        point, _ = paired_ci.prepare(arms, 10, np.random.default_rng(0))
        assert set(point) == {"E1"}


class TestThreshold:
    def test_vectorized_macro_f1_matches_sklearn(self):
        import metrics as metrics_mod
        rng = np.random.default_rng(2)
        p = rng.random(300)
        true = (rng.random(300) < 0.2).astype(int)
        fast = mc_threshold.macro_f1_per_threshold(p, true)
        slow = [metrics_mod.compute_metrics(true, (p >= t).astype(int), 2)["macro_f1"]
                for t in mc_threshold.GRID]
        assert np.allclose(fast, slow)

    def test_best_threshold_separates_clean_data(self):
        p = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
        true = np.array([0, 0, 0, 1, 1, 1])
        t = mc_threshold.best_threshold(p, true)
        assert 0.3 < t <= 0.6

    def test_scored_fold_labels_do_not_move_its_threshold(self):
        rng = np.random.default_rng(1)
        p = rng.random(200)
        true = (p + rng.normal(0, 0.2, 200) > 0.6).astype(int)
        fold_of = np.arange(200) % 5
        before = mc_threshold.lofo_thresholds(p, true, fold_of)
        flipped = true.copy()
        flipped[fold_of == 0] = 1 - flipped[fold_of == 0]
        after = mc_threshold.lofo_thresholds(p, flipped, fold_of)
        assert before[0] == after[0]
        assert any(before[f] != after[f] for f in range(1, 5))

    def test_thresholded_applies_each_folds_own_threshold(self):
        p = np.array([0.3, 0.3, 0.3, 0.3])
        fold_of = np.array([0, 0, 1, 1])
        pred = mc_threshold.thresholded(p, fold_of, {0: 0.2, 1: 0.4})
        assert pred.tolist() == [1, 1, 0, 0]

    def test_untuned_arm_equals_the_argmax_arm(self, runs):
        pred, *_ = mc_threshold.predict(mc_threshold.arm("C", "mc_prob", False), 51)
        ref, *_ = paired_ci.predictions([("C", "stgcn", "mc_prob")], 51)
        assert np.array_equal(pred, ref)

    def test_auroc_ignores_a_pure_probability_shift(self, runs):
        """The synthetic mc_prob is oof_prob + 0.08 (clipped): same ranking."""
        _, _, auc_plain, _ = mc_threshold.operating_point(mc_threshold.arm("E1", "oof_prob", False))
        _, _, auc_mc, _ = mc_threshold.operating_point(mc_threshold.arm("E1", "mc_prob", False))
        assert np.allclose(auc_plain, auc_mc, atol=2e-3)

    def test_main_runs_end_to_end(self, runs, capsys, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["mc_threshold.py", "--draws", "20"])
        mc_threshold.main()
        out = capsys.readouterr().out
        assert "E1 + thr -> E1+mc" in out
        assert "runs missing" not in out


class TestEnsembleWithoutMc:
    def test_one_pass_arms_read_no_mc_predictions(self):
        for name, specs in ensemble_without_mc.ARM.items():
            if "+mc" not in name:
                assert all(key == "oof_prob" for _, _, key, *_ in specs), name

    def test_each_mc_ensemble_has_a_one_pass_twin(self):
        arm = ensemble_without_mc.ARM
        for name, specs in arm.items():
            if "+mc &" in name:
                twin = arm[name.replace("+mc", "")]
                assert [s[:2] for s in twin] == [s[:2] for s in specs]

    def test_main_runs_end_to_end(self, runs, capsys, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["ensemble_without_mc.py", "--draws", "20"])
        ensemble_without_mc.main()
        out = capsys.readouterr().out
        assert "E1 & MLP -> E1+mc & MLP" in out
        assert "runs missing" not in out

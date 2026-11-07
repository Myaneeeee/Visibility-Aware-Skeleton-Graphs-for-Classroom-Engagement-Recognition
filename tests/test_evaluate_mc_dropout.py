"""MC-dropout: a second uncertainty source for config D.

The trap this file exists to pin: model.train() enables dropout AND switches
BatchNorm to batch statistics while mutating its running stats. That corrupts
predictions and the model itself, and it fails silently -- you get plausible
numbers that are wrong.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

import baselines as bl
import common
import evaluate
from model import STGCNClassifier


def _model():
    torch.manual_seed(0)
    return STGCNClassifier(in_channels=3, hidden_dim=128, num_classes=2)


class TestRngIsolation:
    """MC-dropout runs between fold i's evaluation and fold i+1's model init and
    DataLoader shuffle. If its 20 stochastic passes draw from the global torch
    RNG, then MC_PASSES silently changes the weights that configs A/B/C train --
    D's uncertainty machinery reaching back into the classifier the ladder says
    it leaves untouched."""

    def test_mc_passes_leave_the_global_rng_untouched(self):
        m = _model()
        X = torch.randn(4, 3, 90, 17)
        torch.manual_seed(1234)
        before = torch.get_rng_state()
        evaluate.mc_dropout_probs(m, X, "cpu", passes=5)
        assert torch.equal(torch.get_rng_state(), before)

    def test_the_next_draw_is_unaffected_by_how_many_passes_were_taken(self):
        """The failure this prevents: fold i+1's init depends on MC_PASSES."""
        X = torch.randn(4, 3, 90, 17)

        def next_draw_after(passes):
            model = _model()          # seeds the RNG itself; keep it out of the window
            torch.manual_seed(99)
            evaluate.mc_dropout_probs(model, X, "cpu", passes=passes)
            return torch.randn(3)

        assert torch.equal(next_draw_after(2), next_draw_after(20))

    def test_the_rng_is_restored_even_when_a_pass_raises(self):
        model = _model()
        bad_shape = torch.randn(4, 9)
        torch.manual_seed(7)
        before = torch.get_rng_state()
        with pytest.raises(ValueError, match="4D input"):
            evaluate.mc_dropout_probs(model, bad_shape, "cpu", passes=3)
        assert torch.equal(torch.get_rng_state(), before)


class TestDropoutFreeModels:
    def test_has_dropout_distinguishes_the_stgcn_from_the_gru_baseline(self):
        assert evaluate.has_dropout(_model())
        assert not evaluate.has_dropout(bl.KeypointGRU(num_classes=2))

    def test_mc_dropout_refuses_a_model_with_no_dropout(self):
        """Sampling a dropout-free model returns the deterministic prediction 20
        times over; saved, that reads as 'MC-dropout changed nothing'."""
        with pytest.raises(ValueError, match="no nn.Dropout"):
            evaluate.enable_mc_dropout(bl.KeypointGRU(num_classes=2))


class TestEnableMcDropout:
    def test_dropout_modules_are_training(self):
        m = _model()
        evaluate.enable_mc_dropout(m)
        drops = [mod for mod in m.modules() if isinstance(mod, nn.Dropout)]
        assert drops, "the model has no dropout to sample from"
        assert all(d.training for d in drops)

    def test_batchnorm_stays_in_eval_mode(self):
        m = _model()
        evaluate.enable_mc_dropout(m)
        bns = [mod for mod in m.modules()
               if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d))]
        assert bns, "expected BatchNorm in the ST-GCN"
        assert not any(b.training for b in bns)

    def test_batchnorm_running_stats_are_not_mutated(self):
        """The classic MC-dropout bug: an MC pass silently updating BN stats."""
        m = _model()
        x = torch.randn(8, 3, 90, 17)
        before = [b.running_mean.clone() for b in m.modules()
                  if isinstance(b, (nn.BatchNorm1d, nn.BatchNorm2d))]
        evaluate.mc_dropout_probs(m, x, device="cpu", passes=5)
        after = [b.running_mean.clone() for b in m.modules()
                 if isinstance(b, (nn.BatchNorm1d, nn.BatchNorm2d))]
        for b0, b1 in zip(before, after):
            torch.testing.assert_close(b0, b1)


class TestMcDropoutProbs:
    def test_returns_a_probability_distribution(self):
        m = _model()
        x = torch.randn(6, 3, 90, 17)
        p = evaluate.mc_dropout_probs(m, x, device="cpu", passes=5)
        assert p.shape == (6, 2)
        np.testing.assert_allclose(p.sum(axis=1), np.ones(6), rtol=1e-5)

    def test_differs_from_the_deterministic_prediction(self):
        """If it matched exactly, dropout was not actually active."""
        m = _model()
        x = torch.randn(16, 3, 90, 17)
        m.eval()
        with torch.no_grad():
            deterministic = torch.softmax(m(x), dim=1).numpy()
        mc = evaluate.mc_dropout_probs(m, x, device="cpu", passes=20)
        assert not np.allclose(deterministic, mc)

    def test_with_dropout_disabled_it_matches_the_deterministic_prediction(self):
        m = _model()
        for mod in m.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.p = 0.0
        x = torch.randn(6, 3, 90, 17)
        m.eval()
        with torch.no_grad():
            deterministic = torch.softmax(m(x), dim=1).numpy()
        mc = evaluate.mc_dropout_probs(m, x, device="cpu", passes=5)
        np.testing.assert_allclose(deterministic, mc, rtol=1e-4, atol=1e-6)

    def test_default_pass_count_is_twenty(self):
        assert evaluate.MC_PASSES == 20


class TestMcProbInResults:
    def test_cross_validation_returns_mc_probabilities(self, synthetic_dataset):
        data = common.load_labeled_data(
            pt_dir=synthetic_dataset["pt_dir"],
            annotations_csv=synthetic_dataset["annotations_csv"])
        result = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="stgcn", seed=42)
        n = len(data["labels"])
        assert result["mc_prob"].shape == (n, 2)
        np.testing.assert_allclose(result["mc_prob"].sum(axis=1), np.ones(n),
                                   rtol=1e-4)

    def test_saved_file_carries_mc_probabilities(self, synthetic_dataset, tmp_path,
                                                 monkeypatch):
        import paths
        monkeypatch.setattr(paths, "OUTPUTS_DIR", str(tmp_path))
        data = common.load_labeled_data(
            pt_dir=synthetic_dataset["pt_dir"],
            annotations_csv=synthetic_dataset["annotations_csv"])
        result = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="stgcn", seed=42)
        evaluate.save_run_outputs(result, "C_seed42")
        blob = torch.load(paths.oof_probs_file("C_seed42"), weights_only=False)
        assert "mc_prob" in blob
        np.testing.assert_array_equal(blob["mc_prob"], result["mc_prob"])

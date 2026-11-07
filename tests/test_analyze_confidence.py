"""The config D report: turns one saved run into the calibration table."""
import numpy as np
import pytest
import torch

import analyze_confidence


def _blob(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 1.5, size=(n, 2))
    z = logits - logits.max(axis=1, keepdims=True)
    prob = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    y = (rng.random(n) > prob[:, 0]).astype(int)
    over = np.exp(z * 3.0) / np.exp(z * 3.0).sum(axis=1, keepdims=True)
    return {
        "oof_prob": over.astype(np.float32),
        "mc_prob": prob.astype(np.float32),
        "oof_true": y,
        "fold_of": np.repeat(np.arange(5), n // 5),
        "window_ids": [f"w{i}" for i in range(n)],
        "run_label": "C_seed42",
    }


class TestConfidenceReport:
    def test_reports_the_three_named_variants(self):
        report = analyze_confidence.confidence_report(_blob())
        assert sorted(report["variants"]) == ["+mc", "+temp", "raw"]

    def test_every_variant_carries_the_full_metric_set(self):
        report = analyze_confidence.confidence_report(_blob())
        for name, metrics in report["variants"].items():
            assert set(metrics) >= {"brier", "nll", "ece", "aurc", "accuracy"}, name

    def test_temperature_scaling_improves_calibration_of_overconfident_output(self):
        report = analyze_confidence.confidence_report(_blob())
        assert (report["variants"]["+temp"]["nll"]
                < report["variants"]["raw"]["nll"])

    def test_temperature_scaling_leaves_accuracy_identical(self):
        """The identical-accuracy property that the ablation table footnotes."""
        report = analyze_confidence.confidence_report(_blob())
        assert (report["variants"]["+temp"]["accuracy"]
                == report["variants"]["raw"]["accuracy"])

    def test_reports_one_temperature_per_fold(self):
        report = analyze_confidence.confidence_report(_blob())
        assert sorted(report["temperatures"]) == [0, 1, 2, 3, 4]

    def test_missing_mc_probabilities_are_reported_not_guessed(self):
        blob = _blob()
        del blob["mc_prob"]
        with pytest.raises(KeyError, match="mc_prob"):
            analyze_confidence.confidence_report(blob)


class TestCli:
    def test_main_reads_a_saved_run_and_prints_the_table(self, tmp_path, capsys):
        path = tmp_path / "C_seed42_oof_probs.pt"
        torch.save(_blob(), path)
        analyze_confidence.main([str(path)])
        out = capsys.readouterr().out
        assert "raw" in out
        assert "+temp" in out
        assert "+mc" in out
        assert "brier" in out.lower()

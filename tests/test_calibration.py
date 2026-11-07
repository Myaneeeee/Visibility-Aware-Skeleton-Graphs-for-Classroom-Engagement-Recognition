"""Calibration and abstention for config D."""
import numpy as np
import pytest

import calibration


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _calibrated_set(n=4000, seed=0):
    """Probabilities whose stated confidence matches reality, by construction:
    labels are SAMPLED from the probabilities themselves."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 1.5, size=(n, 2))
    prob = _softmax(logits)
    y = (rng.random(n) > prob[:, 0]).astype(int)
    return logits, prob, y


class TestApplyTemperature:
    def test_temperature_one_changes_nothing(self):
        _, prob, _ = _calibrated_set(n=100)
        np.testing.assert_allclose(calibration.apply_temperature(prob, 1.0), prob,
                                   rtol=1e-6, atol=1e-8)

    def test_output_rows_are_probability_distributions(self):
        _, prob, _ = _calibrated_set(n=100)
        out = calibration.apply_temperature(prob, 2.5)
        np.testing.assert_allclose(out.sum(axis=1), np.ones(len(out)), rtol=1e-6)

    def test_high_temperature_reduces_confidence(self):
        _, prob, _ = _calibrated_set(n=100)
        out = calibration.apply_temperature(prob, 3.0)
        assert out.max(axis=1).mean() < prob.max(axis=1).mean()

    def test_it_never_changes_the_predicted_class(self):
        """Temperature scaling is monotonic. This is exactly why D-temp's
        accuracy, macro-F1 and MCC are identical to C's by construction."""
        _, prob, _ = _calibrated_set(n=500)
        for T in (0.4, 1.0, 2.0, 7.5):
            np.testing.assert_array_equal(
                calibration.apply_temperature(prob, T).argmax(axis=1),
                prob.argmax(axis=1))


class TestFitTemperature:
    def test_already_calibrated_data_yields_temperature_near_one(self):
        _, prob, y = _calibrated_set()
        assert 0.85 < calibration.fit_temperature(prob, y) < 1.15

    def test_overconfident_data_yields_temperature_above_one(self):
        logits, _, y = _calibrated_set()
        overconfident = _softmax(logits * 3.0)
        assert calibration.fit_temperature(overconfident, y) > 1.8

    def test_it_recovers_a_known_scaling(self):
        logits, _, y = _calibrated_set()
        overconfident = _softmax(logits * 3.0)
        T = calibration.fit_temperature(overconfident, y)
        assert 2.4 < T < 3.6, f"expected ~3.0, got {T}"

    def test_underconfident_data_yields_temperature_below_one(self):
        logits, _, y = _calibrated_set()
        underconfident = _softmax(logits * 0.4)
        assert calibration.fit_temperature(underconfident, y) < 0.8

    def test_temperature_is_always_positive(self):
        _, prob, y = _calibrated_set(n=200)
        assert calibration.fit_temperature(prob, y) > 0


class TestLeaveOneFoldOutTemperature:
    def _folded(self, n_per_fold=1200, n_folds=5, seed=1):
        rng = np.random.default_rng(seed)
        logits = rng.normal(0, 1.5, size=(n_per_fold * n_folds, 2))
        prob = _softmax(logits)
        y = (rng.random(len(prob)) > prob[:, 0]).astype(int)
        fold_of = np.repeat(np.arange(n_folds), n_per_fold)
        return logits, prob, y, fold_of

    def test_returns_one_temperature_per_fold(self):
        _, prob, y, fold_of = self._folded()
        _, temps = calibration.leave_one_fold_out_temperature(prob, y, fold_of)
        assert sorted(temps) == [0, 1, 2, 3, 4]

    def test_output_shape_matches_the_input(self):
        _, prob, y, fold_of = self._folded()
        cal, _ = calibration.leave_one_fold_out_temperature(prob, y, fold_of)
        assert cal.shape == prob.shape
        np.testing.assert_allclose(cal.sum(axis=1), np.ones(len(cal)), rtol=1e-6)

    def test_a_folds_temperature_never_sees_that_fold(self):
        """The anti-circularity guarantee, and the whole reason this function
        exists instead of one global fit.

        Fold 0 is made wildly overconfident while folds 1-4 stay calibrated. If
        fold 0's temperature were fit on its own data it would come out large;
        fit on folds 1-4 it must come out near 1.
        """
        logits, prob, y, fold_of = self._folded()
        prob = prob.copy()
        fold0 = fold_of == 0
        prob[fold0] = _softmax(logits[fold0] * 8.0)

        _, temps = calibration.leave_one_fold_out_temperature(prob, y, fold_of)
        assert 0.8 < temps[0] < 1.25, (
            f"fold 0's temperature is {temps[0]}, which means its own "
            "miscalibrated data leaked into its own fit")

    def test_predicted_classes_are_unchanged(self):
        _, prob, y, fold_of = self._folded()
        cal, _ = calibration.leave_one_fold_out_temperature(prob, y, fold_of)
        np.testing.assert_array_equal(cal.argmax(axis=1), prob.argmax(axis=1))

    def test_rejects_unassigned_windows(self):
        """fold_of is -1 for any window no fold held out. Silently calibrating
        those would mean calibrating a prediction that does not exist."""
        _, prob, y, fold_of = self._folded(n_per_fold=100, n_folds=2)
        fold_of = fold_of.copy()
        fold_of[0] = -1
        with pytest.raises(ValueError, match="-1"):
            calibration.leave_one_fold_out_temperature(prob, y, fold_of)


class TestProperScoringRules:
    def test_brier_matches_a_hand_computed_value(self):
        prob = np.array([[0.7, 0.3]])
        y = np.array([0])
        # (0.7-1)^2 + (0.3-0)^2 = 0.09 + 0.09 = 0.18
        assert abs(calibration.brier_score(prob, y) - 0.18) < 1e-9

    def test_brier_is_zero_for_perfect_confident_predictions(self):
        prob = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert calibration.brier_score(prob, np.array([0, 1])) == pytest.approx(0.0)

    def test_brier_is_worse_when_confidently_wrong(self):
        y = np.array([0])
        right = calibration.brier_score(np.array([[0.9, 0.1]]), y)
        wrong = calibration.brier_score(np.array([[0.1, 0.9]]), y)
        assert wrong > right

    def test_nll_matches_a_hand_computed_value(self):
        prob = np.array([[0.7, 0.3]])
        assert abs(calibration.nll(prob, np.array([0])) - (-np.log(0.7))) < 1e-9

    def test_nll_punishes_confident_mistakes_hard(self):
        y = np.array([0])
        assert calibration.nll(np.array([[0.01, 0.99]]), y) > 4.0

    def test_nll_is_finite_for_a_zero_probability(self):
        """Clamping matters: an unclamped log(0) would make the whole report inf."""
        assert np.isfinite(calibration.nll(np.array([[0.0, 1.0]]), np.array([0])))


class TestEce:
    def test_perfectly_calibrated_data_scores_near_zero(self):
        _, prob, y = _calibrated_set(n=8000)
        assert calibration.ece_equal_mass(prob, y) < 0.05

    def test_overconfident_data_scores_higher(self):
        logits, prob, y = _calibrated_set(n=8000)
        overconfident = _softmax(logits * 3.0)
        assert (calibration.ece_equal_mass(overconfident, y)
                > calibration.ece_equal_mass(prob, y))

    def test_bins_hold_equal_counts(self):
        """Equal-MASS bins, not equal-width. At n~679 equal-width bins leave some
        nearly empty, which is what makes ECE unstable at this scale."""
        _, prob, y = _calibrated_set(n=1000)
        _, _, counts = calibration.reliability_curve(prob, y, n_bins=10)
        assert counts.max() - counts.min() <= 1


class TestReliabilityCurve:
    def test_returns_one_entry_per_bin(self):
        _, prob, y = _calibrated_set(n=1000)
        conf, acc, counts = calibration.reliability_curve(prob, y, n_bins=10)
        assert len(conf) == len(acc) == len(counts) == 10

    def test_calibrated_data_tracks_the_diagonal(self):
        _, prob, y = _calibrated_set(n=8000)
        conf, acc, _ = calibration.reliability_curve(prob, y, n_bins=10)
        assert np.abs(conf - acc).max() < 0.1

    def test_confidence_increases_across_bins(self):
        _, prob, y = _calibrated_set(n=1000)
        conf, _, _ = calibration.reliability_curve(prob, y, n_bins=10)
        assert np.all(np.diff(conf) > 0)


class TestRiskCoverage:
    def test_curve_matches_a_hand_computed_case(self):
        """Four predictions, confidences 0.9 > 0.8 > 0.7 > 0.6, and the 0.7 one
        is the only mistake. Sorted most-confident-first the errors are
        [0,0,1,0], so cumulative risk is [0, 0, 1/3, 1/4]."""
        prob = np.array([[0.9, 0.1],
                         [0.8, 0.2],
                         [0.3, 0.7],
                         [0.6, 0.4]])
        y = np.array([0, 0, 0, 0])
        coverage, risk = calibration.risk_coverage_curve(prob, y)
        np.testing.assert_allclose(coverage, [0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(risk, [0.0, 0.0, 1 / 3, 0.25])

    def test_aurc_matches_a_hand_computed_case(self):
        prob = np.array([[0.9, 0.1],
                         [0.8, 0.2],
                         [0.3, 0.7],
                         [0.6, 0.4]])
        y = np.array([0, 0, 0, 0])
        expected = (0.0 + 0.0 + 1 / 3 + 0.25) / 4
        assert abs(calibration.aurc(prob, y) - expected) < 1e-9

    def test_full_coverage_risk_equals_the_overall_error_rate(self):
        _, prob, y = _calibrated_set(n=2000)
        _, risk = calibration.risk_coverage_curve(prob, y)
        error_rate = (prob.argmax(axis=1) != y).mean()
        assert abs(risk[-1] - error_rate) < 1e-9

    def test_coverage_runs_from_just_above_zero_to_one(self):
        _, prob, y = _calibrated_set(n=500)
        coverage, _ = calibration.risk_coverage_curve(prob, y)
        assert coverage[0] == pytest.approx(1 / 500)
        assert coverage[-1] == pytest.approx(1.0)

    def test_a_perfect_model_has_zero_aurc(self):
        prob = np.array([[0.99, 0.01], [0.01, 0.99]])
        assert calibration.aurc(prob, np.array([0, 1])) == pytest.approx(0.0)

    def test_useful_confidence_beats_random_confidence(self):
        """The point of the metric: a model whose confidence ranks its own
        mistakes last scores better than one whose confidence is noise.

        Same predictions in both cases -- only the confidence ordering differs --
        so any AURC gap is attributable to the confidence alone.
        """
        rng = np.random.default_rng(3)
        _, prob, y = _calibrated_set(n=3000)
        pred = prob.argmax(axis=1)
        rows = np.arange(len(prob))

        noise = rng.uniform(0.51, 0.99, size=len(prob))
        scrambled = np.zeros_like(prob)
        scrambled[rows, pred] = noise
        scrambled[rows, 1 - pred] = 1.0 - noise

        np.testing.assert_array_equal(scrambled.argmax(axis=1), pred)
        assert calibration.aurc(prob, y) < calibration.aurc(scrambled, y)


class TestTemperatureGuards:
    """fit_temperature's output is divided into every logit downstream, so an
    unusable T must fail loudly here rather than surface as NaN in the report."""

    def test_a_fully_saturated_float32_input_is_flagged(self, capsys):
        """The trap: float32 rounds a large logit gap to exactly 1.0/0.0, the
        cross-entropy gradient underflows, and the fit returns T=1.0 --
        'perfectly calibrated' for the most overconfident input possible."""
        prob = np.zeros((200, 2), dtype=np.float32)
        y = (np.arange(200) % 2)
        prob[np.arange(200), y] = 1.0          # every prediction maximally confident

        T = calibration.fit_temperature(prob, y)
        warning = capsys.readouterr().out

        assert "saturated" in warning, (
            "a saturated fit must announce itself; silently returning T=1.0 "
            "reports the most overconfident possible input as well calibrated")
        assert np.isfinite(T) and T > 0

    def test_an_unsaturated_fit_says_nothing(self, capsys):
        _, prob, y = _calibrated_set(n=400)
        calibration.fit_temperature(prob, y)
        assert "saturated" not in capsys.readouterr().out

    def test_the_returned_temperature_is_always_usable(self):
        _, prob, y = _calibrated_set(n=400)
        T = calibration.fit_temperature(prob, y)
        assert np.isfinite(T) and T > 0
        np.testing.assert_array_equal(
            np.isfinite(calibration.apply_temperature(prob, T)),
            np.ones_like(prob, dtype=bool))

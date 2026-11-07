"""The imbalance arm must never be mistaken for a pre-registered rung.

PREREGISTRATION.md Sec. 5b.3 fixed the training objective before any result existed.
A "+w"/"+f" run changes exactly that, so however well it scores it has not beaten
the ladder -- it has answered a different question. The failure this guards is
silent: build_ablation picks the best config by macro-F1, and an arm run that
happened to win would have quietly become "Best ST-GCN rung" in the headline of
02_ablation.md with nothing to flag it.
"""
import pytest

import make_report as mr


class TestLossArmDetection:
    def test_plain_configs_are_not_arms(self):
        for c in ("A", "B", "C", mr.MC_RUNG):
            assert mr.loss_arm_of(c) == (c, None)

    def test_weighted_config_reports_its_base_rung(self):
        assert mr.loss_arm_of("C+w") == ("C", "+w")

    def test_focal_config_reports_its_base_rung(self):
        assert mr.loss_arm_of("B+f") == ("B", "+f")

    def test_every_arm_suffix_is_detected(self):
        for suffix in mr.LOSS_ARM_SUFFIX:
            base, found = mr.loss_arm_of("A" + suffix)
            assert (base, found) == ("A", suffix)

    def test_arms_are_excluded_from_the_ladder(self):
        for suffix in mr.LOSS_ARM_SUFFIX:
            assert "C" + suffix not in mr.LADDER_CONFIGS

    def test_the_ladder_is_exactly_the_preregistered_rungs(self):
        assert set(mr.LADDER_CONFIGS) == {"A", "B", "C", mr.MC_RUNG}


def seed_means_from(scores, seeds=(51, 52)):
    """{(config, model): {seed: metrics}} with one macro-F1 per (config, model)."""
    return {key: {s: {k: f1 for k in mr.SCALARS} for s in seeds}
            for key, f1 in scores.items()}


class TestHeadlineIgnoresTheArm:
    """The headline comparison must quote the ladder, whatever the arm scores."""

    WINNING_ARM = {("A", "stgcn"): 0.60, ("B", "stgcn"): 0.70,
                   ("C", "stgcn"): 0.69, ("C+w", "stgcn"): 0.95,
                   ("A", "mlp"): 0.73, ("C+w", "mlp"): 0.99}

    def test_best_rung_is_never_an_arm(self):
        best = mr.best_ladder_config_per_model(seed_means_from(self.WINNING_ARM))
        assert best["stgcn"] == ("B", pytest.approx(0.70)), \
            "an arm run hijacked the headline ST-GCN rung"

    def test_best_baseline_is_never_an_arm(self):
        """The same hazard on the baseline side: the MLP is the study's bar."""
        best = mr.best_ladder_config_per_model(seed_means_from(self.WINNING_ARM))
        assert best["mlp"] == ("A", pytest.approx(0.73))

    def test_the_mc_rung_is_still_eligible(self):
        """D+mc is a real rung and currently the best one -- excluding the arm
        must not exclude it too."""
        best = mr.best_ladder_config_per_model(seed_means_from(
            {("B", "stgcn"): 0.70, (mr.MC_RUNG, "stgcn"): 0.72}))
        assert best["stgcn"][0] == mr.MC_RUNG

    def test_a_ladder_only_result_is_unchanged(self):
        """Regression: with no arm on disk the headline must be what it was."""
        best = mr.best_ladder_config_per_model(seed_means_from(
            {("A", "stgcn"): 0.6461, ("B", "stgcn"): 0.7025,
             ("C", "stgcn"): 0.6992, (mr.MC_RUNG, "stgcn"): 0.7202,
             ("A", "mlp"): 0.7370}))
        assert best["stgcn"][0] == mr.MC_RUNG
        assert best["mlp"][0] == "A"

    def test_an_arm_pairs_against_its_own_base_rung(self):
        seed_means = {("C", "stgcn"): {}, ("C+w", "stgcn"): {},
                      ("B+f", "stgcn"): {}, ("A", "mlp"): {}}
        pairs = sorted({(base, c) for (c, m) in seed_means if m == "stgcn"
                        for base, suffix in [mr.loss_arm_of(c)] if suffix})
        assert pairs == [("B", "B+f"), ("C", "C+w")]

    def test_no_arm_yields_no_extra_comparisons(self):
        seed_means = {("A", "stgcn"): {}, ("B", "stgcn"): {}, ("C", "stgcn"): {}}
        pairs = {(base, c) for (c, m) in seed_means if m == "stgcn"
                 for base, suffix in [mr.loss_arm_of(c)] if suffix}
        assert pairs == set()

"""make_report's reporting guarantees.

These pin the parts of PREREGISTRATION.md Sec. 5b.3 that are promises about the
REPORT rather than about the experiment: which variants appear, which metrics
carry the calibration claims, and the footnote that stops config D from looking
like an independently-measured rung.
"""
import json
import os

import numpy as np
import pytest
import torch

import make_report
import paths


def _blob(label, model, n=120, seed=0, dropout=True):
    """A saved-run dict shaped exactly like evaluate.save_run_outputs writes."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(np.int64)
    # Probabilities correlated with the truth, so metrics are not degenerate.
    p1 = np.clip(0.25 + 0.5 * y + rng.normal(0, 0.15, n), 0.02, 0.98)
    prob = np.stack([1 - p1, p1], axis=1)
    mc = np.clip(prob + rng.normal(0, 0.03, prob.shape), 0.01, 0.99)
    mc = mc / mc.sum(axis=1, keepdims=True)
    return {
        "oof_prob": prob.astype(np.float32),
        "mc_prob": (mc if dropout
                    else np.full_like(prob, np.nan)).astype(np.float32),
        "oof_true": y,
        "fold_of": np.tile(np.arange(5), n // 5)[:n],
        "window_ids": [f"w{i}" for i in range(n)],
        "run_label": label,
        "model": model,
    }


@pytest.fixture
def runs(tmp_path, monkeypatch):
    """Three configs x one seed x stgcn+mlp, written where make_report looks."""
    monkeypatch.setattr(paths, "OUTPUTS_DIR", str(tmp_path), raising=False)
    written = {}
    for i, config in enumerate(["A", "B", "C"]):
        for model in ["stgcn", "mlp"]:
            label = f"{config}_seed7"
            d = tmp_path / "runs" / label / model
            d.mkdir(parents=True)
            path = d / "oof_probs.pt"
            torch.save(_blob(label, model, seed=i * 2 + (model == "mlp")), path)
            written[(config, 7, model)] = str(path)
    return written


class TestConfidenceReport:
    def test_reports_all_three_variants(self, runs):
        md, _, _ = make_report.build_confidence(runs)
        for variant in ("raw", "+temp", "+mc"):
            assert f"| {variant} |" in md, f"{variant} missing from the table"

    def test_carries_mcc_and_aurc(self, runs):
        """Sec. 9.1 makes MCC primary and AURC the abstention metric."""
        md, _, _ = make_report.build_confidence(runs)
        header = next(l for l in md.splitlines() if l.startswith("| run |"))
        for column in ("MCC", "AURC", "Brier", "NLL", "ECE"):
            assert f" {column} " in header, f"{column} column missing"

    def test_temp_cannot_move_the_argmax(self, runs):
        """raw and +temp must agree on accuracy/macro F1/MCC, per Sec. 9.1."""
        md, _, _ = make_report.build_confidence(runs)
        rows = [l.split("|")[1:-1] for l in md.splitlines()
                if l.startswith("| A_seed7 | ST-GCN |")]
        by_variant = {r[2].strip(): [c.strip() for c in r[3:6]] for r in rows}
        assert by_variant["raw"] == by_variant["+temp"]

    def test_names_itself_config_d(self, runs):
        md, _, _ = make_report.build_confidence(runs)
        assert "config D" in md.splitlines()[0]

    def test_skips_mc_when_the_model_has_no_dropout(self, tmp_path, monkeypatch):
        monkeypatch.setattr(paths, "OUTPUTS_DIR", str(tmp_path), raising=False)
        d = tmp_path / "runs" / "A_seed7" / "mlp"
        d.mkdir(parents=True)
        torch.save(_blob("A_seed7", "mlp", dropout=False), d / "oof_probs.pt")
        md, _, _ = make_report.build_confidence({("A", 7, "mlp"): str(d / "oof_probs.pt")})
        assert "| +mc |" not in md
        assert "No `+mc` row" in md

    def test_returns_a_curve_per_model_run(self, runs):
        _, _, curves = make_report.build_confidence(runs)
        assert set(curves) == {(f"{c}_seed7", m)
                               for c in "ABC" for m in ("stgcn", "mlp")}


def _ablation_rows(md):
    """{(config, model): cells} -- key on both, or C/MLP shadows C/ST-GCN."""
    return {(l.split("|")[1].strip(), l.split("|")[2].strip()):
            [c.strip() for c in l.split("|")[3:-1]]
            for l in md.splitlines() if l.startswith("| ")}


class TestAblationReport:
    def test_carries_both_d_rows_and_their_footnotes(self, runs):
        md, _ = make_report.build_ablation(runs)
        assert "| D +temp † |" in md
        assert "| D +mc ‡ |" in md
        assert "identical to C's by construction" in md
        assert "NOT identical to C" in md

    def test_temp_row_repeats_config_c(self, runs):
        """+temp is monotonic -- its discriminative numbers must match C exactly."""
        md, _ = make_report.build_ablation(runs)
        rows = _ablation_rows(md)
        assert rows[("C", "ST-GCN")] == rows[("D +temp †", "ST-GCN")]

    def test_mc_row_is_computed_from_mc_prob_not_copied(self, runs):
        """+mc can move the argmax, so copying C's row would be a lie."""
        md, _ = make_report.build_ablation(runs)
        rows = _ablation_rows(md)
        assert rows[("C", "ST-GCN")] != rows[("D +mc ‡", "ST-GCN")]

    def test_no_mc_row_when_the_run_has_no_mc_estimate(self, tmp_path, monkeypatch):
        monkeypatch.setattr(paths, "OUTPUTS_DIR", str(tmp_path), raising=False)
        d = tmp_path / "runs" / "C_seed7" / "stgcn"
        d.mkdir(parents=True)
        torch.save(_blob("C_seed7", "stgcn", dropout=False), d / "oof_probs.pt")
        md, _ = make_report.build_ablation(
            {("C", 7, "stgcn"): str(d / "oof_probs.pt")})
        assert "D +mc" not in md
        assert "| D +temp † |" in md

    def test_paired_table_reports_p_and_win_counts(self, runs):
        md, _ = make_report.build_ablation(runs)
        header = next(l for l in md.splitlines() if l.startswith("| comparison |"))
        assert " p " in header
        assert "per-fold wins" in header

    def test_states_that_p_is_weak_and_wins_are_descriptive(self, runs):
        """A p-value without its caveat overstates a handful of seeds."""
        md, _ = make_report.build_ablation(runs)
        assert "not as a gate" in md
        assert "not a sign test" in md
        assert "df, critical t" in md

    def test_paired_prose_reports_the_real_seed_and_pair_counts(self, runs):
        """The caveat was once hardcoded to n=5/25 pairs and went stale."""
        md, seed_means = make_report.build_ablation(runs)
        n_seeds = len(seed_means[("A", "stgcn")])
        assert f"over the {n_seeds} seed-level means" in md
        assert f"({n_seeds - 1} df" in md
        assert "same source videos" in md
        assert "same rooms" not in md


class TestFigures:
    def test_plots_every_run_not_just_the_best(self, runs, tmp_path, monkeypatch):
        monkeypatch.setattr(paths, "OUTPUTS_DIR", str(tmp_path), raising=False)
        _, _, curves = make_report.build_confidence(runs)
        made = make_report.plot_confidence_figures(curves)
        assert len(made) == 2 * len(curves)
        assert sum("reliability" in m for m in made) == len(curves)
        assert sum("risk_coverage" in m for m in made) == len(curves)


# ------------------------------------------------------- learning curves

def _curve(model, means, pretrained=False, modes=("rooms", "windows")):
    """A data_need payload whose seeds bracket each requested mean exactly."""
    rows = []
    for mode in modes:
        for frac, mean in means.items():
            for seed, offset in zip((51, 52, 53), (-0.01, 0.0, 0.01)):
                rows.append({"mode": mode, "frac": frac, "seed": seed,
                             "epochs": 40, "n_train_mean": 2620 * frac,
                             "n_rooms_mean": 197 * frac,
                             "macro_f1": mean + offset,
                             "balanced_acc": mean, "mcc": mean - 0.2})
    return {"n_windows": 3275, "n_rooms": 197, "model": model,
            "pretrained": pretrained, "fixed_steps": False,
            "learning_curve": rows}


def _write_curves(tmp_path, monkeypatch, payloads):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", str(tmp_path))
    os.makedirs(paths.diagnostics_dir(), exist_ok=True)
    for name, payload in payloads.items():
        with open(paths.data_need_file(name), "w") as f:
            json.dump(payload, f)
    return make_report.load_data_need()


def test_curve_points_averages_over_seeds(tmp_path, monkeypatch):
    """The +/-0.01 seed offsets must cancel, and sd must come from them."""
    payload = _curve("stgcn", {0.25: 0.60, 1.00: 0.70})
    pts = make_report.curve_points(payload, "rooms")
    assert pts[0.25]["mean"] == pytest.approx(0.60)
    assert pts[1.00]["mean"] == pytest.approx(0.70)
    assert pts[0.25]["sd"] == pytest.approx(0.01)


def test_curve_points_separates_the_two_subsampling_modes(tmp_path):
    payload = _curve("stgcn", {1.00: 0.70}, modes=("rooms",))
    assert make_report.curve_points(payload, "rooms")
    assert make_report.curve_points(payload, "windows") == {}


def test_data_need_reports_b_minus_a_and_the_climbing_verdict(tmp_path, monkeypatch):
    """A curve rising far beyond seed noise must not be reported as flat."""
    found = _write_curves(tmp_path, monkeypatch, {
        "data_need.json": _curve("mlp", {0.25: .79, .50: .79, .75: .79, 1.00: .79}),
        "data_need_stgcn.json": _curve(
            "stgcn", {0.25: .60, .50: .65, .75: .68, 1.00: .70}),
        "data_need_stgcn_pretrained.json": _curve(
            "stgcn", {0.25: .65, .50: .69, .75: .71, 1.00: .73}, pretrained=True),
    })
    md = make_report.build_data_need(found)
    assert "+0.0500" in md            # B - A at 25%, i.e. .65 - .60
    assert "+0.1000" in md            # the ST-GCN's 25% -> 100% total
    assert "still rising" in md
    # The flat MLP gains exactly nothing, so it must not be called rising.
    mlp_row = [l for l in md.splitlines() if l.startswith("| MLP baseline |")][-1]
    assert "still rising" not in mlp_row


def test_data_need_verdict_ignores_a_single_noisy_step(tmp_path, monkeypatch):
    """A curve flat overall must read flat even if its last step is positive.

    Guards the bar this table is judged on: an earlier version compared the
    last step against one point's 2-df sd and flipped verdicts between the
    two subsampling panels of the same run.
    """
    found = _write_curves(tmp_path, monkeypatch, {
        "data_need.json": _curve("mlp", {0.25: .79, .50: .79, .75: .785, 1.00: .79}),
    })
    md = make_report.build_data_need(found)
    assert "still rising" not in md


def test_data_need_survives_a_missing_curve(tmp_path, monkeypatch):
    """Only the MLP curve on disk: no B-A column, no crash."""
    found = _write_curves(tmp_path, monkeypatch, {
        "data_need.json": _curve("mlp", {0.25: .79, .50: .79, .75: .79, 1.00: .79}),
    })
    md = make_report.build_data_need(found)
    assert "B − A" not in md
    assert "MLP baseline" in md


def test_data_need_report_is_written_when_no_curves_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", str(tmp_path))
    md = make_report.build_data_need({})
    assert "No curves on disk" in md


@pytest.fixture
def multi_seed_runs(tmp_path, monkeypatch):
    """A/B/C x three seeds x stgcn+mlp.

    The ensemble arm reports a paired test across seeds, so it deliberately
    stays silent on a single-seed fixture; these tests need more than one.
    """
    monkeypatch.setattr(paths, "OUTPUTS_DIR", str(tmp_path), raising=False)
    written = {}
    for i, config in enumerate(["A", "B", "C"]):
        for seed in (7, 8, 9):
            for model in ["stgcn", "mlp"]:
                label = f"{config}_seed{seed}"
                d = tmp_path / "runs" / label / model
                d.mkdir(parents=True)
                path = d / "oof_probs.pt"
                torch.save(_blob(label, model,
                                 seed=100 * seed + i * 2 + (model == "mlp")),
                           path)
                written[(config, seed, model)] = str(path)
    return written


class TestEnsembleArm:
    """The exploratory model-averaging arm (02_ablation.md).

    Not pre-registered, so the guarantees that matter are about CONTAINMENT:
    it must be visibly exploratory and it must never contaminate the headline
    that PREREGISTRATION.md Sec. 5b reserved for the ladder.
    """

    def test_section_appears_with_the_average_row(self, multi_seed_runs):
        md, _ = make_report.build_ablation(multi_seed_runs)
        assert "## Exploratory: model averaging" in md
        assert "probability average" in md

    def test_declares_itself_not_pre_registered(self, multi_seed_runs):
        md, _ = make_report.build_ablation(multi_seed_runs)
        section = md.split("## Exploratory: model averaging", 1)[1]
        assert "not pre-registered" in section.lower()
        assert "5b.8" in section

    def test_excluded_from_the_headline_comparison(self, multi_seed_runs):
        """A post-hoc arm scoring higher must not become 'the best ST-GCN rung'."""
        md, _ = make_report.build_ablation(multi_seed_runs)
        headline = md.split("## Headline comparison", 1)[1]
        headline = headline.split("## Exploratory", 1)[0]
        assert "probability average" not in headline
        assert "average" not in headline.lower()

    def test_average_is_scored_the_same_way_as_the_ladder(self, multi_seed_runs):
        """Per-fold metrics then averaged -- not pooled out-of-fold (trap 1)."""
        em = make_report.ensemble_seed_means(multi_seed_runs)
        blob = torch.load(multi_seed_runs[("A", 7, "mlp")], map_location="cpu",
                          weights_only=False)
        expected = make_report._fold_mean_macro_f1(
            np.asarray(blob["oof_prob"], dtype=np.float64),
            np.asarray(blob["oof_true"]), np.asarray(blob["fold_of"]))
        assert em["MLP baseline"][7] == pytest.approx(expected)

    def test_absent_when_a_member_is_missing(self, multi_seed_runs):
        """No stgcn runs -> no ensemble section, not a crash."""
        only_mlp = {k: v for k, v in multi_seed_runs.items() if k[2] == "mlp"}
        md, _ = make_report.build_ablation(only_mlp)
        assert "## Exploratory: model averaging" not in md

    def test_skips_members_whose_probabilities_are_nan(self, multi_seed_runs,
                                                       tmp_path):
        """A model with no MC-dropout stores NaN; it must not poison the mean."""
        d = tmp_path / "runs" / "C_seed7" / "stgcn"
        blob = torch.load(d / "oof_probs.pt", map_location="cpu",
                          weights_only=False)
        blob["mc_prob"] = np.full_like(np.asarray(blob["oof_prob"]), np.nan)
        torch.save(blob, d / "oof_probs.pt")
        em = make_report.ensemble_seed_means(multi_seed_runs)
        assert 7 not in em.get("ensemble", {})
        assert 8 in em.get("ensemble", {})


class TestRepresentationReport:
    """05_representation.md -- the exploratory 'why does the ST-GCN lose' report."""

    @pytest.fixture
    def payload(self):
        return {
            "n_windows": 3275, "n_source_videos": 197, "k_folds": 5,
            "distracted_rate": 0.198,
            "unobserved_conf_threshold": 0.35,
            "feature_groups": [
                {"name": "mean only (static posture)", "start": 0, "stop": 34,
                 "n_features": 34, "macro_f1": 0.6223, "macro_f1_sd_fold": 0.016,
                 "balanced_acc": 0.60, "mcc": 0.30},
                {"name": "std only (per-joint motion spread)", "start": 34,
                 "stop": 68, "n_features": 34, "macro_f1": 0.7796,
                 "macro_f1_sd_fold": 0.023, "balanced_acc": 0.75, "mcc": 0.57},
                {"name": "all 85 (the MLP/LR baseline features)", "start": 0,
                 "stop": 85, "n_features": 85, "macro_f1": 0.7926,
                 "macro_f1_sd_fold": 0.021, "balanced_acc": 0.77, "mcc": 0.59},
            ],
            "joint_confidence": [
                {"joint": "shoulder_L", "index": 5, "mean_confidence": 0.96,
                 "observed": True},
                {"joint": "ankle_L", "index": 15, "mean_confidence": 0.19,
                 "observed": False},
            ],
            "human_ceiling": {
                "n_overlap_windows": 514, "n_all_three_trainable": 281,
                "per_annotator": [
                    {"annotator": "annotator_1", "n": 261, "accuracy": 0.9655,
                     "macro_f1": 0.9314, "mcc": 0.86}],
                "mean_accuracy": 0.9631, "mean_macro_f1": 0.9270,
            },
        }

    def test_declares_itself_exploratory(self, payload):
        md = make_report.build_representation(payload)
        assert "Exploratory" in md.splitlines()[2]
        assert "5b" in md

    def test_reports_all_three_measurements(self, payload):
        md = make_report.build_representation(payload)
        for heading in ("Where the signal lives", "Which joints actually exist",
                        "The human ceiling"):
            assert heading in md

    def test_flags_unobserved_joints(self, payload):
        md = make_report.build_representation(payload)
        assert "ankle_L" in md
        assert "1 of 2 joints" in md

    def test_human_ceiling_carries_its_upper_bound_caveat(self, payload):
        """Quoting it as 'human accuracy' would overstate a biased subset."""
        md = make_report.build_representation(payload)
        assert "upper bound" in md
        assert "281" in md

    def test_states_the_gap_against_the_best_model(self, payload):
        seed_means = {("A", "mlp"): {7: {"macro_f1": 0.7996}}}
        md = make_report.build_representation(payload, seed_means)
        assert "0.7996" in md and "0.9270" in md
        assert "+0.1274" in md

    def test_degrades_to_a_pointer_when_not_measured(self):
        md = make_report.build_representation(None)
        assert "diagnose_representation.py" in md

    def test_refuses_to_overclaim_a_problem_ceiling(self, payload):
        """The measured limit is the modality, not the task -- say so."""
        md = make_report.build_representation(payload)
        assert "Not supported" in md

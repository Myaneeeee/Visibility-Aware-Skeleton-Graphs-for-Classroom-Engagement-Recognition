"""Config/seed/output-path plumbing for evaluate.py.

The ablation ladder (PREREGISTRATION.md Sec. 5b.3) runs four configs, each at several
seeds. Before this existed, evaluate.py had no CLI at all: SEED was hardcoded and
every run wrote the same outputs/oof_probs.pt. Running B after A silently destroyed
A's out-of-fold probabilities -- which is exactly the file config D consumes -- so
no two rows of the ablation table could exist on disk at once.
"""
import os
import pytest

import evaluate
import paths


class TestRunLabel:
    def test_encodes_both_config_and_seed(self):
        assert evaluate.run_label("A", 42) == "A_seed42"

    def test_differs_across_configs_at_the_same_seed(self):
        assert evaluate.run_label("A", 42) != evaluate.run_label("B", 42)

    def test_differs_across_seeds_within_one_config(self):
        assert evaluate.run_label("A", 1) != evaluate.run_label("A", 2)


class TestLossArm:
    """PREREGISTRATION.md Sec. 5b.8: the class-weighted arm.

    A/B/C are pre-registered on unweighted cross-entropy. The weighted arm has to
    be reachable without putting a single reported number at risk, so the whole
    contract is: 'ce' changes nothing, and anything else lands somewhere else.
    """

    def test_default_label_is_unchanged_by_the_loss_argument(self):
        """The regression that matters: every existing A_seed51-style path,
        report row and log filename must survive this feature untouched."""
        assert evaluate.run_label("A", 42) == "A_seed42"
        assert evaluate.run_label("A", 42, "ce") == "A_seed42"

    def test_weighted_run_cannot_overwrite_its_unweighted_twin(self):
        assert evaluate.run_label("C", 51, "weighted") != evaluate.run_label("C", 51)
        assert paths.oof_probs_file(evaluate.run_label("C", 51, "weighted")) != \
            paths.oof_probs_file(evaluate.run_label("C", 51))

    def test_each_loss_gets_its_own_label(self):
        labels = {evaluate.run_label("C", 51, k) for k in ("ce", "weighted", "focal")}
        assert len(labels) == 3

    def test_label_still_parses_into_config_and_seed(self):
        """make_report.discover_runs splits on '_seed', so the suffix has to sit
        on the config side or the weighted runs vanish from every table."""
        config, seed = evaluate.run_label("C", 51, "weighted").split("_seed", 1)
        assert config == "C+w" and int(seed) == 51

    def test_unknown_loss_is_refused(self):
        with pytest.raises(ValueError, match="unknown loss"):
            evaluate.run_label("C", 51, "balanced")

    def test_cli_defaults_to_the_preregistered_loss(self):
        assert evaluate.build_arg_parser().parse_args([]).loss == "ce"

    def test_cli_accepts_every_implemented_loss(self):
        import losses
        for kind in losses.LOSS_CHOICES:
            assert evaluate.build_arg_parser().parse_args(["--loss", kind]).loss == kind

    def test_cli_refuses_an_unimplemented_loss(self):
        with pytest.raises(SystemExit):
            evaluate.build_arg_parser().parse_args(["--loss", "balanced"])


class TestOutputPaths:
    def test_oof_path_is_distinct_per_run(self):
        """The clobbering bug: A and B must not share an output file."""
        assert paths.oof_probs_file("A_seed42") != paths.oof_probs_file("B_seed42")

    def test_confusion_path_is_distinct_per_run(self):
        assert (paths.cv_confusion_matrix_file("A_seed42")
                != paths.cv_confusion_matrix_file("B_seed42"))

    def test_paths_land_under_the_outputs_dir(self):
        p = paths.oof_probs_file("A_seed42")
        assert p.startswith(paths.runs_dir() + os.sep)
        assert paths.runs_dir().startswith(paths.OUTPUTS_DIR + os.sep)
        assert p.endswith("oof_probs.pt")

    def test_redirecting_outputs_dir_redirects_every_artifact(self, monkeypatch, tmp_path):
        """Regression: the bucket paths were module constants derived from
        OUTPUTS_DIR at import time, so monkeypatching OUTPUTS_DIR stopped
        redirecting them and the test suite wrote into the real outputs/."""
        monkeypatch.setattr(paths, "OUTPUTS_DIR", str(tmp_path))
        for produced in (paths.oof_probs_file("A_seed42", "mlp"),
                         paths.fold_metrics_file("A_seed42", "mlp"),
                         paths.train_loss_file("A_seed42", "mlp"),
                         paths.cv_confusion_matrix_file("A_seed42", "mlp"),
                         paths.loss_curve_file("A_seed42", "mlp"),
                         paths.report_file("01_headline.md"),
                         paths.figure_file("x.png"),
                         paths.outputs_readme_file()):
            assert produced.startswith(str(tmp_path) + os.sep), produced

    def test_run_label_appears_in_the_path(self):
        assert "A_seed42" in paths.oof_probs_file("A_seed42")

    def test_each_model_gets_its_own_artifacts(self):
        """The bug this replaced: only the ST-GCN was ever written to disk, so the
        MLP -- the highest-scoring model in the study -- left nothing behind."""
        for fn in (paths.oof_probs_file, paths.fold_metrics_file,
                   paths.train_loss_file, paths.cv_confusion_matrix_file):
            assert fn("A_seed42", "stgcn") != fn("A_seed42", "mlp")
            assert fn("A_seed42", "mlp") != fn("A_seed42", "gru")

    def test_model_defaults_to_stgcn(self):
        assert paths.oof_probs_file("A_seed42") == \
            paths.oof_probs_file("A_seed42", "stgcn")


class TestResolveConfig:
    def test_config_a_trains_from_scratch(self):
        assert evaluate.resolve_config("A")["pretrained"] is False

    def test_config_b_warm_starts_from_ntu(self):
        assert evaluate.resolve_config("B")["pretrained"] is True

    def test_config_c_adds_augmentation_on_top_of_b(self):
        """C = B + augmentation. Dropping B's warm start would make the table's
        B-to-C difference uninterpretable."""
        spec = evaluate.resolve_config("C")
        assert spec["augment"] is True
        assert spec["pretrained"] is True

    def test_configs_a_and_b_do_not_augment(self):
        assert evaluate.resolve_config("A")["augment"] is False
        assert evaluate.resolve_config("B")["augment"] is False

    def test_config_d_is_not_a_training_run(self):
        """D is post-hoc over C's saved run, so it must never resolve to a spec.

        Letting it resolve would train a model bit-identical to C's, cost a full
        GPU run, and file the result under D's name -- two rows of the ablation
        table that are the same model.
        """
        with pytest.raises(ValueError, match="D"):
            evaluate.resolve_config("D")

    def test_config_d_error_names_the_command_that_does_produce_it(self):
        """The old message said 'not implemented yet', which is now false and
        sends the reader looking for work that is already done."""
        with pytest.raises(ValueError) as excinfo:
            evaluate.resolve_config("D")
        message = str(excinfo.value)
        assert "analyze_confidence" in message
        assert "--config C" in message
        assert "not implemented" not in message.lower()

    def test_unknown_config_is_rejected(self):
        with pytest.raises(ValueError):
            evaluate.resolve_config("Z")

    def test_d_is_offered_by_the_cli_so_the_error_is_the_guiding_one(self):
        """Dropping D from `choices` would make argparse emit its generic
        'invalid choice' instead of the message that says what to run."""
        assert "D" in evaluate.build_arg_parser().parse_args(
            ["--config", "D"]).config


class TestArgParser:
    def test_defaults_to_config_a_seed_42(self):
        args = evaluate.build_arg_parser().parse_args([])
        assert args.config == "A"
        assert args.seed == 42

    def test_config_and_seed_are_settable(self):
        args = evaluate.build_arg_parser().parse_args(["--config", "B", "--seed", "7"])
        assert args.config == "B"
        assert args.seed == 7

    def test_out_prefix_defaults_to_none_so_run_label_is_used(self):
        assert evaluate.build_arg_parser().parse_args([]).out_prefix is None

    def test_out_prefix_overrides_the_derived_label(self):
        args = evaluate.build_arg_parser().parse_args(["--out-prefix", "pilot"])
        assert args.out_prefix == "pilot"

    def test_rejects_a_config_outside_the_ladder(self, capsys):
        with pytest.raises(SystemExit):
            evaluate.build_arg_parser().parse_args(["--config", "Q"])

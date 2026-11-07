"""Run-level behaviour of evaluate.py: seeding, config B's warm start, and the
guarantee that one config's outputs survive the next config's run.
"""
import os
import numpy as np
import torch

import augment
import common
import evaluate
import paths
from model import STGCNClassifier


def _cv(data, seed, backbone_path=None):
    return evaluate.run_cross_validation(
        data, k=2, epochs=1, device="cpu", model_kind="stgcn",
        seed=seed, backbone_path=backbone_path)


def _load(synthetic_dataset):
    return common.load_labeled_data(
        pt_dir=synthetic_dataset["pt_dir"],
        annotations_csv=synthetic_dataset["annotations_csv"],
    )


class TestSeeding:
    def test_same_seed_reproduces_identical_probabilities(self, synthetic_dataset):
        """Without this, 'mean +/- std over 3 seeds' measures nothing repeatable."""
        data = _load(synthetic_dataset)
        first = _cv(data, seed=7)
        second = _cv(data, seed=7)
        np.testing.assert_array_equal(first["oof_prob"], second["oof_prob"])

    def test_different_seeds_give_different_probabilities(self, synthetic_dataset):
        """The seed must actually reach the model init and batch order -- otherwise
        multi-seed runs would report a fake zero variance."""
        data = _load(synthetic_dataset)
        assert not np.array_equal(_cv(data, seed=7)["oof_prob"],
                                  _cv(data, seed=8)["oof_prob"])


class TestConfigBWarmStart:
    def _backbone_ckpt(self, tmp_path):
        """A real backbone-shaped checkpoint: a model's state_dict minus the head."""
        donor = STGCNClassifier(in_channels=3, hidden_dim=128, num_classes=2)
        sd = {k: v for k, v in donor.state_dict().items() if not k.startswith("fc.")}
        path = tmp_path / "backbone.pth"
        torch.save(sd, path)
        return str(path), sd

    def test_backbone_path_loads_the_pretrained_weights(self, tmp_path):
        path, sd = self._backbone_ckpt(tmp_path)
        model = evaluate._make_model("stgcn", 2, backbone_path=path)
        loaded = model.state_dict()
        for key, want in sd.items():
            torch.testing.assert_close(loaded[key], want)

    def test_head_stays_randomly_initialised(self, tmp_path):
        """The NTU head predicts 60 action classes; ours predicts 2. Only the
        backbone transfers."""
        path, sd = self._backbone_ckpt(tmp_path)
        model = evaluate._make_model("stgcn", 2, backbone_path=path)
        assert not any(k.startswith("fc.") for k in sd)
        assert model.fc.out_features == 2

    def test_no_backbone_path_trains_from_scratch(self, tmp_path):
        path, sd = self._backbone_ckpt(tmp_path)
        scratch = evaluate._make_model("stgcn", 2, backbone_path=None).state_dict()
        # Compare across the whole backbone, not one key: constant-initialised
        # tensors (BatchNorm weight is all-ones, bias all-zeros) match between any
        # two fresh models, so a single-key assertion can pass or fail by accident.
        differing = [k for k in sd if not torch.equal(scratch[k], sd[k])]
        assert differing, "a from-scratch model must not reproduce the donor weights"

    def test_config_b_resolves_to_the_ntu_backbone(self):
        assert evaluate.resolve_config("B")["pretrained"] is True
        assert evaluate.backbone_path_for("B") == paths.NTU_BACKBONE_FILE
        assert evaluate.backbone_path_for("A") is None


class TestSavedOutputs:
    def test_saved_file_carries_fold_assignment(self, synthetic_dataset, tmp_path,
                                                monkeypatch):
        """Config D fits temperature per fold and compares configs with a PAIRED test
        across matched folds. Both need to know which fold held out each window, and
        recovering it later means re-running the whole CV."""
        monkeypatch.setattr(paths, "OUTPUTS_DIR", str(tmp_path))
        data = _load(synthetic_dataset)
        result = _cv(data, seed=42)
        evaluate.save_run_outputs(result, "A_seed42")

        blob = torch.load(paths.oof_probs_file("A_seed42"), weights_only=False)
        assert set(blob) >= {"oof_prob", "oof_true", "window_ids", "fold_of"}
        np.testing.assert_array_equal(blob["fold_of"], result["fold_of"])
        assert (blob["fold_of"] >= 0).all(), "every window must be held out by some fold"

    def test_a_second_config_does_not_clobber_the_first(self, synthetic_dataset,
                                                        tmp_path, monkeypatch):
        """The regression this whole change exists for."""
        monkeypatch.setattr(paths, "OUTPUTS_DIR", str(tmp_path))
        data = _load(synthetic_dataset)

        result_a = _cv(data, seed=42)
        evaluate.save_run_outputs(result_a, "A_seed42")
        result_b = _cv(data, seed=42, backbone_path=None)
        evaluate.save_run_outputs(result_b, "B_seed42")

        assert os.path.exists(paths.oof_probs_file("A_seed42"))
        assert os.path.exists(paths.oof_probs_file("B_seed42"))
        still_a = torch.load(paths.oof_probs_file("A_seed42"), weights_only=False)
        np.testing.assert_array_equal(still_a["oof_prob"], result_a["oof_prob"])

    def test_run_metadata_is_recorded_in_the_file(self, synthetic_dataset, tmp_path,
                                                 monkeypatch):
        """A bare tensor file cannot say which config produced it."""
        monkeypatch.setattr(paths, "OUTPUTS_DIR", str(tmp_path))
        data = _load(synthetic_dataset)
        evaluate.save_run_outputs(_cv(data, seed=13), "B_seed13")
        blob = torch.load(paths.oof_probs_file("B_seed13"), weights_only=False)
        assert blob["run_label"] == "B_seed13"


class TestAugmentationWiring:
    def test_augmented_run_differs_from_unaugmented(self, synthetic_dataset):
        data = _load(synthetic_dataset)
        plain = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="stgcn", seed=42,
            augment=False)
        augmented = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="stgcn", seed=42,
            augment=True)
        assert not np.array_equal(plain["oof_prob"], augmented["oof_prob"])

    def test_augmented_run_is_reproducible(self, synthetic_dataset):
        data = _load(synthetic_dataset)
        kw = dict(k=2, epochs=1, device="cpu", model_kind="stgcn", seed=42,
                  augment=True)
        first = evaluate.run_cross_validation(data, **kw)
        second = evaluate.run_cross_validation(data, **kw)
        np.testing.assert_array_equal(first["oof_prob"], second["oof_prob"])

    def test_test_folds_are_never_augmented(self, synthetic_dataset, monkeypatch):
        """Augmenting a test fold would inflate the score and corrupt D's
        calibration numbers. Record every window the augmenter is handed and
        assert none of them is that fold's held-out data.

        The check is necessarily per fold, not over the whole run: cross
        validation holds every window out exactly once, so every window is also
        a training window in some other fold. Pooling the augmenter's inputs
        across folds would therefore flag leakage for any correct
        implementation. Folds are told apart by augmenter identity -- the run
        builds one SkeletonAugment per fold, in fold order, so that multi-seed
        runs reproduce."""
        data = _load(synthetic_dataset)
        # (augmenter, window) in call order. Holding the augmenter itself keeps
        # it alive for the whole run: keying on id() instead would let fold 0's
        # augmenter be collected and fold 1's be allocated at the same address,
        # silently merging two folds into one.
        records = []
        real_call = augment.SkeletonAugment.__call__

        def spy(self, window):
            records.append((self, window.clone()))
            return real_call(self, window)

        monkeypatch.setattr(augment.SkeletonAugment, "__call__", spy)
        result = evaluate.run_cross_validation(
            data, k=2, epochs=1, device="cpu", model_kind="stgcn", seed=42,
            augment=True)

        assert records, "augmentation never ran"
        augmenters = []
        for aug, _ in records:
            if not any(aug is seen for seen in augmenters):
                augmenters.append(aug)

        folds = sorted(set(result["fold_of"]))
        assert len(augmenters) == len(folds), (
            "expected one augmenter per fold; the fold<->augmenter mapping this "
            "test relies on no longer holds")

        tensors = torch.stack(data["tensors"])
        for fold, aug in zip(folds, augmenters):
            fed = [w for a, w in records if a is aug]
            for w in tensors[result["fold_of"] == fold]:
                for s in fed:
                    assert not torch.equal(s, w), (
                        f"fold {fold} augmented one of its own held-out windows")

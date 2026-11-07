"""Class-weighted and focal objectives for the cross-validated protocol.

PREREGISTRATION.md Sec. 5b.8: the imbalance remedy existed only in
steps/train.py, the quarantined single-split script, so every reported number
came from plain unweighted cross-entropy on a 16.8%-minority set. That made
"more data will not help" a claim about undirected labeling only. These are the
objectives that let evaluate.py test the other half.

The weights must come from the TRAINING fold alone. Deriving them from the whole
labeled set would leak the test fold's class prior into the objective -- a mild
leak, but this project's whole protocol is built on cross-video honesty.
"""
import pytest
import torch
import torch.nn.functional as F

import losses


class TestClassWeights:
    def test_balanced_labels_give_uniform_weights(self):
        w = losses.class_weights([0, 0, 1, 1], num_classes=2)
        assert torch.allclose(w, torch.ones(2), atol=1e-5)

    def test_minority_class_gets_the_larger_weight(self):
        w = losses.class_weights([0] * 90 + [1] * 10, num_classes=2)
        assert w[1] > w[0]

    def test_weight_ratio_is_inverse_frequency(self):
        """90/10 split -> the minority is worth 9x the majority per sample."""
        w = losses.class_weights([0] * 90 + [1] * 10, num_classes=2)
        assert float(w[1] / w[0]) == pytest.approx(9.0, rel=1e-3)

    def test_weights_are_normalized_to_num_classes(self):
        """Keeps the loss on the same scale as unweighted CE, so the learning
        rate and epoch budget carry over unchanged from configs A/B/C."""
        w = losses.class_weights([0] * 90 + [1] * 10, num_classes=2)
        assert float(w.sum()) == pytest.approx(2.0, rel=1e-5)

    def test_absent_class_does_not_divide_by_zero(self):
        w = losses.class_weights([0, 0, 0], num_classes=2)
        assert torch.isfinite(w).all()

    def test_accepts_a_tensor_of_labels(self):
        w = losses.class_weights(torch.tensor([0, 0, 1, 1]), num_classes=2)
        assert torch.allclose(w, torch.ones(2), atol=1e-5)

    def test_reads_only_the_labels_it_is_given(self):
        """The train-fold seam: a fold's weights must not see held-out labels."""
        train_only = losses.class_weights([0] * 90 + [1] * 10, num_classes=2)
        whole_set = losses.class_weights([0] * 90 + [1] * 90, num_classes=2)
        assert not torch.allclose(train_only, whole_set)


class TestFocalLoss:
    def test_gamma_zero_without_alpha_is_cross_entropy(self):
        logits = torch.tensor([[2.0, -1.0], [0.5, 0.4]])
        target = torch.tensor([0, 1])
        fl = losses.FocalLoss(gamma=0.0)
        assert float(fl(logits, target)) == pytest.approx(
            float(F.cross_entropy(logits, target)), rel=1e-5)

    def test_gamma_downweights_easy_examples(self):
        """An already-correct, confident example contributes less under gamma>0."""
        # Not a wider margin than this: cross-entropy underflows to exactly 0.0
        # in float32 well before the logit gap reaches 20, and "0 < 0" would fail
        # for a reason that has nothing to do with gamma.
        easy = (torch.tensor([[3.0, -3.0]]), torch.tensor([0]))
        ce = float(F.cross_entropy(*easy))
        focal = float(losses.FocalLoss(gamma=2.0)(*easy))
        assert focal < ce

    def test_alpha_scales_the_minority_class(self):
        logits = torch.tensor([[0.5, 0.4]])
        minority = torch.tensor([1])
        plain = float(losses.FocalLoss(gamma=2.0)(logits, minority))
        weighted = float(losses.FocalLoss(
            alpha=torch.tensor([0.5, 1.5]), gamma=2.0)(logits, minority))
        assert weighted == pytest.approx(plain * 1.5, rel=1e-5)


class TestMakeLossFn:
    def test_ce_is_plain_cross_entropy_regardless_of_imbalance(self):
        """Config A/B/C's objective. Must not change when this module lands."""
        fn = losses.make_loss_fn("ce", [0] * 90 + [1] * 10, num_classes=2)
        logits = torch.tensor([[2.0, -1.0], [0.5, 0.4]])
        target = torch.tensor([0, 1])
        assert float(fn(logits, target)) == pytest.approx(
            float(F.cross_entropy(logits, target)), rel=1e-6)

    def test_weighted_penalizes_a_minority_error_more_than_ce(self):
        fn = losses.make_loss_fn("weighted", [0] * 90 + [1] * 10, num_classes=2)
        logits = torch.tensor([[2.0, -1.0]])
        minority_truth = torch.tensor([1])
        assert float(fn(logits, minority_truth)) > float(
            F.cross_entropy(logits, minority_truth))

    def test_weighted_and_ce_agree_on_balanced_labels(self):
        fn = losses.make_loss_fn("weighted", [0, 0, 1, 1], num_classes=2)
        logits = torch.tensor([[2.0, -1.0], [0.5, 0.4]])
        target = torch.tensor([0, 1])
        assert float(fn(logits, target)) == pytest.approx(
            float(F.cross_entropy(logits, target)), rel=1e-5)

    def test_focal_carries_the_class_alpha(self):
        fn = losses.make_loss_fn("focal", [0] * 90 + [1] * 10, num_classes=2)
        logits = torch.tensor([[2.0, -1.0]])
        minority_truth = torch.tensor([1])
        unweighted_focal = losses.FocalLoss(gamma=losses.FOCAL_GAMMA)
        assert float(fn(logits, minority_truth)) > float(
            unweighted_focal(logits, minority_truth))

    def test_unknown_kind_names_the_valid_choices(self):
        with pytest.raises(ValueError, match="weighted"):
            losses.make_loss_fn("balanced", [0, 1], num_classes=2)

    def test_every_advertised_choice_builds(self):
        for kind in losses.LOSS_CHOICES:
            fn = losses.make_loss_fn(kind, [0, 0, 1], num_classes=2)
            out = fn(torch.tensor([[0.5, 0.4]]), torch.tensor([1]))
            assert torch.isfinite(out)

    def test_loss_stays_differentiable(self):
        for kind in losses.LOSS_CHOICES:
            fn = losses.make_loss_fn(kind, [0, 0, 1], num_classes=2)
            logits = torch.tensor([[0.5, 0.4]], requires_grad=True)
            fn(logits, torch.tensor([1])).backward()
            assert logits.grad is not None and torch.isfinite(logits.grad).all()

    def test_weights_land_on_the_requested_device(self):
        fn = losses.make_loss_fn("weighted", [0, 0, 1], num_classes=2,
                                 device="cpu")
        out = fn(torch.tensor([[0.5, 0.4]]), torch.tensor([1]))
        assert out.device.type == "cpu"

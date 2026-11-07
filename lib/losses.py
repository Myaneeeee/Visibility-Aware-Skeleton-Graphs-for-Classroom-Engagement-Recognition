"""Training objectives for the cross-validated ablation.

Configs A/B/C are pre-registered on plain unweighted cross-entropy
(`PREREGISTRATION.md` §5b.3) and `make_loss_fn("ce", ...)` reproduces it exactly,
so nothing here changes a reported number unless a run explicitly asks for it.

What it adds is a test of class balance: the learning curves keep the class
ratio fixed, so "more data will not help" holds only at the dataset's minority
share and says nothing about a better balance. These objectives let the
cross-validated protocol test that.

`alpha` is applied per sample and the batch is averaged over N -- deliberately
NOT torch's own `F.cross_entropy(weight=...)`, which divides by the summed
weights of the batch instead. That renormalization makes a batch of one class
identical to unweighted, and it would make the "weighted" and "focal" rungs
disagree on scale for no stated reason.
"""
import numpy as np
import torch
import torch.nn.functional as F

LOSS_CHOICES = ("ce", "weighted", "focal")

FOCAL_GAMMA = 2.0


def class_weights(labels, num_classes):
    """Inverse-frequency weights, normalized to sum to `num_classes`.

    Pass the TRAINING fold's labels only. Deriving these from the whole labeled
    set would leak the held-out fold's class prior into the objective.

    The normalization keeps a weighted loss on the same scale as an unweighted
    one (uniform labels give all-ones), so configs sharing this module also
    share a learning rate and epoch budget -- §9.1 requires an identical
    training budget across rungs.
    """
    labels = np.asarray(labels.tolist() if torch.is_tensor(labels) else labels)
    counts = np.bincount(labels.astype(int), minlength=num_classes)[:num_classes]
    # A fold missing a class entirely would divide by zero. Clamping to 1 gives
    # that class the weight of a single-sample class instead of an infinity that
    # would silently poison the whole run.
    w = labels.size / np.maximum(counts, 1).astype(np.float64)
    w = w / w.sum() * num_classes
    return torch.tensor(w, dtype=torch.float32)


class FocalLoss(torch.nn.Module):
    """Focal loss with optional per-class alpha (Lin et al., 2017).

    gamma=0 and alpha=None reduces exactly to cross-entropy, which is what makes
    it safe to route the unweighted rungs through the same code path.
    """

    def __init__(self, alpha=None, gamma=FOCAL_GAMMA):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha) if isinstance(
            alpha, torch.Tensor) else setattr(self, "alpha", alpha)

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none")
        loss = ce
        if self.gamma:
            pt = torch.exp(-ce)
            loss = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            loss = loss * self.alpha.to(inputs.device)[targets]
        return loss.mean()


def make_loss_fn(kind, train_labels, num_classes, device=None):
    """Build the objective for one fold from that fold's TRAINING labels.

    Returns a plain callable(logits, targets) -> scalar tensor.
    """
    if kind not in LOSS_CHOICES:
        raise ValueError(
            f"unknown loss {kind!r}; expected one of {list(LOSS_CHOICES)}")
    if kind == "ce":
        return F.cross_entropy
    alpha = class_weights(train_labels, num_classes)
    if device is not None:
        alpha = alpha.to(device)
    gamma = FOCAL_GAMMA if kind == "focal" else 0.0
    return FocalLoss(alpha=alpha, gamma=gamma)

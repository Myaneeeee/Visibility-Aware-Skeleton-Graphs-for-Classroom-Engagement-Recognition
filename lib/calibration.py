"""Confidence calibration and selective prediction for config D.

Everything here is post-hoc over saved out-of-fold predictions. Nothing retrains,
so config D's classifier stays bit-identical to config C's.

A note on working from probabilities rather than logits: evaluate.py saves
softmax outputs. Softmax is shift-invariant, so log(p) is a valid logit vector --
recovering it and rescaling gives the same result as scaling the original logits
would have, EXCEPT where float32 has already saturated the probability. A logit
gap beyond ~87 stores as exactly 1.0/0.0, and no amount of care downstream can
recover what rounding removed; the _EPS clamp then caps the recovered gap at
27.6 nats. fit_temperature warns when it sees saturated inputs. Saving logits
instead of probabilities would remove the caveat entirely, and is the right fix
if saturation ever turns out to be common in real runs.
"""
import numpy as np
import torch
import torch.nn.functional as F

_EPS = 1e-12


def _logits_from_prob(prob):
    return torch.log(torch.clamp(torch.as_tensor(prob, dtype=torch.float64),
                                 min=_EPS))


def apply_temperature(prob, T):
    """Divide the recovered logits by T and re-softmax. Returns (N, K) float64.

    T > 1 lowers stated confidence, T < 1 raises it, T = 1 is a no-op. Because
    the transform is monotonic it can never change which class wins.
    """
    scaled = _logits_from_prob(prob) / float(T)
    return torch.softmax(scaled, dim=1).numpy()


def fit_temperature(prob, y, max_iter=200):
    """Find the T minimizing NLL on (prob, y). Returns a positive float.

    Optimizes log(T) so T stays positive without a constrained solver.

    Saturation caveat: evaluate.py saves float32 softmax output, so a logit gap
    beyond ~87 rounds to exactly 1.0/0.0 and the information needed to recover
    the true gap is gone before this function sees it. _EPS then caps the
    recovered gap at -log(1e-12) = 27.6 nats. On a FULLY saturated input the
    cross-entropy gradient underflows and this returns T = 1.0 -- "perfectly
    calibrated" for the most overconfident input possible. Saturation is
    therefore detected and reported rather than silently fitted.
    """
    prob = np.asarray(prob, dtype=np.float64)
    saturated = int((prob >= 1.0).sum() + (prob <= 0.0).sum())
    if saturated:
        print(f"  WARNING: {saturated} of {prob.size} saturated probabilities "
              f"(exactly 0 or 1) in this temperature fit. float32 already "
              f"discarded their true confidence; T is a lower bound.")

    logits = _logits_from_prob(prob)
    target = torch.as_tensor(np.asarray(y), dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / torch.exp(log_t), target)
        loss.backward()
        return loss

    opt.step(closure)
    T = float(torch.exp(log_t).detach())
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError(
            f"temperature fit did not converge to a usable value (T={T!r}). "
            "Dividing by it would push NaN through every config D metric "
            "silently. Check the input probabilities for NaN or all-one-class y.")
    return T


def leave_one_fold_out_temperature(prob, y, fold_of):
    """Calibrate each fold with a temperature fit on the OTHER folds.

    Returns (calibrated_prob, {fold: T}).

    Why not the textbook inner-validation split: carving a validation set out of
    each training fold would shrink config D's training data relative to config
    C's, so D's accuracy would move for a reason unrelated to confidence and the
    ladder's one-thing-per-rung logic would break. Fitting on the other folds
    keeps D's classifier bit-identical to C's and needs no retraining.

    Why not one global fit on all out-of-fold predictions: that is fitting on the
    evaluation set.

    The assumption -- that miscalibration is comparable across folds -- is
    reported rather than assumed: the returned per-fold temperatures are a
    diagnostic, and a wide spread is itself a finding about video-to-video variation.
    """
    prob = np.asarray(prob, dtype=np.float64)
    y = np.asarray(y)
    fold_of = np.asarray(fold_of)

    if (fold_of < 0).any():
        raise ValueError(
            f"{int((fold_of < 0).sum())} window(s) have fold_of == -1, meaning no "
            "fold held them out. They have no out-of-fold prediction to calibrate.")

    calibrated = np.zeros_like(prob)
    temperatures = {}
    for fold in sorted(set(fold_of.tolist())):
        held_out = fold_of == fold
        others = ~held_out
        T = fit_temperature(prob[others], y[others])
        temperatures[int(fold)] = T
        calibrated[held_out] = apply_temperature(prob[held_out], T)
    return calibrated, temperatures


def brier_score(prob, y):
    """Multiclass Brier score: mean squared error against the one-hot label.

    A proper scoring rule with no binning, so it is stable at n ~ 679 where ECE
    is not. Reported as a headline number alongside NLL.
    """
    prob = np.asarray(prob, dtype=np.float64)
    onehot = np.zeros_like(prob)
    onehot[np.arange(len(prob)), np.asarray(y)] = 1.0
    return float(((prob - onehot) ** 2).sum(axis=1).mean())


def nll(prob, y):
    """Mean negative log likelihood of the true class.

    Punishes confident mistakes far harder than Brier does, which is the property
    that matters for an abstention system.
    """
    prob = np.asarray(prob, dtype=np.float64)
    picked = prob[np.arange(len(prob)), np.asarray(y)]
    return float(-np.log(np.clip(picked, _EPS, None)).mean())


def _equal_mass_bins(n, n_bins):
    return np.array_split(np.arange(n), n_bins)


def reliability_curve(prob, y, n_bins=15):
    """Equal-mass reliability data. Returns (mean_conf, mean_acc, counts).

    Equal MASS, not equal width: at this n, equal-width bins leave some nearly
    empty and the resulting curve is dominated by noise in those bins.
    """
    prob = np.asarray(prob, dtype=np.float64)
    conf = prob.max(axis=1)
    correct = (prob.argmax(axis=1) == np.asarray(y)).astype(np.float64)
    order = np.argsort(conf)
    conf, correct = conf[order], correct[order]

    mean_conf, mean_acc, counts = [], [], []
    for idx in _equal_mass_bins(len(conf), n_bins):
        if len(idx) == 0:
            continue
        mean_conf.append(conf[idx].mean())
        mean_acc.append(correct[idx].mean())
        counts.append(len(idx))
    return np.array(mean_conf), np.array(mean_acc), np.array(counts)


def ece_equal_mass(prob, y, n_bins=15):
    """Expected calibration error over equal-mass bins.

    SECONDARY metric only. It is binning-dependent and biased at small n, so it
    is reported for comparability with prior work while Brier and NLL carry the
    claims. The binning scheme must be stated wherever this number appears.
    """
    mean_conf, mean_acc, counts = reliability_curve(prob, y, n_bins=n_bins)
    weights = counts / counts.sum()
    return float((weights * np.abs(mean_acc - mean_conf)).sum())


def risk_coverage_curve(prob, y):
    """Error rate as a function of how much of the data the model answers on.

    Sorts most-confident-first and sweeps every coverage level. Returns
    (coverage, risk), both length N.

    The whole curve is reported rather than accuracy at two hand-picked coverage
    levels, because quoting a couple of thresholds invites picking the pair that
    flatters the model.
    """
    prob = np.asarray(prob, dtype=np.float64)
    conf = prob.max(axis=1)
    wrong = (prob.argmax(axis=1) != np.asarray(y)).astype(np.float64)
    order = np.argsort(-conf, kind="stable")
    wrong = wrong[order]

    n = len(wrong)
    answered = np.arange(1, n + 1)
    coverage = answered / n
    risk = np.cumsum(wrong) / answered
    return coverage, risk


def aurc(prob, y):
    """Area under the risk-coverage curve. Lower is better.

    One threshold-free number summarizing how well the model's confidence ranks
    its own mistakes.
    """
    _, risk = risk_coverage_curve(prob, y)
    return float(risk.mean())

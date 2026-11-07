"""Do any other combination rules beat the simple probability average?

Everything here runs over the SAVED out-of-fold predictions -- nothing is retrained.
Any rule with free parameters is fit LEAVE-ONE-FOLD-OUT: the fold being scored never
contributes to fitting the rule that scores it. Fitting on all OOF predictions at once
would be fitting on the evaluation set, which is the error this file exists to avoid.
"""
import os, sys
import numpy as np, torch
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata, ttest_rel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "steps"))
import _bootstrap  # noqa: F401
import metrics as M, paths

SEEDS = range(51, 61)
EPS = 1e-12


def load(seed):
    out = {}
    for name, (cfg, model, key) in {"mlp": ("A", "mlp", "oof_prob"),
                                    "gru": ("A", "gru", "oof_prob"),
                                    "stgcn": ("C", "stgcn", "mc_prob")}.items():
        b = torch.load(os.path.join(paths.runs_dir(), f"{cfg}_seed{seed}", model, "oof_probs.pt"),
                       map_location="cpu", weights_only=False)
        out[name] = np.asarray(b[key], dtype=np.float64)
        true, fold = np.asarray(b["oof_true"]), np.asarray(b["fold_of"])
    return out, true, fold


def macro_f1(prob, true, fold):
    """Per-fold macro-F1 averaged -- the same convention as every other report."""
    pred = prob.argmax(1)
    return float(np.mean([M.compute_metrics(true[fold == f], pred[fold == f], 2)["macro_f1"]
                          for f in sorted(set(fold.tolist())) if f >= 0]))


def norm(p):
    return p / p.sum(1, keepdims=True)


# ---------------------------------------------------------------- combination rules
def mean_prob(ps, **kw):
    return np.mean(ps, axis=0)


def geometric(ps, **kw):
    """Log-opinion pool: the product of probabilities. Penalises confident disagreement."""
    return norm(np.exp(np.mean([np.log(p + EPS) for p in ps], axis=0)))


def rank_avg(ps, **kw):
    """Average the RANK of P(Distracted); scale-free, ignores miscalibration."""
    r = np.mean([rankdata(p[:, 1]) / len(p) for p in ps], axis=0)
    return np.stack([1 - r, r], axis=1)


def max_conf(ps, **kw):
    """Defer to whichever member is more certain on this window."""
    conf = np.stack([p.max(1) for p in ps])
    pick = conf.argmax(0)
    return np.stack(ps)[pick, np.arange(ps[0].shape[0])]


def weighted(ps, w=None, **kw):
    return w * ps[0] + (1 - w) * ps[1]


def fit_weight(ps, true, fold, tr):
    """Best w for member 0 on the training folds only."""
    def loss(w):
        p = norm(np.clip(w * ps[0][tr] + (1 - w) * ps[1][tr], EPS, 1))
        return -np.mean(np.log(p[np.arange(tr.sum()), true[tr]] + EPS))
    return float(minimize_scalar(loss, bounds=(0, 1), method="bounded").x)


def fit_temp_pair(ps, true, fold, tr):
    """One temperature per member, fitted jointly by NLL on the training folds."""
    def sharpen(p, T):
        return norm(np.clip(p, EPS, 1) ** (1.0 / T))
    best, T = None, (1.0, 1.0)
    for t0 in np.linspace(0.5, 3.0, 26):
        for t1 in np.linspace(0.5, 3.0, 26):
            p = 0.5 * sharpen(ps[0][tr], t0) + 0.5 * sharpen(ps[1][tr], t1)
            nll = -np.mean(np.log(p[np.arange(tr.sum()), true[tr]] + EPS))
            if best is None or nll < best:
                best, T = nll, (t0, t1)
    return T


def fit_stack(ps, true, fold, tr):
    """Logistic regression on the members' P(Distracted) -- classic stacking."""
    from sklearn.linear_model import LogisticRegression
    X = np.column_stack([p[tr, 1] for p in ps])
    return LogisticRegression(max_iter=1000).fit(X, true[tr])


def fit_thresh(ps, true, fold, tr):
    """Best decision threshold on the averaged probability, for macro-F1."""
    p = np.mean(ps, axis=0)[tr, 1]
    grid = np.linspace(0.05, 0.95, 91)
    scores = [M.compute_metrics(true[tr], (p >= t).astype(int), 2)["macro_f1"] for t in grid]
    return float(grid[int(np.argmax(scores))])


def loo_fold(ps, true, fold, fit, apply):
    """Fit the rule on the other folds, apply it to the held-out fold. No leakage."""
    out = np.zeros_like(ps[0])
    for f in sorted(set(fold.tolist())):
        if f < 0:
            continue
        te, tr = fold == f, (fold != f) & (fold >= 0)
        out[te] = apply([p[te] for p in ps], fit(ps, true, fold, tr))
    return out


# ---------------------------------------------------------------- run
PAIR = ["mlp", "stgcn"]
rows, per_seed = {}, {}
for seed in SEEDS:
    P, true, fold = load(seed)
    ps = [P[k] for k in PAIR]
    res = {
        "MLP alone": P["mlp"],
        "ST-GCN D+mc alone": P["stgcn"],
        "probability average": mean_prob(ps),
        "geometric mean (log pool)": geometric(ps),
        "rank average": rank_avg(ps),
        "max-confidence switch": max_conf(ps),
        "weighted avg (w fit LOFO)": loo_fold(ps, true, fold, fit_weight,
                                              lambda q, w: norm(w * q[0] + (1 - w) * q[1])),
        "per-member temperature (LOFO)": loo_fold(
            ps, true, fold, fit_temp_pair,
            lambda q, T: 0.5 * norm(np.clip(q[0], EPS, 1) ** (1 / T[0]))
                       + 0.5 * norm(np.clip(q[1], EPS, 1) ** (1 / T[1]))),
        "stacking: LR on probs (LOFO)": loo_fold(
            ps, true, fold, fit_stack,
            lambda q, m: m.predict_proba(np.column_stack([x[:, 1] for x in q]))),
        "tuned threshold (LOFO)": loo_fold(
            ps, true, fold, fit_thresh,
            lambda q, t: (lambda pd: np.stack([1 - pd, pd], 1))(
                np.where(np.mean(q, 0)[:, 1] >= t, 0.5 + np.mean(q, 0)[:, 1] / 2,
                         np.mean(q, 0)[:, 1] / (2 * t + EPS)))),
        "3-way avg (+GRU)": mean_prob([P["mlp"], P["stgcn"], P["gru"]]),
    }
    for k, p in res.items():
        rows.setdefault(k, []).append(macro_f1(p, true, fold))

base = np.array(rows["probability average"])
print(f"Pair: MLP + ST-GCN D+mc. {len(list(SEEDS))} seeds, per-fold macro-F1, "
      f"GroupKFold-5 by source video.\n")
hdr = f"{'combination rule':<32}{'macro-F1':>10}{'sd(seed)':>10}{'vs avg':>9}{'p':>8}{'wins':>7}"
print(hdr); print("-" * len(hdr))
for k, v in sorted(rows.items(), key=lambda kv: -np.mean(kv[1])):
    v = np.array(v)
    d = v - base
    if k == "probability average":
        extra = f"{'--':>9}{'--':>8}{'--':>7}"
    else:
        p = ttest_rel(v, base).pvalue if np.std(d) > 0 else float("nan")
        extra = f"{d.mean():>+9.4f}{p:>8.3f}{(d > 0).sum():>5}/10"
    print(f"{k:<32}{v.mean():>10.4f}{v.std(ddof=1):>10.4f}{extra}")

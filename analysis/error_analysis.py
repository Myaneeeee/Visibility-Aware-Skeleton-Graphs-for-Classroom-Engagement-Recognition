"""Where does each model win? MLP vs an ST-GCN +mc, over the saved OOF predictions.

Usage:  .venv/bin/python analysis/error_analysis.py [stgcn config, default E1]
"""
import os, sys
import numpy as np, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "steps"))
import _bootstrap  # noqa: F401
import common, paths

SEEDS = list(range(51, 61))
STGCN = sys.argv[1] if len(sys.argv) > 1 else "E1"
MEM = {"mlp": ("A", "mlp", "oof_prob"), "stgcn": (STGCN, "stgcn", "mc_prob")}
print(f"MLP vs ST-GCN {STGCN}+mc")

# ---- predictions across seeds
pred = {k: [] for k in MEM}
true = fold_of = None
for s in SEEDS:
    for k, (cfg, model, key) in MEM.items():
        b = torch.load(os.path.join(paths.runs_dir(), f"{cfg}_seed{s}", model, "oof_probs.pt"),
                       map_location="cpu", weights_only=False)
        pred[k].append(np.asarray(b[key]).argmax(1))
        true = np.asarray(b["oof_true"]); fold_of = np.asarray(b["fold_of"])
mlp = np.array(pred["mlp"]); stg = np.array(pred["stgcn"])       # (10, N)
mlp_ok = mlp == true; stg_ok = stg == true

n_mlp_ok = mlp_ok.sum(0); n_stg_ok = stg_ok.sum(0)               # 0..10 per window
STRICT = 8
stg_wins = (n_stg_ok >= STRICT) & (n_mlp_ok <= 10 - STRICT)
mlp_wins = (n_mlp_ok >= STRICT) & (n_stg_ok <= 10 - STRICT)
both_ok  = (n_mlp_ok >= STRICT) & (n_stg_ok >= STRICT)
both_bad = (n_mlp_ok <= 10 - STRICT) & (n_stg_ok <= 10 - STRICT)

# ---- window descriptors, in the same order the loader produces
d = common.load_labeled_data(paths.SAV_WINDOWS_DIR, paths.ANNOTATIONS_FILE)
X = torch.stack(d["tensors"])                                     # (N, 3, 90, 17)
y = np.array(d["labels"])
assert len(y) == len(true) and (y == true).all(), "order mismatch"

xy, conf = X[:, :2], X[:, 2]                                      # (N,2,90,17), (N,90,17)
motion = xy.std(dim=2).mean(dim=(1, 2)).numpy()                   # mean per-joint temporal std
speed = (xy[:, :, 1:] - xy[:, :, :-1]).pow(2).sum(1).sqrt().mean((1, 2)).numpy()
mean_conf = conf.mean((1, 2)).numpy()
LOWER = [13, 14, 15, 16]                                          # knees, ankles
UPPER = [5, 6, 7, 8, 9, 10]                                       # shoulders, elbows, wrists
low_conf = conf[:, :, LOWER].mean((1, 2)).numpy()
up_conf = conf[:, :, UPPER].mean((1, 2)).numpy()

groups = [("ST-GCN wins", stg_wins), ("MLP wins", mlp_wins),
          ("both right", both_ok), ("both wrong", both_bad)]

print(f"N = {len(y)} windows, {y.mean():.1%} Distracted, consistency threshold {STRICT}/10 seeds\n")
hdr = f"{'group':<14}{'n':>6}{'%Distr':>9}{'motion':>9}{'speed':>9}{'conf':>8}{'lower':>8}{'upper':>8}"
print(hdr); print("-" * len(hdr))
for name, m in groups:
    print(f"{name:<14}{m.sum():>6}{y[m].mean():>8.1%}{motion[m].mean():>9.4f}"
          f"{speed[m].mean():>9.4f}{mean_conf[m].mean():>8.3f}{low_conf[m].mean():>8.3f}"
          f"{up_conf[m].mean():>8.3f}")
print(f"{'ALL':<14}{len(y):>6}{y.mean():>8.1%}{motion.mean():>9.4f}{speed.mean():>9.4f}"
      f"{mean_conf.mean():>8.3f}{low_conf.mean():>8.3f}{up_conf.mean():>8.3f}")

# ---- what each model does on its disjoint set: which way does it err?
print("\nOn the disjoint windows, the losing model's prediction:")
for name, m, loser in (("ST-GCN wins", stg_wins, mlp), ("MLP wins", mlp_wins, stg)):
    if not m.any(): continue
    maj = (loser[:, m] == 0).mean()          # predicted Focused (majority)
    print(f"  {name:<12} loser predicts Focused {maj:.1%} of the time "
          f"(true class is Distracted for {y[m].mean():.1%})")

# ---- motion tercile breakdown
print("\nWin rate by motion tercile (low / mid / high per-joint temporal std):")
q = np.quantile(motion, [1/3, 2/3]); terc = np.digitize(motion, q)
print(f"{'tercile':<10}{'n':>6}{'%Distr':>9}{'ST-GCN wins':>13}{'MLP wins':>11}")
for t, lab in enumerate(["low", "mid", "high"]):
    m = terc == t
    print(f"{lab:<10}{m.sum():>6}{y[m].mean():>8.1%}{stg_wins[m].mean():>12.1%}{mlp_wins[m].mean():>11.1%}")

# ---- per-class
print("\nPer true class:")
for c, nm in enumerate(common.CLASS_NAMES):
    m = y == c
    print(f"  {nm:<11} n={m.sum():<5} ST-GCN wins {stg_wins[m].mean():>5.1%} | "
          f"MLP wins {mlp_wins[m].mean():>5.1%} | both right {both_ok[m].mean():>5.1%} | "
          f"both wrong {both_bad[m].mean():>5.1%}")

# ---------------------------------------------------------------- follow-up
from scipy import stats
print("\n" + "=" * 72)
print("Does ANYTHING separate the two disjoint sets? (ST-GCN-wins vs MLP-wins)")
print("=" * 72)
frac_conf = (conf > 0.5).float().mean((1, 2)).numpy()
descriptors = {"motion (per-joint std)": motion, "speed": speed,
               "mean joint conf": mean_conf, "lower-body conf": low_conf,
               "upper-body conf": up_conf, "frac joints conf>0.5": frac_conf}
print(f"{'descriptor':<24}{'ST-GCN wins':>13}{'MLP wins':>11}{'Mann-Whitney p':>16}")
print("-" * 64)
for nm, v in descriptors.items():
    p = stats.mannwhitneyu(v[stg_wins], v[mlp_wins]).pvalue
    print(f"{nm:<24}{v[stg_wins].mean():>13.4f}{v[mlp_wins].mean():>11.4f}{p:>16.3f}")
p = stats.fisher_exact([[ (y[stg_wins]==1).sum(), (y[stg_wins]==0).sum()],
                        [ (y[mlp_wins]==1).sum(), (y[mlp_wins]==0).sum()]]).pvalue
print(f"{'true class (Distracted)':<24}{y[stg_wins].mean():>13.4f}{y[mlp_wins].mean():>11.4f}{p:>16.3f}  (Fisher)")

# ---- are the disjoint windows near the decision boundary?
pm = np.mean([np.asarray(torch.load(os.path.join(paths.runs_dir(), f"A_seed{s}", "mlp",
              "oof_probs.pt"), map_location="cpu", weights_only=False)["oof_prob"])
              for s in SEEDS], axis=0)
ps = np.mean([np.asarray(torch.load(os.path.join(paths.runs_dir(), f"C_seed{s}", "stgcn",
              "oof_probs.pt"), map_location="cpu", weights_only=False)["mc_prob"])
              for s in SEEDS], axis=0)
conf_m, conf_s = pm.max(1), ps.max(1)
print("\nSeed-averaged predicted confidence (1.0 = certain, 0.5 = coin flip):")
print(f"{'group':<14}{'MLP conf':>10}{'ST-GCN conf':>13}{'|p_mlp - p_stgcn|':>19}")
for name, m in groups:
    print(f"{name:<14}{conf_m[m].mean():>10.3f}{conf_s[m].mean():>13.3f}"
          f"{np.abs(pm[m,1]-ps[m,1]).mean():>19.3f}")
print(f"{'ALL':<14}{conf_m.mean():>10.3f}{conf_s.mean():>13.3f}{np.abs(pm[:,1]-ps[:,1]).mean():>19.3f}")
print(f"\nCorrelation of the two models' P(Distracted): r = {np.corrcoef(pm[:,1], ps[:,1])[0,1]:.3f}")

"""
Computes the difference between GIN+CAL and its control seed by seed, with a 95% interval. It does no training.

Needs: the by-seed result files in gin_cal/results.
"""
import os
import sys
 
import numpy as np
import pandas as pd
 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "core"))                      # shared modules (compare_vanilla_cal*.py)
HERE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "results"))   # all CSV results of gin_cal live here
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)
 
 
def boot_ci(v, n=5000, seed=0):
    """Bootstrap 95% confidence interval of the mean of v (resampling over seeds)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (n, len(v)))
    return np.percentile(v[idx].mean(axis=1), [2.5, 97.5])
 
 
def per_seed(df, variant, metric):
    """Mean cross-environment value of a metric for each seed."""
    d = df[(df.variant == variant) & (df.experiment == "cross")]
    return d.groupby("seed")[metric].mean()
 
 
def main():
    """Print the paired differences between GIN+CAL, its control and GIN for the given steps."""
    tag = sys.argv[1]
    steps = [int(a) for a in sys.argv[2:] if a.lstrip("-").isdigit()]
    names = [a for a in sys.argv[2:] if not a.lstrip("-").isdigit()]
    pairs = [("cal_both", "nocausal_both"), ("cal_both", "vanilla"), ("nocausal_both", "vanilla")]
    if len(names) >= 2:
        pairs = [(names[-2], names[-1])]
    for s in steps:
        path = os.path.join(HERE, f"compare_vanilla_cal_backward_A_{tag}_step{s}_by_seed_3loss.csv")
        if not os.path.exists(path):
            print(f"step {s}: {path} not found")
            continue
        df = pd.read_csv(path)
        print(f"\n=== {tag}, step {s}: {len(set(df.seed))} seeds, variants {sorted(set(df.variant))} ===")
        for metric in ("mismatched_acc", "mismatched_f1"):
            for a, b in pairs:
                if a not in set(df.variant) or b not in set(df.variant):
                    continue
                d = (per_seed(df, a, metric) - per_seed(df, b, metric)).dropna().values
                if len(d) < 2:
                    continue
                lo, hi = boot_ci(d)
                print(f"  {metric:<15} {a} - {b:<14} mean {d.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
                      f"SD {d.std(ddof=1):.3f}  seeds +/-/0: {int((d > 0).sum())}/{int((d < 0).sum())}/{int((d == 0).sum())} (n={len(d)})")
 
 
if __name__ == "__main__":
    main()
"""
Tunes the two loss weights of GIN+CAL by random search at steps 0 and 3 of the backward elimination, and compares them with the fixed weights.

Needs: compare_vanilla_cal_backward.py and lambda_chooser.py.
"""
from __future__ import annotations
import os
import sys
import time
 
import numpy as np
import pandas as pd
 
import os as _os, sys as _sys
 
_sys.path.insert(0, _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "core")))
 
from lambda_chooser import make_chooser, VAL_FRACTION
import compare_vanilla_cal_backward as bw
 
ef = bw.ef
m = bw.m
HERE = bw.HERE
 
EXP = "A"                                   # experiment A, all sessions
STEPS = [0, 3]                              # 0 = all features, 3 = rms, peak, trans_b1 removed
RUN_VARIANTS = {"cal_node": False, "cal_edge": False, "cal_both": True}
CONTROLS = ["nocausal_node", "nocausal_edge", "nocausal_both", "vanilla"]   # copied from the fixed-lambda csv
 
N_PAIRS = 6
LAMBDA_RANGE = (0.1, 1.0)
LAMBDA_SEED = 2026
INCLUDE_DEFAULT = True
 
 
def sample_candidates():
    """Sample the candidate pairs of loss weights."""
    rng = np.random.default_rng(LAMBDA_SEED)
    lo, hi = LAMBDA_RANGE
    pairs = [(round(float(a), 2), round(float(b), 2)) for a, b in rng.uniform(lo, hi, size=(N_PAIRS, 2))]
    if INCLUDE_DEFAULT and (0.5, 0.5) not in pairs:
        pairs.append((0.5, 0.5))
    return pairs
 
 
def tuned_path(step):
    """Path of the result file with tuned loss weights for one step."""
    return os.path.join(HERE, f"tuned_lambda_backward_{EXP}_{bw.TAG}_step{step}_by_seed_3loss"
                        + ("_quick" if m.QUICK else "") + ".csv")
 
 
def run_step(step, ons, candidates):
    """Tune the loss weights of the CAL variants at one elimination step and save the results."""
    removed, node_keep, edge_keep = bw.kept_features(step)
    print("\n" + "#" * 100 + f"\n# STEP {step}: removed so far {removed or 'nothing'}  |  tuning {ons}\n" + "#" * 100)
    base_csv, out_csv = bw.csv_path(EXP, step), tuned_path(step)
 
    # copy lambda-independent variants (controls) from the fixed-lambda csv, once
    if os.path.exists(base_csv) and not os.path.exists(out_csv):
        b = pd.read_csv(base_csv)
        b = b[b.variant.isin(CONTROLS) & b.seed.isin(m.SEEDS)]
        b.to_csv(out_csv, index=False)
        print(f"copied {len(b)} control rows from {os.path.basename(base_csv)}")
    elif not os.path.exists(base_csv):
        print(f"note: {os.path.basename(base_csv)} not found; no fixed-lambda reference is available")
 
    m.build_dataset = bw.make_builder(EXP, step)
    m.EDGE_ATTR_DIM = len(edge_keep)
    m.DATASET_LABEL = f"{ef.LABEL[EXP]}, backward step {step}, tuned lambda"
    m.CSV_PATH = out_csv
    m.VARIANTS_TO_RUN = ons
    m.choose_lambda = make_chooser(m, candidates)
    m.report = lambda res: None
    m.main()
    return pd.read_csv(out_csv), (pd.read_csv(base_csv) if os.path.exists(base_csv) else None)
 
 
def seed_values(df, v, exp, metric):
    """Return one value of a metric per seed."""
    return ef.per_seed(df, v, exp, metric)
 
 
def report(tuned_by_step, base_by_step, ons):
    """Print the chosen loss weights and compare tuned and fixed weights with the control."""
    for step, tuned in tuned_by_step.items():
        base = base_by_step[step]
        print("\n" + "=" * 100 + f"\nSTEP {step}: chosen lambdas, tuned vs fixed (0.5, 0.5), tuned CAL vs control\n" + "=" * 100)
        for v in ons:
            for exp in ["random", "cross"]:
                d = tuned[(tuned.variant == v) & (tuned.experiment == exp)]
                if d.empty:
                    continue
                pairs = list(zip(d.lam1.round(2), d.lam2.round(2)))
                top = pd.Series(pairs).value_counts().head(3)
                print(f"  {v:<9} {exp:<7} lambda1 {d.lam1.mean():.2f} +/- {d.lam1.std(ddof=1):.2f}  "
                      f"lambda2 {d.lam2.mean():.2f} +/- {d.lam2.std(ddof=1):.2f}  most frequent: "
                      + ", ".join(f"{p}: {c}/{len(pairs)}" for p, c in top.items()))
        if base is not None:
            print("\n  tuned minus fixed (same seed, same split; bootstrap over seeds)")
            for v in ons:
                for exp, metric, label in [("random", "test_acc", "random test acc"),
                                           ("cross", "mismatched_acc", "cross mismatched acc"),
                                           ("cross", "mismatched_f1", "cross mismatched F1")]:
                    d = (seed_values(tuned, v, exp, metric) - seed_values(base, v, exp, metric)).dropna().values
                    if len(d) > 1:
                        print(m.diff_line(f"  {v} {label}", d))
        print("\n  tuned CAL minus control (nocausal, fixed), cross mismatched acc")
        for v in ons:
            ctrl = v.replace("cal_", "nocausal_")
            if ctrl in set(tuned.variant):
                d = (seed_values(tuned, v, "cross", "mismatched_acc")
                     - seed_values(tuned, ctrl, "cross", "mismatched_acc")).dropna().values
                if len(d) > 1:
                    print(m.diff_line(f"  {v} - {ctrl}", d))
 
    if 0 in tuned_by_step and 3 in tuned_by_step:
        print("\n" + "=" * 100 + "\nSTEP 3 minus STEP 0 with tuned lambda, cross mismatched acc\n" + "=" * 100)
        for v in ons:
            d = (seed_values(tuned_by_step[3], v, "cross", "mismatched_acc")
                 - seed_values(tuned_by_step[0], v, "cross", "mismatched_acc")).dropna().values
            if len(d) > 1:
                print(m.diff_line(f"  {v}", d))
 
 
if __name__ == "__main__":
    steps = STEPS
    if len(sys.argv) > 1 and sys.argv[1] == "steps":
        steps = [int(a) for a in sys.argv[2:]]
    ons = [v for v, on in RUN_VARIANTS.items() if on]
    candidates = sample_candidates()
    print(f"candidate (lambda1, lambda2) pairs: {candidates}")
    print(f"experiment {EXP}; steps {steps}; tuned variants {ons}; seeds {m.SEEDS}; epochs {m.EPOCHS}; "
          f"val fraction {VAL_FRACTION}\n")
    t0 = time.time()
    tuned_by_step, base_by_step = {}, {}
    for s in steps:
        tuned_by_step[s], base_by_step[s] = run_step(s, ons, candidates)
        print(f"\n[step {s}] done, total elapsed {(time.time() - t0) / 60:.1f} min")
    report(tuned_by_step, base_by_step, ons)
 
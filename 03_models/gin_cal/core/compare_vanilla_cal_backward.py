"""
Backward feature elimination (Experiment 1): removes the input features one at a time, starting with the most temperature-sensitive, and records the cross-environment accuracy at each step. Also gives setting 2b.

Needs: compare_vanilla_cal_edge_features.py and compare_vanilla_cal.py; the removal order from feature_temperature_damage_analysis.py.
"""
from __future__ import annotations
import os
import sys
 
import numpy as np
import pandas as pd
import torch
 
import compare_vanilla_cal_edge_features as ef   # re-uses the loaded compare_vanilla_cal module and helpers
 
m = ef.m
HERE = ef.HERE
 
TAG = "n186"                                      # result files carry this tag
 
# (feature, kind, Healthy-only week-grouped RF R2 on all sessions), largest R2 first
REMOVAL_ORDER = [("rms", "node", 0.95), ("peak", "node", 0.95), ("trans_b1", "edge", 0.82),
                 ("trans_t1", "edge", 0.66), ("trans_b2", "edge", 0.64), ("freq_hz", "node", 0.24),
                 ("coh_b1", "edge", 0.20), ("coh_b2", "edge", 0.14), ("coh_t1", "edge", -0.05)]
MAX_STEP = len(REMOVAL_ORDER)                      # 9
NODE_COL = {"rms": 0, "peak": 1, "freq_hz": 2}
INDICATOR_COLS = [3, 4, 5]
EDGE_COL = ef.EDGE_COL
EXPERIMENT = "A"
DEFAULT_VARIANTS = ["vanilla"]
_orig_build = m.build_dataset
 
 
def kept_features(step):
    """Return the removed features at a step and the node and edge features that remain."""
    removed = [f for f, _, _ in REMOVAL_ORDER[:step]]
    node = [f for f in NODE_COL if f not in removed]
    edge = [f for f in EDGE_COL if f not in removed]
    return removed, node, edge
 
 
def transform(g, node_keep, edge_keep):
    """Keep the selected node and edge feature columns of a graph, together with the sensor-type indicators."""
    x = g.x
    assert x.shape[1] == 6, f"expected 6 base node features, got {x.shape[1]}"
    ind = x[:, 3:6]
    assert torch.all((ind == 0) | (ind == 1)) and torch.all(ind.sum(1) == 1), \
        "columns 3-5 are not one-hot indicators"
    assert g.edge_attr.shape[1] == 7, f"expected 7 edge features, got {g.edge_attr.shape[1]}"
    g2 = g.clone()
    g2.x = x[:, [NODE_COL[f] for f in node_keep] + INDICATOR_COLS].contiguous()
    g2.edge_attr = g.edge_attr[:, [EDGE_COL[f] for f in edge_keep]].contiguous()
    return g2
 
 
def make_builder(exp, step):
    """Return a graph builder that applies the feature removal of one step."""
    removed, node_keep, edge_keep = kept_features(step)
 
    def build(df, **kw):
        """Build the graphs and remove the features of the current step."""
        graphs, enc = _orig_build(df, **kw)
        if exp == "B":
            graphs, enc = ef.exclude_classes(graphs, enc, ef.EXCLUDED)
        ef.check_edge_columns(graphs)
        graphs = [transform(g, node_keep, edge_keep) for g in graphs]
        print(f"Step {step}: removed {removed or 'nothing'}; node features {node_keep} + 3 indicators "
              f"(dim {graphs[0].x.shape[1]}); edge features {edge_keep} (dim {graphs[0].edge_attr.shape[1]}); "
              f"{len(graphs)} graphs, {len(enc.classes_)} classes")
        return graphs, enc
    return build
 
 
def csv_path(exp, step):
    """Path of the result file for one experiment and step."""
    return os.path.join(HERE, f"compare_vanilla_cal_backward_{exp}_{TAG}_step{step}_by_seed_3loss"
                        + ("_quick" if m.QUICK else "") + ".csv")
 
 
def curve(exp, steps, variants):
    """Collect the results of all steps into the backward elimination curve and the paired differences."""
    frames = {s: pd.read_csv(csv_path(exp, s)) for s in steps if os.path.exists(csv_path(exp, s))}
    if 0 not in frames:
        print("note: step 0 not available; paired differences against step 0 are skipped")
    rows = []
    for s, df in frames.items():
        removed, node_keep, edge_keep = kept_features(s)
        for v in variants:
            if v not in set(df.variant):
                continue
            rnd = ef.per_seed(df, v, "random", "test_acc")
            acc = ef.per_seed(df, v, "cross", "mismatched_acc")
            f1 = ef.per_seed(df, v, "cross", "mismatched_f1")
            row = dict(step=s, removed_last=(removed[-1] if removed else "-"), n_node_response=len(node_keep),
                       n_edge=len(edge_keep), variant=v, random_acc=rnd.mean(), cross_acc=acc.mean(),
                       cross_acc_sd=acc.std(ddof=1), cross_f1=f1.mean(), n_seeds=len(acc))
            if 0 in frames and v in set(frames[0].variant) and s != 0:
                d = (acc - ef.per_seed(frames[0], v, "cross", "mismatched_acc")).dropna().values
                if len(d) > 1:
                    lo, hi = m.boot_ci(d)
                    row.update(diff_vs_step0=d.mean(), ci_lo=lo, ci_hi=hi)
            rows.append(row)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\n" + "=" * 110 + f"\nBACKWARD ELIMINATION CURVE, experiment {exp} ({ef.LABEL[exp]}); order: largest temperature R2 first\n"
          + "=" * 110)
    for v in variants:
        r = res[res.variant == v]
        if len(r):
            print(f"\nvariant: {v}")
            print(r.drop(columns=["variant"]).round(3).to_string(index=False))
    suffix = "" if variants == DEFAULT_VARIANTS else "_" + "_".join(variants)   # do not overwrite the vanilla curve
    out = os.path.join(HERE, f"compare_vanilla_cal_backward_{exp}_{TAG}_curve{suffix}" + ("_quick" if m.QUICK else "") + ".csv")
    res.to_csv(out, index=False)
    print(f"\ncurve saved to {out}") 
 
def parse(argv):
    """Read the experiment, steps and variants from the command line arguments."""
    exp, steps, variants, mode = EXPERIMENT, list(range(MAX_STEP + 1)), list(DEFAULT_VARIANTS), None
    for a in argv:
        if a in ("A", "B"):
            exp = a
        elif a in ("steps", "variants"):
            mode = a
            if a == "steps":
                steps = []
            else:
                variants = []
        elif mode == "steps":
            steps.append(int(a))
        elif mode == "variants":
            variants.append(a)
        else:
            raise SystemExit(f"unexpected argument {a}")
    assert all(0 <= s <= MAX_STEP for s in steps), f"steps must be in 0..{MAX_STEP}"
    for v in variants:
        assert v in m.VARIANTS, f"unknown variant {v}; choose from {list(m.VARIANTS)}"
    return exp, steps, variants
 
 
if __name__ == "__main__":
    exp, steps, variants = parse(sys.argv[1:])
    assert hasattr(m, "EDGE_ATTR_DIM"), "compare_vanilla_cal.py must take the edge width from a module-level EDGE_ATTR_DIM"
    print(f"experiment {exp} ({ef.LABEL[exp]}); steps {steps}; variants {variants}; weight decay {m.WEIGHT_DECAY} (ON); "
          f"seeds {m.SEEDS}; lambdas ({m.LAMBDA1}, {m.LAMBDA2})")
    m.report = lambda res: None                     # the per-step report is replaced by the curve printed at the end
    for s in steps:
        removed, node_keep, edge_keep = kept_features(s)
        print("\n" + "#" * 100 + f"\n# STEP {s}: removed so far {removed or 'nothing'}\n" + "#" * 100)
        m.build_dataset = make_builder(exp, s)
        m.EDGE_ATTR_DIM = len(edge_keep)
        m.DATASET_LABEL = f"{ef.LABEL[exp]}, backward step {s}"
        m.CSV_PATH = csv_path(exp, s)
        m.VARIANTS_TO_RUN = variants
        m.main()
    curve(exp, list(range(MAX_STEP + 1)), variants)     # every step that has a csv, so the curve file is always complete
 
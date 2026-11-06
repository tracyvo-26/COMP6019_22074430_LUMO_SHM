
"""
Exploratory run with different sets of node features.

Needs: compare_vanilla_cal.py.
"""
from __future__ import annotations
import importlib.util
import os
import sys
from types import SimpleNamespace
 
import numpy as np
import pandas as pd
import torch
 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "core"))                      # shared modules (compare_vanilla_cal*.py)
HERE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "results"))   # all CSV results of gin_cal live here
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)
spec = importlib.util.spec_from_file_location("cvc", os.path.join(CORE_DIR, "compare_vanilla_cal.py"))
m = importlib.util.module_from_spec(spec)
sys.modules["cvc"] = m
spec.loader.exec_module(m)
 
FEATURE_SETS = ["no_freq", "indicators_only", "edge_avg", "base_plus_edge_avg"]
FEATURE_SETS_TO_RUN = ["edge_avg", "no_freq", "indicators_only"]
EXPERIMENT = "A"
EXCLUDED = ["DAM3_010", "DAM4_010"]
BASE_CSV = {"A": "compare_vanilla_cal_by_seed_3loss.csv", "B": "compare_vanilla_cal_excl_by_seed_3loss.csv"}
LABEL = {"A": "all sessions", "B": "DAM3_010 and DAM4_010 removed"}
ALL_VARIANTS = list(m.VARIANTS)[:8]
_orig_build = m.build_dataset
 
 
def incident_mean(edge_index, edge_attr, n_nodes):
    """Mean of edge_attr over all directed edges that have the node as source or target."""
    d = edge_attr.shape[1]
    acc = torch.zeros(n_nodes, d, dtype=edge_attr.dtype)
    cnt = torch.zeros(n_nodes, 1, dtype=edge_attr.dtype)
    ones = torch.ones(edge_attr.shape[0], 1, dtype=edge_attr.dtype)
    for end in (edge_index[0], edge_index[1]):
        acc.index_add_(0, end, edge_attr)
        cnt.index_add_(0, end, ones)
    return acc / cnt.clamp(min=1.0)
 
 
def transform(g, name):
    """Create the chosen set of node features from the six base node features."""
    x = g.x
    assert x.shape[1] == 6, f"expected 6 base node features, got {x.shape[1]}"
    ind = x[:, 3:6]
    assert torch.all((ind == 0) | (ind == 1)) and torch.all(ind.sum(1) == 1), \
        "columns 3-5 are not one-hot indicators"
    if name == "base":
        new = x
    elif name == "no_freq":
        new = x[:, [0, 1, 3, 4, 5]]
    elif name == "indicators_only":
        new = ind
    elif name in ("edge_avg", "base_plus_edge_avg"):
        em = incident_mean(g.edge_index, g.edge_attr, x.shape[0])
        new = torch.cat([em, ind], 1) if name == "edge_avg" else torch.cat([x, em], 1)
    else:
        raise ValueError(name)
    g2 = g.clone()
    g2.x = new.contiguous()
    return g2
 
 
def exclude_classes(graphs, enc, excluded):
    """Remove the graphs of the excluded classes and relabel the remaining classes."""
    names = list(enc.classes_)
    keep = [c for c in names if c not in excluded]
    assert len(keep) < len(names), f"none of {excluded} found in {names}"
    idx = {c: i for i, c in enumerate(keep)}
    out = []
    for g in graphs:
        name = names[g.y.item()]
        if name in excluded:
            continue
        g2 = g.clone()
        g2.y = torch.tensor([idx[name]], dtype=g.y.dtype)
        out.append(g2)
    return out, SimpleNamespace(classes_=np.array(keep))
 
 
def make_builder(exp, name):
    """Return a graph builder that uses one set of node features."""
    def build(df, **kw):
        """Build the graphs and create the node features of the current set."""
        graphs, enc = _orig_build(df, **kw)
        if exp == "B":
            graphs, enc = exclude_classes(graphs, enc, EXCLUDED)
        graphs = [transform(g, name) for g in graphs]
        print(f"Node feature set '{name}': node dim {graphs[0].x.shape[1]}, edge dim {graphs[0].edge_attr.shape[1]}, "
              f"{len(graphs)} graphs, {len(enc.classes_)} classes")
        return graphs, enc
    return build
 
 
def per_seed(df, v, exp, metric):
    """Return one value of a metric per seed for a variant and experiment."""
    d = df[(df.variant == v) & (df.experiment == exp)]
    return d.set_index("seed")[metric] if exp == "random" else d.groupby("seed")[metric].mean()
 
 
def summary(exp, names):
    """Print the results of all node feature sets next to the base set."""
    frames = {}
    base_path = os.path.join(HERE, BASE_CSV[exp])
    if os.path.exists(base_path):
        frames["base"] = pd.read_csv(base_path)
    else:
        print(f"note: {BASE_CSV[exp]} not found; no comparison with 'base' possible")
    for n in names:
        p = csv_path(exp, n)
        if os.path.exists(p):
            frames[n] = pd.read_csv(p)
    pd.set_option("display.width", 250)
    print("\n" + "=" * 110 + f"\nSUMMARY across node feature sets, experiment {exp} ({LABEL[exp]}), mean over seeds\n" + "=" * 110)
    for metric, e, title in [("test_acc", "random", "RANDOM SPLIT: test accuracy"),
                             ("mismatched_acc", "cross", "CROSS-ENVIRONMENT: mismatched accuracy (mean of 6 settings)"),
                             ("mismatched_f1", "cross", "CROSS-ENVIRONMENT: mismatched macro-F1")]:
        tab = {fs: {v: per_seed(df, v, e, metric).mean() for v in ALL_VARIANTS if v in set(df.variant)}
               for fs, df in frames.items()}
        print(f"\n{title}")
        print(pd.DataFrame(tab).T.round(3).to_string())
    if "base" in frames:
        print("\nPAIRED difference (feature set - base), same variant, same seed; bootstrap over seeds")
        for fs in [n for n in names if n in frames]:
            print(f"\n  feature set: {fs}")
            for v in ALL_VARIANTS:
                if v not in set(frames[fs].variant) or v not in set(frames["base"].variant):
                    continue
                for metric, e, lab in [("test_acc", "random", "random test acc"),
                                       ("mismatched_acc", "cross", "cross mism. acc")]:
                    d = (per_seed(frames[fs], v, e, metric) - per_seed(frames["base"], v, e, metric)).dropna().values
                    if len(d) > 1:
                        print(m.diff_line(f"{v} {lab}", d))
 
 
def csv_path(exp, name):
    """Path of the result file for one experiment and node feature set."""
    return os.path.join(HERE, f"compare_vanilla_cal_nodefeat_{exp}_{name}_by_seed_3loss"
                        + ("_quick" if m.QUICK else "") + ".csv")
 
 
if __name__ == "__main__":
    args = sys.argv[1:]
    exp = EXPERIMENT
    if args and args[0] in ("A", "B"):
        exp, args = args[0], args[1:]
    names = args or FEATURE_SETS_TO_RUN
    for n in names:
        assert n in FEATURE_SETS, f"unknown feature set {n}; choose from {FEATURE_SETS}"
    print(f"experiment {exp} ({LABEL[exp]}); feature sets to run: {names}; weight decay {m.WEIGHT_DECAY} (ON); "
          f"seeds {m.SEEDS}; lambdas ({m.LAMBDA1}, {m.LAMBDA2})")
    for n in names:
        print("\n" + "#" * 100 + f"\n# NODE FEATURE SET: {n}\n" + "#" * 100)
        m.build_dataset = make_builder(exp, n)
        m.DATASET_LABEL = f"{LABEL[exp]}, node features: {n}"
        m.CSV_PATH = csv_path(exp, n)
        m.VARIANTS_TO_RUN = ALL_VARIANTS
        m.main()
    summary(exp, names)
 

"""
Helper for the model scripts. Runs the same models with a chosen set of edge features and compares them with a reference run.

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
CORE_DIR = SCRIPT_DIR                      # shared modules (compare_vanilla_cal*.py)
HERE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "results"))   # all CSV results of gin_cal live here
spec = importlib.util.spec_from_file_location("cvc", os.path.join(CORE_DIR, "compare_vanilla_cal.py"))
m = importlib.util.module_from_spec(spec)
sys.modules["cvc"] = m
spec.loader.exec_module(m)
 
EDGE_COL = {"xcov": 0, "coh_b1": 1, "coh_t1": 2, "coh_b2": 3, "trans_b1": 4, "trans_t1": 5, "trans_b2": 6}
EDGE_SETS = {
    "coh_plus_trans_t1": ["coh_t1", "coh_b2", "trans_t1"],
    "coherence":         ["coh_b1", "coh_t1", "coh_b2"],
    "drop_xcov_cohb1":   ["coh_t1", "coh_b2", "trans_b1", "trans_t1", "trans_b2"],
    "trans_t1_only":     ["trans_t1"],
    "transmissibility":  ["trans_b1", "trans_t1", "trans_b2"],
    # same 5 edge features as drop_xcov_cohb1, but with the original 6 node features (rms, peak, freq_hz kept)
    "drop_xcov_cohb1_node6": ["coh_t1", "coh_b2", "trans_b1", "trans_t1", "trans_b2"],
}
EDGE_SETS["drop_xcov_cohb1_transb1"] = ["coh_t1", "coh_b2", "trans_t1", "trans_b2"]   # also drops trans_b1 (3 indicator nodes)
# drop_xcov_cohb1_transb1 = drop_xcov_cohb1 with trans_b1 also removed (indicator nodes).
NODE_OF_SET = {"drop_xcov_cohb1_node6": "base"}      # sets not listed use the 3 indicators
SETS_TO_RUN = ["coh_plus_trans_t1", "coherence", "drop_xcov_cohb1", "trans_t1_only"]
NODE_SET = "indicators_only"            # "indicators_only" (3 sensor-type columns) or "base" (original 6)
EXPERIMENT = "A"
EXCLUDED = ["DAM3_010", "DAM4_010"]
BASE_CSV = {"A": "compare_vanilla_cal_by_seed_3loss.csv", "B": "compare_vanilla_cal_excl_by_seed_3loss.csv"}
LABEL = {"A": "all sessions", "B": "DAM3_010 and DAM4_010 removed"}
ALL_VARIANTS = list(m.VARIANTS)[:8]
_orig_build = m.build_dataset
_checked = {"done": False}
 
 
def check_edge_columns(graphs):
    """Print the value range of each edge column and stop if the assumed column order is implausible."""
    if _checked["done"]:
        return
    E = torch.cat([g.edge_attr for g in graphs]).numpy()
    assert E.shape[1] == 7, f"expected 7 edge features, got {E.shape[1]}"
    names = list(EDGE_COL)
    print("raw edge feature ranges (min / median / max):")
    for name, j in EDGE_COL.items():
        col = E[:, j]
        print(f"  col {j} {name:9s} {np.nanmin(col):10.4g} {np.nanmedian(col):10.4g} {np.nanmax(col):10.4g}")
    for name in ("coh_b1", "coh_t1", "coh_b2"):
        col = E[:, EDGE_COL[name]]
        assert np.nanmin(col) >= -1e-6 and np.nanmax(col) <= 1 + 1e-6, \
            f"column {EDGE_COL[name]} ({name}) is outside [0, 1]: the edge column order differs from {names}"
    col = E[:, EDGE_COL["xcov"]]
    assert np.nanmin(col) >= -1 - 1e-6 and np.nanmax(col) <= 1 + 1e-6, \
        f"column 0 (xcov) is outside [-1, 1]: the edge column order differs from {names}"
    _checked["done"] = True
 
 
def transform(g, cols):
    """Keep the selected edge feature columns and the chosen node features of a graph."""
    x = g.x
    assert x.shape[1] == 6, f"expected 6 base node features, got {x.shape[1]}"
    ind = x[:, 3:6]
    assert torch.all((ind == 0) | (ind == 1)) and torch.all(ind.sum(1) == 1), \
        "columns 3-5 are not one-hot indicators"
    assert g.edge_attr.shape[1] == 7, f"expected 7 edge features, got {g.edge_attr.shape[1]}"
    g2 = g.clone()
    g2.x = (ind if NODE_SET == "indicators_only" else x).contiguous()
    g2.edge_attr = g.edge_attr[:, cols].contiguous()
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
 
 
def make_builder(exp, set_name):
    """Return a graph builder that uses one set of edge features."""
    cols = [EDGE_COL[n] for n in EDGE_SETS[set_name]]
 
    def build(df, **kw):
        """Build the graphs and keep the edge features of the current set."""
        graphs, enc = _orig_build(df, **kw)
        if exp == "B":
            graphs, enc = exclude_classes(graphs, enc, EXCLUDED)
        check_edge_columns(graphs)
        graphs = [transform(g, cols) for g in graphs]
        print(f"Edge feature set '{set_name}' {EDGE_SETS[set_name]}: node dim {graphs[0].x.shape[1]} ({NODE_SET}), "
              f"edge dim {graphs[0].edge_attr.shape[1]}, {len(graphs)} graphs, {len(enc.classes_)} classes")
        return graphs, enc
    return build
 
 
def per_seed(df, v, exp, metric):
    """Return one value of a metric per seed for a variant and experiment."""
    d = df[(df.variant == v) & (df.experiment == exp)]
    return d.set_index("seed")[metric] if exp == "random" else d.groupby("seed")[metric].mean()
 
 
def csv_path(exp, name):
    """Path of the result file for one experiment and edge feature set."""
    return os.path.join(HERE, f"compare_vanilla_cal_edgefeat_{exp}_{name}_by_seed_3loss"
                        + ("_quick" if m.QUICK else "") + ".csv")
 
 
def reference_paths(exp):
    """Return the result files used as references for the comparison."""
    ref = os.path.join(HERE, f"compare_vanilla_cal_nodefeat_{exp}_indicators_only_by_seed_3loss.csv") \
        if NODE_SET == "indicators_only" else os.path.join(HERE, BASE_CSV[exp])
    return ref, os.path.join(HERE, BASE_CSV[exp])
 
 
def summary(exp, names):
    """Print the results of all edge feature sets next to the reference runs."""
    ref_path, base_path = reference_paths(exp)
    frames = {}
    if os.path.exists(ref_path):
        frames["all7"] = pd.read_csv(ref_path)
    else:
        print(f"note: reference {os.path.basename(ref_path)} not found; no comparison with 'all7' possible")
    if os.path.exists(base_path) and NODE_SET != "base":
        frames["base"] = pd.read_csv(base_path)
    for n in names:
        p = csv_path(exp, n)
        if os.path.exists(p):
            frames[n] = pd.read_csv(p)
    pd.set_option("display.width", 250)
    print("\n" + "=" * 110 + f"\nSUMMARY across edge feature sets, experiment {exp} ({LABEL[exp]}), node set {NODE_SET}, "
          f"mean over seeds\n" + "=" * 110)
    for metric, e, title in [("test_acc", "random", "RANDOM SPLIT: test accuracy"),
                             ("mismatched_acc", "cross", "CROSS-ENVIRONMENT: mismatched accuracy (mean of 6 settings)"),
                             ("mismatched_f1", "cross", "CROSS-ENVIRONMENT: mismatched macro-F1")]:
        tab = {fs: {v: per_seed(df, v, e, metric).mean() for v in ALL_VARIANTS if v in set(df.variant)}
               for fs, df in frames.items()}
        print(f"\n{title}")
        print(pd.DataFrame(tab).T.round(3).to_string())
    for ref in ("all7", "base"):
        if ref not in frames:
            continue
        print(f"\nPAIRED difference (edge set - {ref}), same variant, same seed; bootstrap over seeds")
        for fs in [n for n in names if n in frames]:
            print(f"\n  edge set: {fs}")
            for v in ALL_VARIANTS:
                if v not in set(frames[fs].variant) or v not in set(frames[ref].variant):
                    continue
                for metric, e, lab in [("test_acc", "random", "random test acc"),
                                       ("mismatched_acc", "cross", "cross mism. acc")]:
                    d = (per_seed(frames[fs], v, e, metric) - per_seed(frames[ref], v, e, metric)).dropna().values
                    if len(d) > 1:
                        print(m.diff_line(f"{v} {lab}", d)) 
 
if __name__ == "__main__":
    args = sys.argv[1:]
    exp = EXPERIMENT
    if args and args[0] in ("A", "B"):
        exp, args = args[0], args[1:]
    run_variants = ALL_VARIANTS
    if "variants" in args:                       # e.g.  ... drop_xcov_cohb1_transb1 variants vanilla
        i = args.index("variants")
        run_variants = args[i + 1:]
        args = args[:i]
        for v in run_variants:
            assert v in ALL_VARIANTS, f"unknown variant {v}; choose from {ALL_VARIANTS}"
        assert run_variants, "give at least one variant after 'variants'"
    names = args or SETS_TO_RUN
    for n in names:
        assert n in EDGE_SETS, f"unknown edge feature set {n}; choose from {list(EDGE_SETS)}"
    assert NODE_SET in ("indicators_only", "base")
    assert hasattr(m, "EDGE_ATTR_DIM"), "compare_vanilla_cal.py must take the edge width from a module-level EDGE_ATTR_DIM"
    print(f"experiment {exp} ({LABEL[exp]}); node set {NODE_SET}; edge sets to run: {names}; weight decay "
          f"{m.WEIGHT_DECAY} (ON); seeds {m.SEEDS}; lambdas ({m.LAMBDA1}, {m.LAMBDA2})")
    for n in names:
        NODE_SET = NODE_OF_SET.get(n, "indicators_only")     # transform() reads this global when the dataset is built
        print("\n" + "#" * 100 + f"\n# EDGE FEATURE SET: {n} {EDGE_SETS[n]}  (node set: {NODE_SET})\n" + "#" * 100)
        m.build_dataset = make_builder(exp, n)
        m.EDGE_ATTR_DIM = len(EDGE_SETS[n])          # the models take their edge input width from this global
        m.DATASET_LABEL = f"{LABEL[exp]}, node features: {NODE_SET}, edge features: {n}"
        m.CSV_PATH = csv_path(exp, n)
        m.VARIANTS_TO_RUN = run_variants
        m.main()
    NODE_SET = "indicators_only"       # summary compares with both 'all7' (indicator nodes) and 'base' (6 node, 7 edge)
    summary(exp, names)
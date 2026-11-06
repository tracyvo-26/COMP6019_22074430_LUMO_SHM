
"""
Exploratory run that replaces the sensor-type indicators with a constant node feature, to see whether the indicators help.

Needs: compare_vanilla_cal_edge_features.py and earlier result files.
"""
from __future__ import annotations
import os
import sys
 
import pandas as pd
import torch
 
import os as _os, sys as _sys
 
_sys.path.insert(0, _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "core")))
 
import compare_vanilla_cal_edge_features as ef   # re-uses the loaded compare_vanilla_cal module and helpers
 
m = ef.m
HERE = ef.HERE
EXPERIMENT = "A"
ALL7 = list(ef.EDGE_COL)
EDGE_SETS = {"all7": ALL7, "drop_xcov_cohb1": ef.EDGE_SETS["drop_xcov_cohb1"]}
REFERENCE_CSV = {"all7": "compare_vanilla_cal_nodefeat_A_indicators_only_by_seed_3loss.csv",
                 "drop_xcov_cohb1": "compare_vanilla_cal_edgefeat_A_drop_xcov_cohb1_by_seed_3loss.csv"}
DEFAULT_VARIANTS = ["vanilla"]
_orig_build = m.build_dataset
 
 
def transform(g, cols):
    """Replace the node features by a constant column and keep the selected edge features."""
    assert g.x.shape[1] == 6, f"expected 6 base node features, got {g.x.shape[1]}"
    assert g.edge_attr.shape[1] == 7, f"expected 7 edge features, got {g.edge_attr.shape[1]}"
    g2 = g.clone()
    g2.x = torch.ones(g.x.shape[0], 1, dtype=g.x.dtype)
    g2.edge_attr = g.edge_attr[:, cols].contiguous()
    return g2
 
 
def make_builder(set_name):
    """Return a graph builder with a constant node feature and one set of edge features."""
    cols = [ef.EDGE_COL[n] for n in EDGE_SETS[set_name]]
 
    def build(df, **kw):
        """Build the graphs with a constant node feature."""
        graphs, enc = _orig_build(df, **kw)
        ef.check_edge_columns(graphs)
        graphs = [transform(g, cols) for g in graphs]
        print(f"Constant node feature (one column of ones); edge features {EDGE_SETS[set_name]} "
              f"(dim {graphs[0].edge_attr.shape[1]}); node dim {graphs[0].x.shape[1]}; "
              f"{len(graphs)} graphs, {len(enc.classes_)} classes")
        return graphs, enc
    return build
 
 
def csv_path(set_name):
    """Path of the result file for one edge feature set."""
    return os.path.join(HERE, f"compare_vanilla_cal_nodeconst_{EXPERIMENT}_{set_name}_by_seed_3loss"
                        + ("_quick" if m.QUICK else "") + ".csv")
 
 
def summary(sets, variants):
    """Print the constant node feature results next to the runs with sensor-type indicators."""
    pd.set_option("display.width", 250)
    print("\n" + "=" * 110 + "\nCONSTANT NODE FEATURE against the 3 sensor-type indicators (same edge features)\n" + "=" * 110)
    for s in sets:
        new_p, ref_p = csv_path(s), os.path.join(HERE, REFERENCE_CSV[s])
        if not os.path.exists(new_p):
            continue
        new = pd.read_csv(new_p)
        ref = pd.read_csv(ref_p) if os.path.exists(ref_p) else None
        print(f"\nedge set: {s} {EDGE_SETS[s]}")
        if ref is None:
            print(f"  note: reference {REFERENCE_CSV[s]} not found; no paired comparison")
        for v in variants:
            if v not in set(new.variant):
                continue
            for metric, e, lab in [("test_acc", "random", "random test acc"),
                                   ("mismatched_acc", "cross", "cross mism. acc"),
                                   ("mismatched_f1", "cross", "cross mism. F1")]:
                a = ef.per_seed(new, v, e, metric)
                line = f"  {v:<14} {lab:<16} constant node {a.mean():.3f}"
                if ref is not None and v in set(ref.variant):
                    b = ef.per_seed(ref, v, e, metric)
                    d = (a - b).dropna().values
                    line += f" | indicators {b.mean():.3f}"
                    print(line)
                    if len(d) > 1:
                        print(m.diff_line(f"    {v} {lab} (constant - indicators)", d))
                else:
                    print(line)
 
 
def parse(argv):
    """Read the edge feature sets and variants from the command line arguments."""
    sets, variants, mode = list(EDGE_SETS), list(DEFAULT_VARIANTS), None
    for a in argv:
        if a in ("edges", "variants"):
            mode = a
            if a == "edges":
                sets = []
            else:
                variants = []
        elif mode == "edges":
            assert a in EDGE_SETS, f"unknown edge set {a}; choose from {list(EDGE_SETS)}"
            sets.append(a)
        elif mode == "variants":
            assert a in m.VARIANTS, f"unknown variant {a}; choose from {list(m.VARIANTS)}"
            variants.append(a)
        else:
            raise SystemExit(f"unexpected argument {a}")
    return sets, variants
 
 
if __name__ == "__main__":
    sets, variants = parse(sys.argv[1:])
    assert hasattr(m, "EDGE_ATTR_DIM"), "compare_vanilla_cal.py must take the edge width from a module-level EDGE_ATTR_DIM"
    print(f"experiment {EXPERIMENT} (all sessions); edge sets {sets}; variants {variants}; weight decay {m.WEIGHT_DECAY} (ON); "
          f"seeds {m.SEEDS}; lambdas ({m.LAMBDA1}, {m.LAMBDA2})")
    m.report = lambda res: None                      # replaced by the paired summary at the end
    for s in sets:
        print("\n" + "#" * 100 + f"\n# CONSTANT NODE FEATURE, edge set {s}\n" + "#" * 100)
        m.build_dataset = make_builder(s)
        m.EDGE_ATTR_DIM = len(EDGE_SETS[s])
        m.DATASET_LABEL = f"all sessions, constant node feature, edge features: {s}"
        m.CSV_PATH = csv_path(s)
        m.VARIANTS_TO_RUN = variants
        m.main()
    summary(sets, variants)
 
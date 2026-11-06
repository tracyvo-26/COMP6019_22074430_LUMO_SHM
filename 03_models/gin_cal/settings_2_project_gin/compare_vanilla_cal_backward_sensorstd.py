"""
Setting 2c: the project GIN with the node features standardized separately for accelerometers and strain gauges.

Needs: compare_vanilla_cal_backward.py.
"""
from __future__ import annotations
import os
import sys
 
import torch
 
import os as _os, sys as _sys
 
_sys.path.insert(0, _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "core")))
 
import compare_vanilla_cal_backward as bw
 
m = bw.m
GROUPS = os.environ.get("SENSOR_STD_GROUPS", "2")
assert GROUPS in ("2", "3"), "SENSOR_STD_GROUPS must be 2 or 3"
bw.TAG = f"n186_sensorstd{GROUPS}"           # result files carry this tag; pooled-standardisation files are left alone
N_IND = 3                                      # is_accel_x, is_accel_y, is_strain: always the last 3 columns of x
 
 
def group_ids(x):
    """Group index of every node from the one-hot indicators in the last 3 columns."""
    ind = x[:, -N_IND:]
    kind = ind.argmax(1)                       # 0 = accel x, 1 = accel y, 2 = strain
    if GROUPS == "2":
        return torch.where(kind == 2, torch.ones_like(kind), torch.zeros_like(kind))
    return kind
 
 
def fit_standardizer(graphs):
    """Standardise the response features per sensor group and the indicators and edges with pooled statistics."""
    X = torch.cat([g.x for g in graphs])
    E = torch.cat([g.edge_attr for g in graphs])
    assert torch.all((X[:, -N_IND:] == 0) | (X[:, -N_IND:] == 1)), "the last 3 columns of x must be the indicators"
    # pooled statistics for the indicators and for the edges: identical to compare_vanilla_cal.py
    mx, sx, me, se = X.mean(0), X.std(0), E.mean(0), E.std(0)
    sx = torch.where(sx < 1e-8, torch.ones_like(sx), sx)
    se = torch.where(se < 1e-8, torch.ones_like(se), se)
    # per-group statistics for the response columns
    gid = group_ids(X)
    n_groups = int(gid.max().item()) + 1 if len(gid) else 1
    n_resp = X.shape[1] - N_IND
    gm = torch.zeros(n_groups, n_resp)
    gs = torch.ones(n_groups, n_resp)
    if n_resp > 0:
        for k in range(n_groups):
            xk = X[gid == k][:, :n_resp]
            if len(xk) == 0:
                continue
            mk = xk.mean(0)
            sk = xk.std(0) if len(xk) > 1 else torch.ones(n_resp)
            gm[k] = mk
            gs[k] = torch.where(sk < 1e-8, torch.ones_like(sk), sk)
    return mx, sx, me, se, gm, gs
 
 
def apply_standardizer(graphs, st):
    """Apply the per-group standardisation to every graph."""
    mx, sx, me, se, gm, gs = st
    n_resp = mx.shape[0] - N_IND
    out = []
    for g in graphs:
        g2 = g.clone()
        x = (g.x - mx) / sx                    # indicators standardised with pooled statistics
        if n_resp > 0:
            gid = group_ids(g.x)
            x[:, :n_resp] = (g.x[:, :n_resp] - gm[gid]) / gs[gid]
        g2.x = x
        g2.edge_attr = (g.edge_attr - me) / se
        out.append(g2)
    return out
 
 
if __name__ == "__main__":
    exp, steps, variants = bw.parse(sys.argv[1:])
    assert hasattr(m, "EDGE_ATTR_DIM"), "compare_vanilla_cal.py must take the edge width from a module-level EDGE_ATTR_DIM"
    m.fit_standardizer = fit_standardizer      # fit_and_eval looks these two up in the module namespace
    m.apply_standardizer = apply_standardizer
    print(f"PER-SENSOR standardisation of the node response features, {GROUPS} groups "
          f"({'accelerometers | strain' if GROUPS == '2' else 'accel x | accel y | strain'}); indicators and edges as before")
    print(f"experiment {exp} ({bw.ef.LABEL[exp]}); steps {steps}; variants {variants}; weight decay {m.WEIGHT_DECAY} (ON); "
          f"seeds {m.SEEDS}; lambdas ({m.LAMBDA1}, {m.LAMBDA2})")
    m.report = lambda res: None
    for s in steps:
        removed, node_keep, edge_keep = bw.kept_features(s)
        print("\n" + "#" * 100 + f"\n# STEP {s} (per-sensor standardisation): removed so far {removed or 'nothing'}\n" + "#" * 100)
        m.build_dataset = bw.make_builder(exp, s)
        m.EDGE_ATTR_DIM = len(edge_keep)
        m.DATASET_LABEL = f"{bw.ef.LABEL[exp]}, backward step {s}, per-sensor standardisation"
        m.CSV_PATH = bw.csv_path(exp, s)
        m.VARIANTS_TO_RUN = variants
        m.main()
    bw.curve(exp, list(range(bw.MAX_STEP + 1)), variants)
 
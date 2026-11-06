
"""
Removes the linear temperature trend from every feature and trains the same models, to compare this common correction with feature removal.

Needs: compare_vanilla_cal_backward.py.
"""
from __future__ import annotations
import os
import sys
 
import numpy as np
import torch
 
import os as _os, sys as _sys
 
_sys.path.insert(0, _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "core")))
 
import compare_vanilla_cal_backward as bw
 
m = bw.m
FIT = os.environ.get("TEMP_REG_FIT", "healthy")
USE_LOG = os.environ.get("TEMP_REG_LOG", "0") == "1"
assert FIT in ("healthy", "all"), "TEMP_REG_FIT must be healthy or all"
bw.TAG = "n186_tempreg" + ("_log" if USE_LOG else "") + ("_fitall" if FIT == "all" else "")
N_IND = 3                                           # sensor-type indicators: always the last 3 columns of x
STATE = {"healthy": None, "printed": False}
_orig_fit, _orig_apply = m.fit_standardizer, m.apply_standardizer
 
 
def feature_vector(g, n_resp):
    """Node response columns and all edge columns of one graph as one flat float64 vector."""
    return torch.cat([g.x[:, :n_resp].reshape(-1), g.edge_attr.reshape(-1)]).double()
 
 
def fit_tempreg(graphs):
    """Fit a linear trend with temperature for every feature of the training graphs."""
    n_resp = graphs[0].x.shape[1] - N_IND
    V = torch.stack([feature_vector(g, n_resp) for g in graphs])          # (G, J)
    T = torch.tensor([g.temp.item() for g in graphs], dtype=torch.float64)
    is_h = torch.tensor([g.y.item() == STATE["healthy"] for g in graphs])
    use = is_h if FIT == "healthy" else torch.ones_like(is_h)
    if int(use.sum()) < 5:                                                  # too few Healthy sessions: use all
        print(f"  warning: only {int(use.sum())} Healthy training graphs, fitting the trend on all training graphs")
        use = torch.ones_like(is_h)
    pos = (V.min(0).values > 0) if USE_LOG else torch.zeros(V.shape[1], dtype=torch.bool)
    Vt = torch.where(pos, torch.log10(V.clamp_min(1e-12)), V)
    Tu, Vu = T[use], Vt[use]
    Tc = Tu - Tu.mean()
    denom = (Tc ** 2).sum()
    slope = ((Tc[:, None] * (Vu - Vu.mean(0))).sum(0) / denom) if denom > 1e-12 else torch.zeros(V.shape[1], dtype=torch.float64)
    icpt = Vu.mean(0) - slope * Tu.mean()
    if not STATE["printed"]:
        STATE["printed"] = True
        print(f"  temperature regression: {int(use.sum())} of {len(graphs)} training graphs used for the fit, "
              f"{int(pos.sum())} of {V.shape[1]} columns log10-transformed, temperature range of the fit "
              f"{Tu.min():.1f} to {Tu.max():.1f} degC")
    return slope, icpt, pos, n_resp
 
 
def apply_tempreg(graphs, reg):
    """Replace every feature value by its residual from the temperature trend."""
    slope, icpt, pos, n_resp = reg
    n_node = graphs[0].x.shape[0] * n_resp
    out = []
    for g in graphs:
        v = feature_vector(g, n_resp)
        v = torch.where(pos, torch.log10(v.clamp_min(1e-12)), v)
        r = (v - (slope * g.temp.item() + icpt)).float()
        g2 = g.clone()
        x = g.x.clone()
        x[:, :n_resp] = r[:n_node].reshape(g.x.shape[0], n_resp)
        g2.x = x
        g2.edge_attr = r[n_node:].reshape(g.edge_attr.shape)
        out.append(g2)
    return out
 
 
def fit_standardizer(graphs):
    """Fit the temperature regression and then the standardiser on the residuals."""
    reg = fit_tempreg(graphs)
    return reg, _orig_fit(apply_tempreg(graphs, reg))
 
 
def apply_standardizer(graphs, st):
    """Remove the temperature trend and then standardise."""
    reg, std = st
    return _orig_apply(apply_tempreg(graphs, reg), std)
 
 
def make_builder(exp, step):
    """Return the step builder with a check that the graphs support per-edge regression."""
    inner = bw.make_builder(exp, step)
 
    def build(df, **kw):
        """Build the graphs and check the Healthy class, edge index and temperature attribute."""
        graphs, enc = inner(df, **kw)
        ids = [i for i, c in enumerate(enc.classes_) if str(c).lower().startswith("healthy")]
        assert len(ids) == 1, f"expected one Healthy class, found {[str(c) for c in enc.classes_]}"
        STATE["healthy"] = ids[0]
        e0 = graphs[0].edge_index
        assert all(g.edge_index.shape == e0.shape and torch.equal(g.edge_index, e0) for g in graphs), \
            "edge_index differs between graphs: per-edge regression needs identical edge positions"
        assert all(hasattr(g, "temp") for g in graphs), "graphs have no .temp attribute"
        return graphs, enc
    return build
 
 
if __name__ == "__main__":
    exp, steps, variants = bw.parse(sys.argv[1:])
    assert hasattr(m, "EDGE_ATTR_DIM"), "compare_vanilla_cal.py must take the edge width from a module-level EDGE_ATTR_DIM"
    m.fit_standardizer = fit_standardizer
    m.apply_standardizer = apply_standardizer
    print(f"TEMPERATURE-REGRESSION baseline: linear trend fitted on {'Healthy' if FIT == 'healthy' else 'all'} "
          f"training graphs, log transform {'ON' if USE_LOG else 'off'}; then pooled standardisation")
    print(f"experiment {exp} ({bw.ef.LABEL[exp]}); steps {steps}; variants {variants}; weight decay {m.WEIGHT_DECAY} (ON); "
          f"seeds {m.SEEDS}; lambdas ({m.LAMBDA1}, {m.LAMBDA2})")
    m.report = lambda res: None
    for s in steps:
        removed, node_keep, edge_keep = bw.kept_features(s)
        print("\n" + "#" * 100 + f"\n# STEP {s} (temperature-regressed features): removed so far {removed or 'nothing'}\n" + "#" * 100)
        STATE["printed"] = False
        m.build_dataset = make_builder(exp, s)
        m.EDGE_ATTR_DIM = len(edge_keep)
        m.DATASET_LABEL = f"{bw.ef.LABEL[exp]}, backward step {s}, temperature regression"
        m.CSV_PATH = bw.csv_path(exp, s)
        m.VARIANTS_TO_RUN = variants
        m.main()
    bw.curve(exp, list(range(bw.MAX_STEP + 1)), variants)
 
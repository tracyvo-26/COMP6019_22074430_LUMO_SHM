"""
Checks whether the intervention term of CAL has any effect, by measuring how much the shortcut representation differs between temperature strata.

Needs: compare_vanilla_cal_backward.py.
"""
from __future__ import annotations
import os
import sys
 
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
 
import os as _os, sys as _sys
 
_sys.path.insert(0, _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "core")))
 
import compare_vanilla_cal_backward as bw
 
m = bw.m
args = sys.argv[1:]
INTERV = "random" if "random" in args else "stratum"
STEP = 3 if "step3" in args else 0
VARIANTS = ["cal_both", "nocausal_both"]
if "variants" in args:
    VARIANTS = args[args.index("variants") + 1:]
m.INTERVENTION = INTERV
m.WEIGHT_DECAY = float(os.environ.get("PAPER_WD", m.WEIGHT_DECAY))     # PAPER_WD=0 -> weight decay 0 as in the reference code
 
 
def train_and_diagnose(cfg, graphs, seed, n_cls):
    """Train one model and measure how much its shortcut representation differs between temperature strata."""
    st = m.fit_standardizer(graphs)
    gs = m.apply_standardizer(graphs, st)
    gs = m.rebin(gs, m.NUM_BINS)
    torch.manual_seed(seed)
    np.random.seed(seed)
    node_dim = gs[0].x.shape[1]
    model = m.build_model(cfg, node_dim, n_cls).to(m.DEVICE)
    opt = m.make_optimizer(model, cfg)
    batch = Batch.from_data_list(gs).to(m.DEVICE)
    l1, l2 = (m.LAMBDA1, m.LAMBDA2) if cfg["causal"] else (0.0, 0.0)
    stats = {}
    for _ in range(m.EPOCHS):
        model.train()
        opt.zero_grad()
        loss, stats = m.compute_loss(model, batch, cfg, l1, l2, None)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        g_c, g_t = out["g_c"], out["g_t"]
        eov = batch.eov_bin.view(-1)
        strata = torch.unique(eov)
        gbars = torch.stack([g_t[eov == t].mean(0) for t in strata])             # (n_strata, hid)
        spread = gbars.std(0).mean().item() if len(strata) > 1 else 0.0
        p0 = F.softmax(model.clf_c(g_c), 1)
        dp = []
        for t, gb in zip(strata, gbars):
            sel = eov == t
            p1 = F.softmax(model.clf_c(g_c[sel] + gb), 1)
            dp.append((p1 - p0[sel]).abs().sum(1))
        dp = torch.cat(dp).mean().item()
        gt_norm, gc_norm = g_t.norm(dim=1).mean().item(), g_c.norm(dim=1).mean().item()
        an, ae = out["att_node"], out["att_edge"]            # attention given to the causal branch (None if off)
        att = dict(att_node_mean=an.mean().item() if an is not None else np.nan,
                   att_node_sat=(an > 0.99).float().mean().item() if an is not None else np.nan,
                   att_edge_mean=ae.mean().item() if ae is not None else np.nan,
                   att_edge_sat=(ae > 0.99).float().mean().item() if ae is not None else np.nan)
    return dict(**att,gt_norm=gt_norm, gc_norm=gc_norm, gt_over_gc=gt_norm / max(gc_norm, 1e-12),
                gbar_spread=spread, spread_rel=spread / max(g_c.abs().mean().item(), 1e-12), dp_interv=dp,
                l_caus_minus_l_sup=stats.get("l_caus", np.nan) - stats.get("l_sup", np.nan))
 
 
if __name__ == "__main__":
    removed, node_keep, edge_keep = bw.kept_features(STEP)
    m.build_dataset = bw.make_builder("A", STEP)
    m.EDGE_ATTR_DIM = len(edge_keep)
    df = m.load_merged_data()
    graphs, enc = m.build_dataset(df, num_eov_bins=m.NUM_BINS)
    n_cls = len(enc.classes_)
    print(f"intervention {INTERV}; step {STEP} (removed {removed or 'nothing'}); {len(graphs)} graphs; "
          f"variants {VARIANTS}; seeds {m.SEEDS}; epochs {m.EPOCHS}\n")
    rows = []
    for v in VARIANTS:
        assert v in m.VARIANTS, f"unknown variant {v}"
        for seed in m.SEEDS:
            r = train_and_diagnose(m.VARIANTS[v], graphs, seed, n_cls)
            rows.append(dict(variant=v, seed=seed, intervention=INTERV, step=STEP, **r))
            print(f"{v:<14} seed {seed:>4} | gt/gc {r['gt_over_gc']:.2e} | spread_rel {r['spread_rel']:.2e} | "
                  f"dp_interv {r['dp_interv']:.2e} | L_caus-L_sup {r['l_caus_minus_l_sup']:+.2e} | "
                  f"att node {r['att_node_mean']:.3f} (>0.99: {r['att_node_sat']:.2f}) "
                  f"edge {r['att_edge_mean']:.3f} (>0.99: {r['att_edge_sat']:.2f})")
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\nMean over seeds:")
    print(res.groupby("variant")[["gt_norm", "gc_norm", "gt_over_gc", "gbar_spread", "spread_rel", "dp_interv",
                                  "l_caus_minus_l_sup", "att_node_mean", "att_node_sat", "att_edge_mean",
                                  "att_edge_sat"]].mean().to_string(float_format=lambda x: f"{x:.3e}"))
    print("\natt_*_mean = mean attention given to the CAUSAL branch; att_*_sat = fraction of nodes / edges with causal "
          "attention above 0.99 (1.0 => the trivial branch receives no input).")
    out = os.path.join(bw.HERE, f"diagnose_gbar_spread_{INTERV}_step{STEP}.csv")
    res.to_csv(out, index=False)
    print(f"saved {out}")
 
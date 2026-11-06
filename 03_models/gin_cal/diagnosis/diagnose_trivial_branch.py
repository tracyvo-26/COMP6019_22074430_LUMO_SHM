"""
Checks what the shortcut branch contains after training, and whether weight decay empties it.

Needs: compare_vanilla_cal.py and graph_builder.py.
"""
from __future__ import annotations
import importlib.util
import os
import sys
import time
from types import SimpleNamespace
 
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Batch
 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "core"))                      # shared modules (compare_vanilla_cal*.py)
HERE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "results"))   # all CSV results of gin_cal live here
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)
spec = importlib.util.spec_from_file_location("cvc", os.path.join(CORE_DIR, "compare_vanilla_cal.py"))
m = importlib.util.module_from_spec(spec)
sys.modules["cvc"] = m
spec.loader.exec_module(m)
 
# (name, base variant in compare_vanilla_cal.py, weight decay on the trivial branch?)
CONFIGS = [
    ("cal_node",          "cal_node",      True),
    ("cal_edge",          "cal_edge",      True),
    ("cal_both",          "cal_both",      True),
    ("nocausal_edge",     "nocausal_edge", True),
    ("cal_edge_nowd",     "cal_edge",      False),
    ("nocausal_edge_nowd", "nocausal_edge", False),
    ("cal_both_nowd",     "cal_both",      False),
]
EXPERIMENT = "A"
EXCLUDED = ["DAM3_010", "DAM4_010"]
SPATIAL = dict(USE_AMP=True, USE_CURV=True, USE_DET=True)
SEEDS = list(m.SEEDS)
_orig_build = m.build_dataset
 
 
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
 
 
# column positions in graph.x / graph.edge_attr (graph_builder order)
NODE_KEEP = [2, 3, 4, 5]                 # freq_hz, is_accel_x, is_accel_y, is_strain
EDGE_KEEP = [0, 1, 2, 3, 5, 6]           # xcov, coh_b1, coh_t1, coh_b2, trans_t1, trans_b2 (trans_b1 = col 4 dropped)
 
 
def select_features(graphs, enc):
    """Keep only the selected node and edge feature columns of every graph."""
    out = []
    for g in graphs:
        g2 = g.clone()
        g2.x = g.x[:, NODE_KEEP].contiguous()
        g2.edge_attr = g.edge_attr[:, EDGE_KEEP].contiguous()
        out.append(g2)
    return out, enc
 
 
def load_graphs(exp):
    """Build the graphs of the chosen experiment."""
    df = m.load_merged_data()
    if exp == "A":
        return _orig_build(df, num_eov_bins=m.NUM_BINS)
    if exp == "S3":
        g, e = _orig_build(df, num_eov_bins=m.NUM_BINS)
        return select_features(g, e)
    if exp == "B":
        g, e = _orig_build(df, num_eov_bins=m.NUM_BINS)
        return exclude_classes(g, e, EXCLUDED)
    if exp == "C":
        import graph_builder as gb
        gb.USE_AMP, gb.USE_CURV, gb.USE_DET = SPATIAL["USE_AMP"], SPATIAL["USE_CURV"], SPATIAL["USE_DET"]
        gb.USE_B2Y = False
        return gb.build_dataset(gb.merge_spatial(df), use_spatial=True, num_eov_bins=m.NUM_BINS)
    raise ValueError(exp)
 
 
def make_optimizer(model, trivial_wd):
    """Create the Adam optimiser, with or without weight decay on the shortcut branch."""
    if trivial_wd:
        return torch.optim.Adam(model.parameters(), lr=m.LEARNING_RATE, weight_decay=m.WEIGHT_DECAY)
    triv = [p for mod in (model.conv_t, model.clf_t) for p in mod.parameters()]
    ids = {id(p) for p in triv}
    rest = [p for p in model.parameters() if id(p) not in ids]
    return torch.optim.Adam([{"params": rest, "weight_decay": m.WEIGHT_DECAY},
                             {"params": triv, "weight_decay": 0.0}], lr=m.LEARNING_RATE)
 
 
def train(cfg, trivial_wd, train_graphs, seed, node_dim, n_cls):
    """Same recipe as compare_vanilla_cal.fit_and_eval; returns the model and last-epoch loss values."""
    st = m.fit_standardizer(train_graphs) if m.STANDARDIZE else None
    tr = m.apply_standardizer(train_graphs, st) if st else train_graphs
    tr = m.rebin(tr, m.NUM_BINS)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = m.build_model(cfg, node_dim, n_cls).to(m.DEVICE)
    opt = make_optimizer(model, trivial_wd)
    batch = Batch.from_data_list(tr).to(m.DEVICE)
    l1, l2 = (m.LAMBDA1, m.LAMBDA2) if cfg["causal"] else (0.0, 0.0)
    stats = {}
    for _ in range(m.EPOCHS):
        model.train()
        opt.zero_grad()
        loss, stats = m.compute_loss(model, batch, cfg, l1, l2, None)
        loss.backward()
        opt.step()
    return model, st, tr, stats
 
 
@torch.no_grad()
def reps(model, graphs):
    """Return the representations, attention values, temperature and labels of a set of graphs."""
    model.eval()
    b = Batch.from_data_list(graphs).to(m.DEVICE)
    out = model(b.x, b.edge_index, b.edge_attr, b.batch)
    raw = np.stack([np.concatenate([g.x.mean(0).numpy(), g.edge_attr.mean(0).numpy()]) for g in graphs])
    return dict(g_c=out["g_c"].cpu().numpy(), g_t=out["g_t"].cpu().numpy(),
                att_node=None if out["att_node"] is None else float(out["att_node"].mean()),
                att_edge=None if out["att_edge"] is None else float(out["att_edge"].mean()),
                temp=np.array([g.temp.item() for g in graphs]), y=b.y.view(-1).cpu().numpy(), raw=raw)
 
 
def zscore(a, b):
    """Standardise the test matrix with the mean and standard deviation of the training matrix."""
    mu, sd = a.mean(0), a.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)          # truly constant columns stay at 0
    return (a - mu) / sd, (b - mu) / sd
 
 
def probe_temp(Xtr, ttr, Xte, tte):
    """R2 of a ridge regression that predicts temperature from a representation."""
    A, B = zscore(Xtr, Xte)
    return r2_score(tte, Ridge(alpha=10.0).fit(A, ttr).predict(B))
 
 
def probe_label(Xtr, ytr, Xte, yte):
    """Accuracy of a logistic regression that predicts the damage label from a representation."""
    A, B = zscore(Xtr, Xte)
    if len(np.unique(ytr)) < 2:
        return float("nan")
    return accuracy_score(yte, LogisticRegression(max_iter=3000).fit(A, ytr).predict(B))
 
 
def main(exp):
    """Train the CAL models with and without weight decay on the shortcut branch and probe its representation."""
    t0 = time.time()
    graphs, enc = load_graphs(exp)
    print(f"Loaded {len(graphs)} graphs")
    m.EDGE_ATTR_DIM = graphs[0].edge_attr.shape[1]       # the models read the edge width from this module variable
    node_dim, n_cls = graphs[0].x.shape[1], len(enc.classes_)
    labels = [g.y.item() for g in graphs]
    out_csv = os.path.join(HERE, f"diagnose_trivial_branch_{exp}_by_seed.csv")
    print(f"experiment {exp}: {len(graphs)} graphs, {n_cls} classes, node dim {node_dim}; seeds {SEEDS}; "
          f"epochs {m.EPOCHS}; lambdas ({m.LAMBDA1}, {m.LAMBDA2}); device {m.DEVICE}")
    print("configs:", [c[0] for c in CONFIGS], "\n")
 
    rows = []
    for name, base, trivial_wd in CONFIGS:
        cfg = m.VARIANTS[base]
        for seed in SEEDS:
            tr_idx, te_idx = train_test_split(range(len(graphs)), test_size=m.TEST_SIZE,
                                              stratify=labels, random_state=seed)
            tr_g, te_g = [graphs[i] for i in tr_idx], [graphs[i] for i in te_idx]
            model, st, tr_std, stats = train(cfg, trivial_wd, tr_g, seed, node_dim, n_cls)
            te_std = m.apply_standardizer(te_g, st) if st else te_g
            R, T = reps(model, tr_std), reps(model, te_std)
            maj = np.bincount(R["y"]).argmax()
            nc, nt = np.linalg.norm(T["g_c"], axis=1).mean(), np.linalg.norm(T["g_t"], axis=1).mean()
            row = dict(config=name, seed=seed, norm_gc=nc, norm_gt=nt, ratio_gt_gc=nt / nc,
                       att_node=np.nan if T["att_node"] is None else T["att_node"],
                       att_edge=np.nan if T["att_edge"] is None else T["att_edge"],
                       l_sup=stats.get("l_sup", np.nan), l_unif=stats.get("l_unif", np.nan),
                       l_caus_minus_l_sup=stats.get("l_caus", np.nan) - stats.get("l_sup", np.nan),
                       r2_temp_gc=probe_temp(R["g_c"], R["temp"], T["g_c"], T["temp"]),
                       r2_temp_gt=probe_temp(R["g_t"], R["temp"], T["g_t"], T["temp"]),
                       r2_temp_raw=probe_temp(R["raw"], R["temp"], T["raw"], T["temp"]),
                       acc_label_gc=probe_label(R["g_c"], R["y"], T["g_c"], T["y"]),
                       acc_label_gt=probe_label(R["g_t"], R["y"], T["g_t"], T["y"]),
                       acc_label_raw=probe_label(R["raw"], R["y"], T["raw"], T["y"]),
                       majority_rate=float((T["y"] == maj).mean()))
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"{name:<19} seed {seed:>4} | |g_c| {nc:8.2f} |g_t| {nt:8.4f} (ratio {nt / nc:.1e}) | "
                  f"R2 temp g_c {row['r2_temp_gc']:+.2f} g_t {row['r2_temp_gt']:+.2f} | "
                  f"label acc g_c {row['acc_label_gc']:.2f} g_t {row['acc_label_gt']:.2f} "
                  f"(majority {row['majority_rate']:.2f})")
 
    res = pd.DataFrame(rows)
    order = [c[0] for c in CONFIGS]
    cols = ["norm_gc", "norm_gt", "ratio_gt_gc", "att_node", "att_edge", "l_unif", "l_caus_minus_l_sup",
            "r2_temp_gc", "r2_temp_gt", "r2_temp_raw", "acc_label_gc", "acc_label_gt", "acc_label_raw",
            "majority_rate"]
    pd.set_option("display.width", 260)
    pd.set_option("display.float_format", lambda x: f"{x:.4g}")
    print("\n" + "=" * 120)
    print(f"TRIVIAL-BRANCH DIAGNOSTICS, experiment {exp}, mean over {len(SEEDS)} seeds (random split, test graphs)")
    print("=" * 120)
    print(res.groupby("config")[cols].mean().loc[order].to_string())
    print("\nSD over seeds:")
    print(res.groupby("config")[cols].std().loc[order].to_string())
    print(f"\nPer-seed results: {out_csv} | {(time.time() - t0) / 60:.1f} min")
 
 
if __name__ == "__main__":
    exp = sys.argv[1] if len(sys.argv) > 1 else EXPERIMENT
    assert exp in ("A", "S3", "B", "C"), "experiment must be A, S3, B or C"
    main(exp)
 
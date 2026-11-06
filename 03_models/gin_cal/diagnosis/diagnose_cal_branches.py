"""
Looks inside the causal and shortcut branches of the trained CAL models to see what each one contains.

Needs: compare_vanilla_cal.py.
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
 
EXCLUDED: list = []                       # e.g. ["DAM3_010", "DAM4_010"]
VARIANTS = ["cal_node", "cal_edge", "cal_both", "nocausal_node", "nocausal_edge", "nocausal_both"]
SEEDS = list(m.SEEDS)
CSV_OUT = os.path.join(HERE, "diagnose_cal_branches_by_seed.csv")
 
 
def exclude_classes(graphs, enc, excluded):
    """Remove the graphs of the excluded classes and relabel the remaining classes."""
    names = list(enc.classes_)
    keep = [c for c in names if c not in excluded]
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
 
 
def train_model(cfg, train_graphs, seed, node_dim, n_cls):
    """Same recipe as compare_vanilla_cal.fit_and_eval, but returns the trained model."""
    st = m.fit_standardizer(train_graphs) if m.STANDARDIZE else None
    tr = m.apply_standardizer(train_graphs, st) if st else train_graphs
    tr = m.rebin(tr, m.NUM_BINS)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = m.build_model(cfg, node_dim, n_cls).to(m.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=m.LEARNING_RATE, weight_decay=m.WEIGHT_DECAY)
    batch = Batch.from_data_list(tr).to(m.DEVICE)
    l1, l2 = (m.LAMBDA1, m.LAMBDA2) if cfg["causal"] else (0.0, 0.0)
    for _ in range(m.EPOCHS):
        model.train()
        opt.zero_grad()
        loss, _ = m.compute_loss(model, batch, cfg, l1, l2, None)
        loss.backward()
        opt.step()
    return model, st, tr
 
 
@torch.no_grad()
def get_reps(model, graphs):
    """Return the causal and shortcut representations, attention values and temperature of a set of graphs."""
    model.eval()
    b = Batch.from_data_list(graphs).to(m.DEVICE)
    out = model(b.x, b.edge_index, b.edge_attr, b.batch)
    temps = np.array([g.temp.item() for g in graphs])
    y = b.y.view(-1).cpu().numpy()
    raw = np.stack([np.concatenate([g.x.mean(0).numpy(), g.edge_attr.mean(0).numpy()]) for g in graphs])
    return {"g_c": out["g_c"].cpu().numpy(), "g_t": out["g_t"].cpu().numpy(),
            "att_node": None if out["att_node"] is None else out["att_node"].cpu().numpy().ravel(),
            "att_edge": None if out["att_edge"] is None else out["att_edge"].cpu().numpy().ravel(),
            "temp": temps, "y": y, "raw": raw}
 
 
def zscore(Xtr, Xte):
    """Standardise the test matrix with the mean and standard deviation of the training matrix."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)          # truly constant features stay at 0
    return (Xtr - mu) / sd, (Xte - mu) / sd
 
 
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
 
 
def main():
    """Train the CAL models and probe what the causal and shortcut representations contain."""
    t0 = time.time()
    df = m.load_merged_data()
    graphs, enc = m.build_dataset(df, num_eov_bins=m.NUM_BINS)
    if EXCLUDED:
        graphs, enc = exclude_classes(graphs, enc, EXCLUDED)
    node_dim, n_cls = graphs[0].x.shape[1], len(enc.classes_)
    labels = [g.y.item() for g in graphs]
    print(f"{len(graphs)} graphs, {n_cls} classes (excluded: {EXCLUDED}); seeds {SEEDS}; "
          f"epochs {m.EPOCHS}; device {m.DEVICE}\n")
 
    rows = []
    for variant in VARIANTS:
        cfg = m.VARIANTS[variant]
        for seed in SEEDS:
            tr_idx, te_idx = train_test_split(range(len(graphs)), test_size=m.TEST_SIZE,
                                              stratify=labels, random_state=seed)
            tr_g, te_g = [graphs[i] for i in tr_idx], [graphs[i] for i in te_idx]
            model, st, tr_std = train_model(cfg, tr_g, seed, node_dim, n_cls)
            te_std = m.apply_standardizer(te_g, st) if st else te_g
            R, T = get_reps(model, tr_std), get_reps(model, te_std)
            maj = np.bincount(R["y"]).argmax()
            row = dict(variant=variant, seed=seed,
                       norm_gc=np.linalg.norm(T["g_c"], axis=1).mean(),
                       norm_gt=np.linalg.norm(T["g_t"], axis=1).mean(),
                       att_node_std=np.nan if T["att_node"] is None else float(T["att_node"].std()),
                       att_edge_std=np.nan if T["att_edge"] is None else float(T["att_edge"].std()),
                       r2_temp_gc=probe_temp(R["g_c"], R["temp"], T["g_c"], T["temp"]),
                       r2_temp_gt=probe_temp(R["g_t"], R["temp"], T["g_t"], T["temp"]),
                       r2_temp_raw=probe_temp(R["raw"], R["temp"], T["raw"], T["temp"]),
                       acc_label_gc=probe_label(R["g_c"], R["y"], T["g_c"], T["y"]),
                       acc_label_gt=probe_label(R["g_t"], R["y"], T["g_t"], T["y"]),
                       acc_label_raw=probe_label(R["raw"], R["y"], T["raw"], T["y"]),
                       majority_rate=float((T["y"] == maj).mean()))
            rows.append(row)
            print(f"{variant:<14} seed {seed:>4} | |g_c| {row['norm_gc']:.3f} |g_t| {row['norm_gt']:.3f} | "
                  f"R2 temp g_c {row['r2_temp_gc']:+.2f} g_t {row['r2_temp_gt']:+.2f} | "
                  f"label acc g_c {row['acc_label_gc']:.2f} g_t {row['acc_label_gt']:.2f} "
                  f"(majority {row['majority_rate']:.2f})")
            pd.DataFrame(rows).to_csv(CSV_OUT, index=False)
 
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    cols = ["norm_gc", "norm_gt", "att_node_std", "att_edge_std", "r2_temp_gc", "r2_temp_gt",
            "r2_temp_raw", "acc_label_gc", "acc_label_gt", "acc_label_raw", "majority_rate"]
    print("\n" + "=" * 110)
    print(f"BRANCH DIAGNOSTICS, mean over {len(SEEDS)} seeds (random split, test graphs; probes fitted on training graphs)")
    print("=" * 110)
    print(res.groupby("variant")[cols].mean().loc[VARIANTS].round(3).to_string())
    print("\nSD over seeds:")
    print(res.groupby("variant")[cols].std().loc[VARIANTS].round(3).to_string())
    print(f"\nPer-seed results: {CSV_OUT} | {(time.time() - t0) / 60:.1f} min") 
if __name__ == "__main__":
    main()
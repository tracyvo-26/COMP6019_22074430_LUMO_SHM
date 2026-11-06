"""
Sensor ablation (Experiment 3): keeps only the top k accelerometers of each ranking, retrains GIN on the smaller graph and records the cross-environment performance.

Needs: classical_osp_rankings.csv (from classical_osp_baselines.py); compare_vanilla_cal_backward.py.
"""
from __future__ import annotations
import glob
import os
import re
import sys
 
import numpy as np
import pandas as pd
 
HERE = os.path.dirname(os.path.abspath(__file__))
GIN_CAL_DIR = os.environ.get("GIN_CAL_DIR", os.path.join(HERE, "..", "..", "03_models", "gin_cal", "core"))
RANKINGS_CSV = os.environ.get("RANKINGS_CSV", os.path.join(HERE, "..", "classical_osp", "classical_osp_rankings.csv"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(HERE, "results"))
RANDOM_DRAWS = int(os.environ.get("RANDOM_DRAWS", "10"))
NODE_LAYOUT = os.environ.get("NODE_LAYOUT", "x_then_y")      # nodes 0-8 = x chain, 9-17 = y chain, 18-20 = strain
KS = [6, 9, 12, 15, 18]
EXP, STEP, VARIANTS = "A", 3, ["vanilla"]
CLASSICAL = ["mke", "efi", "entropy"]
N_ACCEL, N_STRAIN = 18, 3
 
sys.path.insert(0, GIN_CAL_DIR)
 
 
def node_order(layout: str):
    """Channel name of every graph node, in graph order (strain nodes last)."""
    if layout == "interleaved":          # accel01x, accel01y, accel02x, accel02y, ...
        acc = [f"accel{l:02d}{d}" for l in range(1, 10) for d in "xy"]
    elif layout == "x_then_y":           # accel01x ... accel09x, accel01y ... accel09y
        acc = [f"accel{l:02d}x" for l in range(1, 10)] + [f"accel{l:02d}y" for l in range(1, 10)]
    else:
        raise ValueError("NODE_LAYOUT must be interleaved or x_then_y")
    return acc + [f"strain{i}" for i in range(1, N_STRAIN + 1)]
 
 
def load_rankings():
    """Load the sensor rankings of the three classical methods from the rankings file."""
    r = pd.read_csv(RANKINGS_CSV)
    return {
        "mke": list(r.sort_values("mke_rank")["channel"]),
        "efi": list(r.sort_values(["efi_rank", "channel"])["channel"]),      # ties (noise-level) broken by channel name
        "entropy": list(r.sort_values("entropy_rank")["channel"]),
    }
 
 
def random_ranking(draw: int, channels):
    """Return a random ordering of the channels, reproducible by draw number."""
    rng = np.random.default_rng(1000 + draw)
    return list(rng.permutation(channels))
 
 
def configs(which: str):
    """(ranking, k, draw) for every run; k = 18 is the same for all rankings, so it is run once as 'full'."""
    out = [("full", 18, 0)]
    if which in ("classical", "all"):
        out += [(r, k, 0) for r in CLASSICAL for k in KS[:-1]]
    if which in ("random", "all"):
        out += [("random", k, d) for d in range(RANDOM_DRAWS) for k in KS[:-1]]
    if which in ("gin", "all"):
        out += [("gin", k, 0) for k in KS[:-1]]
    return out
 
 
def kept_channels(ranking, k, draw, ranks, channels):
    """Return the channels kept for a ranking and number of sensors k."""
    if ranking == "full":
        return list(channels)
    if ranking == "gin":
        return [f"top-{k} by GIN saliency (chosen per split)"]
    order = random_ranking(draw, channels) if ranking == "random" else ranks[ranking]
    return list(order[:k])
 
 
def tag_of(ranking, k, draw):
    """Short tag identifying a ranking, k and random draw."""
    return f"{ranking}_k{k}" + (f"_d{draw}" if ranking == "random" else "")
 
 
def csv_path(ranking, k, draw):
    """Path of the result file of one ranking, k and random draw."""
    return os.path.join(OUT_DIR, f"sensors_{tag_of(ranking, k, draw)}_step{STEP}_by_seed.csv")
 
 
def subset_graph(g, keep_idx):
    """Keeps the nodes in keep_idx (sorted), relabels them, and drops every edge that touches a removed node."""
    n = g.x.shape[0]
    keep = sorted(keep_idx)
    new_id = -np.ones(n, dtype=np.int64)
    new_id[keep] = np.arange(len(keep))
    ei = g.edge_index.cpu().numpy()
    e_keep = (new_id[ei[0]] >= 0) & (new_id[ei[1]] >= 0)
    g2 = g.clone()
    import torch
    for key, value in g:
        if key in ("edge_index", "y") or not torch.is_tensor(value):
            continue
        if key == "edge_attr":
            g2[key] = value[torch.as_tensor(e_keep)]
        elif value.dim() >= 1 and value.shape[0] == n:
            g2[key] = value[torch.as_tensor(keep)]
    g2.edge_index = torch.as_tensor(new_id[ei[:, e_keep]], dtype=g.edge_index.dtype)
    return g2
 
 
def check_node_types(graphs, names):
    """The last 3 columns of x are the sensor-type indicators: the pattern must agree with the assumed node order."""
    x = graphs[0].x.cpu().numpy()
    types = x[:, -3:].argmax(1)
    expected = [0 if n.endswith("x") else 1 if n.endswith("y") else 2 for n in names]
    # indicator column meaning is not known here, so only the grouping (which nodes share a type) is compared
    same = all((types[i] == types[j]) == (expected[i] == expected[j]) for i in range(len(names)) for j in range(len(names)))
    assert same, "the sensor-type pattern in x does not match NODE_LAYOUT"
 
 
def cmd_inspect():
    """Print the node order, node types and neighbours of the graph."""
    import compare_vanilla_cal_backward as bw
    m = bw.m
    removed, node_keep, edge_keep = bw.kept_features(STEP)
    m.build_dataset = bw.make_builder(EXP, STEP)
    m.EDGE_ATTR_DIM = len(edge_keep)
    df = m.load_merged_data()
    graphs, enc = m.build_dataset(df, num_eov_bins=m.NUM_BINS)
    g = graphs[0]
    ei = g.edge_index.cpu().numpy()
    types = g.x[:, -3:].cpu().numpy().argmax(1)
    print(f"{len(graphs)} graphs; {g.x.shape[0]} nodes; {ei.shape[1]} edges; node feature dim {g.x.shape[1]}")
    print("node index | type column (argmax of the last 3 x columns) | degree | neighbours")
    for i in range(g.x.shape[0]):
        nb = sorted(set(ei[1][ei[0] == i].tolist()))
        print(f"{i:>3} | type {types[i]} | degree {len(nb):>2} | {nb}")
    print("\nAssumed order for NODE_LAYOUT =", NODE_LAYOUT)
    print(", ".join(f"{i}:{n}" for i, n in enumerate(node_order(NODE_LAYOUT))))
 
 
def cmd_plan():
    """Print the sensors kept in every planned run."""
    channels = node_order(NODE_LAYOUT)[:N_ACCEL]
    ranks = load_rankings()
    for ranking, k, draw in configs(sys.argv[2] if len(sys.argv) > 2 else "all"):
        print(f"{tag_of(ranking, k, draw):<22} keeps {len(kept_channels(ranking, k, draw, ranks, channels))} accelerometers: "
              f"{', '.join(kept_channels(ranking, k, draw, ranks, channels))}")
 
 
_RANK_CACHE = {}
 
 
def gin_saliency_ranking(m, train_graphs, seed, names, n_cls):
    """Rank the accelerometer nodes by gradient-times-input saliency of a GIN trained on the training graphs."""
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Batch
    key = (seed, tuple(round(g.temp.item(), 6) for g in train_graphs))
    if key in _RANK_CACHE:
        return _RANK_CACHE[key]
    st = m.fit_standardizer(train_graphs)
    tg = m.apply_standardizer(train_graphs, st)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = m.VARIANTS["vanilla"]
    model = m.build_model(cfg, tg[0].x.shape[1], n_cls).to(m.DEVICE)
    opt = m.make_optimizer(model, cfg)
    b = Batch.from_data_list(tg).to(m.DEVICE)
    for _ in range(m.EPOCHS):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(b.x, b.edge_index, b.edge_attr, b.batch)["logits_c"], b.y.view(-1)).backward()
        opt.step()
    model.eval()
    x = b.x.clone().requires_grad_(True)
    logits = model(x, b.edge_index, b.edge_attr, b.batch)["logits_c"]
    logits.gather(1, b.y.view(-1, 1)).sum().backward()
    sal = (x.grad * x).abs().sum(1).detach().cpu().numpy().reshape(len(tg), -1).mean(0)   # (n_nodes,)
    order = [names[i] for i in np.argsort(-sal[:N_ACCEL], kind="stable")]
    _RANK_CACHE[key] = order
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "gin_rankings_log.csv"), "a") as f:
        f.write(f"{seed},{len(tg)},{'|'.join(order)}\n")
    return order
 
 
def gin_wrapper(m, orig_fit, names, k):
    """Replace fit_and_eval so that the sensors are ranked and reduced on the training graphs before training."""
    def wrapped(cfg, train_graphs, eval_sets, seed, node_dim, n_cls, lam1, lam2):
        """Rank the sensors on the training graphs, keep the top k and train on the reduced graphs."""
        order = gin_saliency_ranking(m, train_graphs, seed, names, n_cls)
        keep_idx = [names.index(c) for c in order[:k]] + [names.index(n) for n in names[N_ACCEL:]]
        tr = [subset_graph(g, keep_idx) for g in train_graphs]
        ev = {nm: [subset_graph(g, keep_idx) for g in gs] for nm, gs in eval_sets.items()}
        return orig_fit(cfg, tr, ev, seed, node_dim, n_cls, lam1, lam2)
    return wrapped
 
 
def cmd_run(which):
    """Run the ablation for the selected configurations and save one result file per run."""
    import compare_vanilla_cal_backward as bw
    m = bw.m
    if os.environ.get("SA_SEEDS", "") == "10":
        m.SEEDS = [42, 7, 123, 2024, 99, 1, 2, 3, 4, 5]
    os.makedirs(OUT_DIR, exist_ok=True)
    names = node_order(NODE_LAYOUT)
    channels = names[:N_ACCEL]
    ranks = load_rankings()
    for key, order in ranks.items():
        assert sorted(order) == sorted(channels), f"channels in {RANKINGS_CSV} ({key}) differ from the node names"
    removed, node_keep, edge_keep = bw.kept_features(STEP)
    m.report = lambda res: None
    todo = configs(which)
    print(f"{len(todo)} configurations; seeds {m.SEEDS}; step {STEP} (removed {removed}); variants {VARIANTS}\n")
    seen_sets = {}      # kept set -> tag of the first run with that set (the result depends on the set, not on the order)
    orig_fit = m.fit_and_eval
    for ranking, k, draw in todo:
        if ranking == "gin":
            # ranking learned inside every split from the TRAINING graphs only (no test leakage); see gin_wrapper
            print("\n" + "#" * 100 + f"\n# {tag_of(ranking, k, draw)}: top-{k} accelerometers by vanilla-GIN saliency "
                  f"(per split, training graphs only) + {N_STRAIN} strain\n" + "#" * 100)
            m.build_dataset = bw.make_builder(EXP, STEP)
            m.EDGE_ATTR_DIM = len(edge_keep)
            m.DATASET_LABEL = f"sensor ablation {tag_of(ranking, k, draw)}, step {STEP}"
            m.CSV_PATH = csv_path(ranking, k, draw)
            m.VARIANTS_TO_RUN = VARIANTS
            m.fit_and_eval = gin_wrapper(m, orig_fit, names, k)
            try:
                m.main()
            finally:
                m.fit_and_eval = orig_fit
            continue
        kept = kept_channels(ranking, k, draw, ranks, channels)
        key = (k, frozenset(kept))
        if ranking != "random" and key in seen_sets and not os.path.exists(csv_path(ranking, k, draw)):
            import shutil
            shutil.copy(csv_path(*seen_sets[key]), csv_path(ranking, k, draw))
            print(f"\n{tag_of(ranking, k, draw)}: same sensor set as {tag_of(*seen_sets[key])}; results copied, no new training")
            continue
        seen_sets.setdefault(key, (ranking, k, draw))
        keep_idx = [names.index(c) for c in kept] + [names.index(n) for n in names[N_ACCEL:]]
        inner = bw.make_builder(EXP, STEP)
 
        def build(df, _keep=keep_idx, _inner=inner, **kw):
            """Build the graphs and keep only the selected sensors."""
            graphs, enc = _inner(df, **kw)
            check_node_types(graphs, names)
            return [subset_graph(g, _keep) for g in graphs], enc
 
        print("\n" + "#" * 100 + f"\n# {tag_of(ranking, k, draw)}: {len(kept)} accelerometers + {N_STRAIN} strain\n"
              f"# kept: {', '.join(kept)}\n" + "#" * 100)
        m.build_dataset = build
        m.EDGE_ATTR_DIM = len(edge_keep)
        m.DATASET_LABEL = f"sensor ablation {tag_of(ranking, k, draw)}, step {STEP}"
        m.CSV_PATH = csv_path(ranking, k, draw)
        m.VARIANTS_TO_RUN = VARIANTS
        m.main()
 
 
def boot_ci(v, n=5000, seed=0):
    """Bootstrap 95% confidence interval of the mean of v."""
    rng = np.random.default_rng(seed)
    v = np.asarray(v, dtype=float)
    idx = rng.integers(0, len(v), (n, len(v)))
    return np.percentile(v[idx].mean(axis=1), [2.5, 97.5])
 
 
def cmd_summarise():
    """Collect all ablation result files and summarise the cross-environment results per ranking and k."""
    rows = []
    for path in sorted(glob.glob(os.path.join(OUT_DIR, f"sensors_*_step{STEP}_by_seed.csv"))):
        mt = re.match(rf"sensors_(\w+?)_k(\d+)(?:_d(\d+))?_step{STEP}_by_seed\.csv", os.path.basename(path))
        if not mt:
            continue
        ranking, k, draw = mt.group(1), int(mt.group(2)), int(mt.group(3) or 0)
        df = pd.read_csv(path)
        d = df[(df.variant == "vanilla") & (df.experiment == "cross")]
        per_seed = d.groupby("seed")[["mismatched_acc", "mismatched_f1"]].mean()
        rows.append(dict(ranking=ranking, k=k, draw=draw, n_seeds=len(per_seed),
                         acc=per_seed.mismatched_acc.mean(), f1=per_seed.mismatched_f1.mean(),
                         f1_lo=boot_ci(per_seed.mismatched_f1)[0] if len(per_seed) > 1 else np.nan,
                         f1_hi=boot_ci(per_seed.mismatched_f1)[1] if len(per_seed) > 1 else np.nan))
    if not rows:
        print("no result files found in", OUT_DIR)
        return
    r = pd.DataFrame(rows)
    out = []
    for (ranking, k), g in r.groupby(["ranking", "k"]):
        if ranking == "random":      # spread over random subsets, not over seeds
            out.append(dict(ranking=ranking, k=k, n_runs=len(g), acc=g.acc.mean(), f1=g.f1.mean(),
                            f1_lo=np.percentile(g.f1, 2.5), f1_hi=np.percentile(g.f1, 97.5)))
        else:
            out.append(dict(ranking=ranking, k=k, n_runs=len(g), acc=g.acc.mean(), f1=g.f1.mean(),
                            f1_lo=g.f1_lo.mean(), f1_hi=g.f1_hi.mean()))
    s = pd.DataFrame(out).sort_values(["ranking", "k"])
    full = s[s.ranking == "full"]
    path = os.path.join(OUT_DIR, "sensor_ablation_summary.csv")
    s.to_csv(path, index=False)
    pd.set_option("display.width", 200)
    print(s.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nsaved {path}")
    print("For random, f1_lo / f1_hi are the 2.5 / 97.5 percentiles over random subsets; for the others they are the "
          "95% bootstrap interval over seeds. 'full' (k = 18) is the reference.")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        f_ref = full.f1.iloc[0] if len(full) else None
        for ranking in ["mke", "efi", "entropy", "gin", "random"]:
            g = s[s.ranking == ranking]
            if g.empty:
                continue
            ks = list(g.k) + ([18] if len(full) else [])
            ys = list(g.f1) + ([f_ref] if len(full) else [])
            ax.plot(ks, ys, marker="o", label=ranking.upper() if ranking != "random" else "random (mean)")
            if ranking == "random":
                ax.fill_between(g.k, g.f1_lo, g.f1_hi, alpha=0.2)
        ax.set_xlabel("number of accelerometers kept (k)")
        ax.set_ylabel("cross-environment macro-F1 (vanilla GIN, step 3)")
        ax.set_xticks(KS)
        ax.legend()
        fig.tight_layout()
        png = os.path.join(OUT_DIR, "sensor_ablation_curve.png")
        fig.savefig(png, dpi=200)
        print(f"saved {png}")
    except ImportError:
        print("matplotlib not installed: no plot saved")
 
 
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plan"
    if cmd == "inspect":
        cmd_inspect()
    elif cmd == "plan":
        cmd_plan()
    elif cmd == "run":
        cmd_run(sys.argv[2] if len(sys.argv) > 2 else "classical")
    elif cmd == "summarise":
        cmd_summarise()
    else:
        print(__doc__)
 
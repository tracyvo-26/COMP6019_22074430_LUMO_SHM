
"""
Runs the original CAL code of Sui et al. on the LUMO graphs (settings 1a, 1b and 1c).

Needs: model.py and gcn_conv.py (03_models/cal_original); compare_vanilla_cal_backward.py.
"""
from __future__ import annotations
import os
import random
import sys
import time
import types
from types import SimpleNamespace
 
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "core"))                      # shared modules (compare_vanilla_cal*.py)
HERE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "results"))   # all CSV results of gin_cal live here
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)
GIN_CAL = CORE_DIR
sys.path.insert(0, os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "cal_original")))   # model.py and gcn_conv.py of the CAL repo
sys.path.append(GIN_CAL)                 # project pipeline
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..")))
 
try:                                     # gcn_conv.py needs torch_scatter.scatter_add; defined here if it is not installed
    import torch_scatter  # noqa: F401
except ImportError:
    shim = types.ModuleType("torch_scatter")
 
    def scatter_add(src, index, dim=0, dim_size=None):
        """Sum the rows of src that share an index (replacement for torch_scatter.scatter_add)."""
        size = list(src.shape)
        size[dim] = int(dim_size) if dim_size is not None else int(index.max()) + 1
        return torch.zeros(size, dtype=src.dtype, device=src.device).index_add_(dim, index, src)
    shim.scatter_add = scatter_add
    sys.modules["torch_scatter"] = shim
 
from torch_geometric.loader import DataLoader  # noqa: E402
import model as ref_model                      # noqa: E402  (the CAL repo's model.py)
import compare_vanilla_cal_backward as bw      # noqa: E402
 
m = bw.m
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QUICK = os.environ.get("REF_QUICK", "0") == "1"
EPOCHS = 3 if QUICK else int(os.environ.get("REF_EPOCHS", "100"))
HIDDEN = int(os.environ.get("REF_HIDDEN", "128"))
LR = float(os.environ.get("REF_LR", "0.001"))
WD = float(os.environ.get("REF_WD", "0"))
BATCH = int(os.environ.get("REF_BATCH", "128"))
E2N_MODE = os.environ.get("REF_EDGE2NODE", "0")         # 0 = off; 1 or mean = mean over incoming edges; row = full edge rows
assert E2N_MODE in ("0", "1", "mean", "row"), "REF_EDGE2NODE must be 0, 1, mean or row"
E2N = E2N_MODE != "0"                                   # give the edge-blind original model the edge information
STD = os.environ.get("REF_STD", "1" if E2N else "0") == "1"   # the aggregated edge features have very different scales
W_C = float(os.environ.get("REF_C", "0.5"))
W_O = float(os.environ.get("REF_O", "1.0"))
W_CO = float(os.environ.get("REF_CO", "0.5"))
SEEDS = m.SEEDS[:1] if QUICK else m.SEEDS
if os.environ.get("REF_SEEDS", "") == "10" and not QUICK:     # 10 seeds: the five default seeds plus five more
    SEEDS = [42, 7, 123, 2024, 99, 1, 2, 3, 4, 5]
ALL_VARIANTS = ["ref_gin", "ref_cal", "ref_cal_nocausal"]
TAG = "refpaper" + (("_e2n_row" if E2N_MODE == "row" else "_e2n") if E2N else "") + ("_quick" if QUICK else "")
 
 
def edge_to_node(g):
    """Append edge features to the node features, as their mean or as all values per source node."""
    ei, ea, n = g.edge_index, g.edge_attr, g.x.shape[0]
    if E2N_MODE == "row":
        full = torch.zeros(n, n, ea.shape[1], dtype=ea.dtype)
        full[ei[1], ei[0], :] = ea                                  # [target node, source node, feature]
        g2 = g.clone()
        g2.x = torch.cat([g.x, full.reshape(n, -1)], dim=1)
        return g2
    s = torch.zeros(n, ea.shape[1], dtype=ea.dtype).index_add_(0, ei[1], ea)
    cnt = torch.zeros(n, dtype=ea.dtype).index_add_(0, ei[1], torch.ones(ei.shape[1], dtype=ea.dtype)).clamp_min(1)
    g2 = g.clone()
    g2.x = torch.cat([g.x, s / cnt[:, None]], dim=1)
    return g2
 
 
def make_args(c, o, co):
    """The arguments model.py reads (names and defaults as in opts.py of the CAL repo)."""
    return SimpleNamespace(layers=3, hidden=HIDDEN, fc_num="222", cat_or_add="add", with_random=True,
                           without_node_attention=False, without_edge_attention=False, c=c, o=o, co=co)
 
 
def build(variant, n_feat, n_cls):
    """Create the original GIN or CAL model with the loss weights of a variant."""
    if variant == "ref_gin":
        return ref_model.GINNet(n_feat, n_cls, HIDDEN), (0.0, 1.0, 0.0)
    c, o, co = (W_C, W_O, W_CO) if variant == "ref_cal" else (0.0, W_O, 0.0)
    return ref_model.CausalGIN(n_feat, n_cls, make_args(c, o, co)), (c, o, co)
 
 
def forward(model, variant, data, eval_random):
    """Return (causal_logprobs, intervened_logprobs, trivial_logprobs) for every variant."""
    if variant == "ref_gin":
        out = model(data)
        return out, out, None
    c_logs, o_logs, co_logs = model(data, eval_random=eval_random)
    return o_logs, co_logs, c_logs
 
 
@torch.no_grad()
def evaluate(model, variant, graphs):
    """Accuracy and macro F1 of the original CAL model on a list of graphs, for the causal and the combined outputs."""
    if len(graphs) == 0:
        return dict(acc=np.nan, f1=np.nan, acc_co=np.nan, f1_co=np.nan)
    model.eval()
    ys, p_o, p_co = [], [], []
    for data in DataLoader(graphs, batch_size=256, shuffle=False):
        data = data.to(DEVICE)
        o_logs, co_logs, _ = forward(model, variant, data, eval_random=False)
        ys.append(data.y.view(-1).cpu().numpy())
        p_o.append(o_logs.argmax(1).cpu().numpy())
        p_co.append(co_logs.argmax(1).cpu().numpy())
    y, p_o, p_co = np.concatenate(ys), np.concatenate(p_o), np.concatenate(p_co)
    lab = np.unique(y)
    return dict(acc=accuracy_score(y, p_o), f1=f1_score(y, p_o, average="macro", labels=lab),
                acc_co=accuracy_score(y, p_co), f1_co=f1_score(y, p_co, average="macro", labels=lab))
 
 
def fit_and_eval(variant, train_graphs, eval_sets, seed, n_feat, n_cls):
    """Train one variant of the original CAL model and evaluate it on every set."""
    if STD:
        st = m.fit_standardizer(train_graphs)
        train_graphs = m.apply_standardizer(train_graphs, st)
        eval_sets = {k: m.apply_standardizer(v, st) for k, v in eval_sets.items()}
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model, (w_c, w_o, w_co) = build(variant, n_feat, n_cls)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    drop_last = len(train_graphs) % BATCH == 1          # BatchNorm cannot train on a batch of one graph
    loader = DataLoader(train_graphs, batch_size=BATCH, shuffle=True, drop_last=drop_last)
    last = {}
    for _ in range(EPOCHS):
        model.train()
        tot = dict(c=0.0, o=0.0, co=0.0)
        n = 0
        for data in loader:
            opt.zero_grad()
            data = data.to(DEVICE)
            y = data.y.view(-1)
            o_logs, co_logs, c_logs = forward(model, variant, data, eval_random=True)
            l_o = F.nll_loss(o_logs, y)
            if variant == "ref_gin":
                loss, l_c, l_co = l_o, torch.zeros(()), torch.zeros(())
            else:
                uniform = torch.ones_like(c_logs) / n_cls
                l_c = F.kl_div(c_logs, uniform, reduction="batchmean")
                l_co = F.nll_loss(co_logs, y)
                loss = w_c * l_c + w_o * l_o + w_co * l_co
            loss.backward()
            opt.step()
            k = data.num_graphs
            tot["c"] += l_c.item() * k
            tot["o"] += l_o.item() * k
            tot["co"] += l_co.item() * k
            n += k
        last = {key: v / n for key, v in tot.items()}
    res = {"train_acc": evaluate(model, variant, train_graphs)["acc"], "l_c": last["c"], "l_o": last["o"],
           "l_co": last["co"], "l_co_minus_l_o": last["co"] - last["o"] if variant != "ref_gin" else np.nan}
    for name, g in eval_sets.items():
        e = evaluate(model, variant, g)
        res.update({f"{name}_acc": e["acc"], f"{name}_f1": e["f1"],
                    f"{name}_acc_co": e["acc_co"], f"{name}_f1_co": e["f1_co"]})
    return res
 
 
def key_of(r):
    """Identify a result row by variant, experiment, threshold, direction and seed."""
    return (r["variant"], r["experiment"], float(r["threshold"]), r["direction"], int(r["seed"]))
 
 
def parse(argv):
    """Read the elimination steps and variants from the command line arguments."""
    steps, variants, mode = [0, 3], list(ALL_VARIANTS), None
    for a in argv:
        if a in ("steps", "variants"):
            mode = a
            if a == "steps":
                steps = []
            else:
                variants = []
        elif mode == "steps":
            steps.append(int(a))
        elif mode == "variants":
            assert a in ALL_VARIANTS, f"unknown variant {a}; choose from {ALL_VARIANTS}"
            variants.append(a)
        else:
            raise SystemExit(f"unexpected argument {a}")
    return steps, variants
 
 
def run_step(step, variants):
    """Run the selected variants of the original CAL model at one elimination step."""
    removed, node_keep, edge_keep = bw.kept_features(step)
    m.build_dataset = bw.make_builder("A", step)
    m.EDGE_ATTR_DIM = len(edge_keep)
    graphs, enc = m.build_dataset(m.load_merged_data(), num_eov_bins=m.NUM_BINS)
    if E2N:
        graphs = [edge_to_node(g) for g in graphs]
    n_feat, n_cls = graphs[0].x.shape[1], len(enc.classes_)
    temps = np.array([g.temp.item() for g in graphs])
    labels = [g.y.item() for g in graphs]
    path = os.path.join(HERE, f"compare_vanilla_cal_backward_A_{TAG}_step{step}_by_seed_3loss.csv")
    rows, done = [], set()
    if os.path.exists(path):
        rows = pd.read_csv(path).to_dict("records")
        done = {key_of(r) for r in rows}
        print(f"resuming: {len(done)} finished runs in {os.path.basename(path)}")
    print(f"\nSTEP {step}: removed {removed or 'nothing'}; node dim {n_feat}; {len(graphs)} graphs, {n_cls} classes; "
          f"epochs {EPOCHS}, hidden {HIDDEN}, lr {LR}, weight decay {WD}, batch {BATCH}, seeds {SEEDS}; device {DEVICE}")
    t0 = time.time()
    for v in variants:
        for seed in SEEDS:                                           # random split
            k = (v, "random", 0.0, "all", seed)
            if k not in done:
                tr_i, te_i = train_test_split(range(len(graphs)), test_size=m.TEST_SIZE, stratify=labels, random_state=seed)
                r = fit_and_eval(v, [graphs[i] for i in tr_i], {"test": [graphs[i] for i in te_i]}, seed, n_feat, n_cls)
                rows.append(dict(variant=v, experiment="random", threshold=0.0, direction="all", seed=seed, **r))
                done.add(k)
                pd.DataFrame(rows).to_csv(path, index=False)
                print(f"[random] {v:<17} seed {seed:>4} | train {r['train_acc']:.3f} | test acc {r['test_acc']:.3f} "
                      f"f1 {r['test_f1']:.3f} | l_c {r['l_c']:.4f} l_o {r['l_o']:.4f} l_co {r['l_co']:.4f}")
        for thr in m.THRESHOLDS_C:                                   # cross-environment
            low = [g for g, t in zip(graphs, temps) if t < thr]
            high = [g for g, t in zip(graphs, temps) if t >= thr]
            for direction, (train_env, other_env) in (("train_low_test_high", (low, high)),
                                                      ("train_high_test_low", (high, low))):
                accs = []
                for seed in SEEDS:
                    k = (v, "cross", float(thr), direction, seed)
                    if k in done:
                        continue
                    tr, matched = m.split_within_environment(train_env, m.MATCHED_TEST_SIZE, seed)
                    r = fit_and_eval(v, tr, {"matched": matched, "mismatched": other_env}, seed, n_feat, n_cls)
                    rows.append(dict(variant=v, experiment="cross", threshold=float(thr), direction=direction,
                                     seed=seed, **r))
                    done.add(k)
                    accs.append(r["mismatched_acc"])
                    pd.DataFrame(rows).to_csv(path, index=False)
                if accs:
                    print(f"[cross ] {v:<17} thr {thr:>4} {direction:<20} mismatched acc {np.mean(accs):.3f} over "
                          f"{len(accs)} new seeds | elapsed {(time.time() - t0) / 60:.1f} min")
    df = pd.DataFrame(rows)
    cx = df[df.experiment == "cross"]
    print("\nMean over the six cross-environment settings and the seeds (step %d):" % step)
    print(cx.groupby("variant")[["mismatched_acc", "mismatched_f1", "mismatched_acc_co", "mismatched_f1_co",
                                 "l_c", "l_o", "l_co", "l_co_minus_l_o"]].mean().round(3).to_string())
    print(f"saved {path}")
 
 
if __name__ == "__main__":
    steps, variants = parse(sys.argv[1:])
    for s in steps:
        run_step(s, variants)
 
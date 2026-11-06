
"""
Trains and tests the eight model variants (GIN, GIN with causal attention, and the controls without the causal losses) on a random split and on the cross-environment splits. All other model scripts build on this one.

Needs: graph_builder.py, lumo_features.csv and between_channel_features.csv.
"""
from __future__ import annotations
import os
import sys
import time
 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv, MessagePassing, global_mean_pool
 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = SCRIPT_DIR                      # shared modules (compare_vanilla_cal*.py)
HERE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "results"))   # all CSV results of gin_cal live here
sys.path.append(os.path.join(SCRIPT_DIR, "..", ".."))
from graph_builder import build_dataset, EDGE_ATTR_DIM
 
LUMO_CSV = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "01_data_and_features", "lumo_features.csv"))
BETWEEN_CHANNEL_CSV = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "01_data_and_features", "between_channel and temp explore", "between_channel_features", "between_channel_features.csv"))
DATASET_LABEL = "all sessions"    

QUICK = False                      # True: 30 epochs, 2 seeds, separate CSV (smoke test)
SEEDS = [42, 7, 123, 2024, 99, 1, 2, 3, 4, 5]
EPOCHS = 1000
HIDDEN_DIM = 64
NUM_LAYERS = 3                     # message-passing layers on every path (vanilla: 3; CAL: 1 shared + 2 branch)
DROPOUT = 0.3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
TEST_SIZE = 0.2                    # random split
MATCHED_TEST_SIZE = 0.2            # cross-environment, within the training environment
THRESHOLDS_C = [10.0, 15.0, 20.0]
NUM_BINS = 5                       # temperature strata for the backdoor term
STANDARDIZE = True                 # fit on training graphs only
 
LAMBDA1, LAMBDA2 = 0.5, 0.5   # best range in Sui et al.: lambda1 0.3-0.7, lambda2 0.3-0.5
INTERVENTION = "stratum"           # "stratum" (temperature strata) | "random" (as in the paper)
 
RUN_RANDOM = True
RUN_CROSS = True
VARIANTS = {
    "vanilla":        dict(kind="vanilla", attn=None,   causal=False, class_weight=False),
    "vanilla_bal":    dict(kind="vanilla", attn=None,   causal=False, class_weight=True),
    "cal_node":       dict(kind="cal", attn="node",     causal=True,  class_weight=False),
    "cal_edge":       dict(kind="cal", attn="edge",     causal=True,  class_weight=False),
    "cal_both":       dict(kind="cal", attn="both",     causal=True,  class_weight=False),
    "nocausal_node":  dict(kind="cal", attn="node",     causal=False, class_weight=False),
    "nocausal_edge":  dict(kind="cal", attn="edge",     causal=False, class_weight=False),
    "nocausal_both":  dict(kind="cal", attn="both",     causal=False, class_weight=False),
    # env = L_env: trivial branch must predict the temperature stratum
    # adv = L_adv: head on the causal rep predicts the stratum through gradient reversal (removes temperature from g_c)
    # wd_trivial0 = no weight decay on the trivial branch (tests whether weight decay empties it)
    "cal_edge_env":           dict(kind="cal", attn="edge", causal=True,  class_weight=False, env=True),
    "cal_edge_envadv":        dict(kind="cal", attn="edge", causal=True,  class_weight=False, env=True, adv=True),
    "nocausal_edge_envadv":   dict(kind="cal", attn="edge", causal=False, class_weight=False, env=True, adv=True),
    "cal_edge_envadv_nowd":   dict(kind="cal", attn="edge", causal=True,  class_weight=False, env=True, adv=True,
                                   wd_trivial0=True),
    "nocausal_edge_nowd":     dict(kind="cal", attn="edge", causal=False, class_weight=False, wd_trivial0=True),
    "cal_both_envadv":        dict(kind="cal", attn="both", causal=True,  class_weight=False, env=True, adv=True),
    "nocausal_both_envadv":   dict(kind="cal", attn="both", causal=False, class_weight=False, env=True, adv=True),
}
# Extra losses (not in Sui et al.; active only for variants with env / adv):
#   L_env = CE(env_head(g_t), temperature stratum)         -> g_t is forced to carry temperature
#   L_adv = CE(adv_head(GRL(g_c)), temperature stratum)    -> g_c is pushed to carry NO temperature
# They apply also when causal=False (lambda1 = lambda2 = 0), so nocausal_*_envadv isolates the effect of
# L_unif / L_caus once the trivial branch is alive.
LAMBDA_ENV = 1.0
LAMBDA_ADV = 1.0
GRL_STRENGTH = 1.0
# Variants run by default. The *_env / *_envadv / *_nowd variants above are defined but not run unless added here.
VARIANTS_TO_RUN = ["vanilla", "vanilla_bal", "cal_node", "cal_edge", "cal_both",
                   "nocausal_node", "nocausal_edge", "nocausal_both"]
 
# The report uses three losses (L_sup, L_unif, L_caus) and the first five seeds (42, 7, 123, 2024, 99).
# Result files carry the suffix _3loss.
SEEDS = SEEDS[:5]
if QUICK:
    EPOCHS, SEEDS = 30, SEEDS[:2]
CSV_PATH = os.path.join(HERE, "compare_vanilla_cal_by_seed_3loss"
                        + ("_quick" if QUICK else "") + ".csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
 
def load_merged_data():
    """Load the session table and join the between-channel features onto it."""
    df = pd.read_csv(LUMO_CSV)
    bc = pd.read_csv(BETWEEN_CHANNEL_CSV)
    return df.merge(bc.drop(columns=["damage_label", "temperature_degC"]),
                    left_on="source_file", right_on="session", how="left")
 
 
def rebin(graphs, n_bins):
    """Temperature quantile bins computed on these graphs only (the training pool)."""
    temps = np.array([g.temp.item() for g in graphs])
    edges = np.quantile(temps, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-6
    out = []
    for g in graphs:
        g2 = g.clone()
        b = int(np.searchsorted(edges, g.temp.item(), side="right") - 1)
        g2.eov_bin = torch.tensor([min(max(b, 0), n_bins - 1)], dtype=torch.long)
        out.append(g2)
    return out
 
 
def fit_standardizer(graphs):
    """Compute mean and standard deviation of node and edge features on the training graphs."""
    X = torch.cat([g.x for g in graphs])
    E = torch.cat([g.edge_attr for g in graphs])
    mx, sx, me, se = X.mean(0), X.std(0), E.mean(0), E.std(0)
    sx = torch.where(sx < 1e-8, torch.ones_like(sx), sx)
    se = torch.where(se < 1e-8, torch.ones_like(se), se)
    return mx, sx, me, se
 
 
def apply_standardizer(graphs, st):
    """Standardise the node and edge features of every graph with the given statistics."""
    mx, sx, me, se = st
    out = []
    for g in graphs:
        g2 = g.clone()
        g2.x = (g.x - mx) / sx
        g2.edge_attr = (g.edge_attr - me) / se
        out.append(g2)
    return out
 
 
def split_within_environment(env_graphs, test_size, seed):
    """Train pool + matched test inside one environment; classes with a single session stay in train."""
    labels = [g.y.item() for g in env_graphs]
    counts = pd.Series(labels).value_counts().to_dict()
    single = [i for i, l in enumerate(labels) if counts[l] < 2]
    multi = [i for i, l in enumerate(labels) if counts[l] >= 2]
    if len(multi) < 2:
        return [env_graphs[i] for i in single + multi], []
    tr, te = train_test_split(multi, test_size=test_size, stratify=[labels[i] for i in multi],
                              random_state=seed)
    return [env_graphs[i] for i in single + tr], [env_graphs[i] for i in te]
 
 
class WeightedGINE(MessagePassing):
    """GINE layer with per-edge message weights: MLP((1+eps) x_i + sum_j w_ji ReLU(x_j + W e_ji))."""
 
    def __init__(self, dim, edge_dim):
        super().__init__(aggr="add")
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.lin_edge = nn.Linear(edge_dim, dim)
        self.eps = nn.Parameter(torch.zeros(1))
 
    def forward(self, x, edge_index, edge_attr, edge_w):
        """Aggregate edge-weighted messages and apply the GIN update."""
        aggr = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_w=edge_w)
        return self.mlp((1.0 + self.eps) * x + aggr)
 
    def message(self, x_j, edge_attr, edge_w):
        """Message of an edge: weight times ReLU of the neighbour features plus the transformed edge features."""
        return edge_w * F.relu(x_j + self.lin_edge(edge_attr))
 
 
def gine(in_dim, hid, edge_dim):
    """Create a GINE convolution layer with a two-layer MLP."""
    return GINEConv(nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, hid)),
                    edge_dim=edge_dim)
 
 
class VanillaGIN(nn.Module):
    def __init__(self, node_dim, edge_dim, hid, n_cls, n_layers, dropout):
        super().__init__()
        dims = [node_dim] + [hid] * n_layers
        self.convs = nn.ModuleList([gine(dims[i], hid, edge_dim) for i in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.clf_c = nn.Linear(hid, n_cls)
 
    def forward(self, x, edge_index, edge_attr, batch):
        """Return the class logits of the plain GIN from mean-pooled node representations."""
        for conv in self.convs:
            x = self.drop(F.relu(conv(x, edge_index, edge_attr)))
        return {"logits_c": self.clf_c(global_mean_pool(x, batch))}
 
 
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, strength):
        """Return the input unchanged."""
        ctx.strength = strength
        return x.view_as(x)
 
    @staticmethod
    def backward(ctx, grad):
        """Reverse and scale the gradient."""
        return -ctx.strength * grad, None
 
 
class GINCAL(nn.Module):
    def __init__(self, node_dim, edge_dim, hid, n_cls, n_layers, dropout, attn, env=False, adv=False):
        super().__init__()
        self.use_node = attn in ("node", "both")
        self.use_edge = attn in ("edge", "both")
        self.enc = gine(node_dim, hid, edge_dim)
        self.node_att = nn.Sequential(nn.Linear(hid, hid // 2), nn.ReLU(), nn.Linear(hid // 2, 2)) \
            if self.use_node else None
        self.edge_att = nn.Sequential(nn.Linear(2 * hid + edge_dim, hid // 2), nn.ReLU(),
                                      nn.Linear(hid // 2, 2)) if self.use_edge else None
        self.conv_c = nn.ModuleList([WeightedGINE(hid, edge_dim) for _ in range(n_layers - 1)])
        self.conv_t = nn.ModuleList([WeightedGINE(hid, edge_dim) for _ in range(n_layers - 1)])
        self.drop = nn.Dropout(dropout)
        self.clf_c = nn.Linear(hid, n_cls)
        self.clf_t = nn.Linear(hid, n_cls)
        # created last so that the initialisation of the other layers does not depend on it
        self.env_head = nn.Linear(hid, NUM_BINS) if env else None
        self.adv_head = nn.Sequential(nn.Linear(hid, hid // 2), nn.ReLU(), nn.Linear(hid // 2, NUM_BINS)) \
            if adv else None
 
    def forward(self, x, edge_index, edge_attr, batch):
        """Split the graph into causal and shortcut parts with node and edge attention and return both representations and logits."""
        h = self.drop(F.relu(self.enc(x, edge_index, edge_attr)))
        n_edges = edge_index.shape[1]
        if self.use_node:
            a = F.softmax(self.node_att(h), dim=1)
            a_c, a_t = a[:, 0:1], a[:, 1:2]
        else:
            a_c = a_t = 0.5
        if self.use_edge:
            b = F.softmax(self.edge_att(torch.cat([h[edge_index[0]], h[edge_index[1]], edge_attr], 1)), dim=1)
            w_c, w_t = b[:, 0:1], b[:, 1:2]
        else:
            w_c = w_t = torch.full((n_edges, 1), 0.5, device=h.device)
        xc, xt = h * a_c, h * a_t
        for cc, ct in zip(self.conv_c, self.conv_t):
            xc = self.drop(F.relu(cc(xc, edge_index, edge_attr, w_c)))
            xt = self.drop(F.relu(ct(xt, edge_index, edge_attr, w_t)))
        g_c, g_t = global_mean_pool(xc, batch), global_mean_pool(xt, batch)
        return {"logits_c": self.clf_c(g_c), "g_c": g_c, "g_t": g_t,
                "env_logits": self.env_head(g_t) if self.env_head is not None else None,
                "adv_logits": self.adv_head(_GradReverse.apply(g_c, GRL_STRENGTH))
                if self.adv_head is not None else None,
                "att_node": a_c if self.use_node else None,
                "att_edge": w_c if self.use_edge else None}
 
 
def build_model(cfg, node_dim, n_cls):
    """Create the vanilla GIN or the GIN+CAL model from a variant configuration."""
    if cfg["kind"] == "vanilla":
        return VanillaGIN(node_dim, EDGE_ATTR_DIM, HIDDEN_DIM, n_cls, NUM_LAYERS, DROPOUT)
    return GINCAL(node_dim, EDGE_ATTR_DIM, HIDDEN_DIM, n_cls, NUM_LAYERS, DROPOUT, cfg["attn"],
                  env=cfg.get("env", False), adv=cfg.get("adv", False))
 
 
def make_optimizer(model, cfg):
    """Adam with weight decay; wd_trivial0 switches weight decay off on the trivial branch."""
    if not cfg.get("wd_trivial0"):
        return torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    triv = [p for mod in (model.conv_t, model.clf_t, model.env_head) if mod is not None for p in mod.parameters()]
    ids = {id(p) for p in triv}
    rest = [p for p in model.parameters() if id(p) not in ids]
    return torch.optim.Adam([{"params": rest, "weight_decay": WEIGHT_DECAY},
                             {"params": triv, "weight_decay": 0.0}], lr=LEARNING_RATE)
 
 
def compute_loss(model, batch, cfg, lam1, lam2, class_w):
    """Compute the supervised loss and, for CAL variants, the uniformity and causal losses."""
    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    y = batch.y.view(-1)
    l_sup = F.cross_entropy(out["logits_c"], y, weight=class_w)
    if cfg["kind"] == "vanilla":
        return l_sup, {}
    g_c, g_t = out["g_c"], out["g_t"]
 
    log_pt = F.log_softmax(model.clf_t(g_t), dim=1)
    l_unif = F.kl_div(log_pt, torch.full_like(log_pt, 1.0 / log_pt.shape[1]), reduction="batchmean")
 
    if INTERVENTION == "random":
        perm = torch.randperm(g_t.shape[0], device=g_t.device)
        l_caus = F.cross_entropy(model.clf_c(g_c + g_t[perm]), y)
    else:
        eov = batch.eov_bin.view(-1)
        n = g_t.shape[0]
        total = torch.zeros(n, device=g_t.device)
        for t in torch.unique(eov):
            mask = eov == t
            g_bar = g_t[mask].mean(dim=0, keepdim=True).detach()
            p = F.softmax(model.clf_c(g_c + g_bar), dim=1).gather(1, y.view(-1, 1)).squeeze(1)
            total = total + (mask.sum().item() / n) * p
        l_caus = -torch.log(total + 1e-8).mean()
 
    total_loss = l_sup + lam1 * l_unif + lam2 * l_caus
    extra = {}
    if out.get("env_logits") is not None:
        l_env = F.cross_entropy(out["env_logits"], batch.eov_bin.view(-1))
        total_loss = total_loss + LAMBDA_ENV * l_env
        extra["l_env"] = l_env.item()
    if out.get("adv_logits") is not None:
        l_adv = F.cross_entropy(out["adv_logits"], batch.eov_bin.view(-1))
        total_loss = total_loss + LAMBDA_ADV * l_adv
        extra["l_adv"] = l_adv.item()
    stats = {"l_sup": l_sup.item(), "l_unif": l_unif.item(), "l_caus": l_caus.item(), **extra,
             "att_node": out["att_node"].mean().item() if out["att_node"] is not None else np.nan,
             "att_edge": out["att_edge"].mean().item() if out["att_edge"] is not None else np.nan}
    return total_loss, stats
 
 
@torch.no_grad()
def evaluate(model, graphs):
    """Return accuracy and macro F1 of a model on a list of graphs."""
    if len(graphs) == 0:
        return np.nan, np.nan
    model.eval()
    b = Batch.from_data_list(graphs).to(DEVICE)
    pred = model(b.x, b.edge_index, b.edge_attr, b.batch)["logits_c"].argmax(1).cpu().numpy()
    y = b.y.view(-1).cpu().numpy()
    return accuracy_score(y, pred), f1_score(y, pred, average="macro", labels=np.unique(y))
 
 
def fit_and_eval(cfg, train_graphs, eval_sets, seed, node_dim, n_cls, lam1, lam2):
    """Train on train_graphs, evaluate on every list in eval_sets (name -> graphs)."""
    if STANDARDIZE:
        st = fit_standardizer(train_graphs)
        train_graphs = apply_standardizer(train_graphs, st)
        eval_sets = {k: apply_standardizer(v, st) for k, v in eval_sets.items()}
    if cfg["kind"] == "cal" and (INTERVENTION == "stratum" or cfg.get("env") or cfg.get("adv")):
        train_graphs = rebin(train_graphs, NUM_BINS)
 
    class_w = None
    if cfg["class_weight"]:
        counts = np.bincount([g.y.item() for g in train_graphs], minlength=n_cls).astype(float)
        present = counts > 0
        w = np.zeros(n_cls)
        w[present] = counts.sum() / (present.sum() * counts[present])
        class_w = torch.tensor(w, dtype=torch.float32, device=DEVICE)
 
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(cfg, node_dim, n_cls).to(DEVICE)
    opt = make_optimizer(model, cfg)
    batch = Batch.from_data_list(train_graphs).to(DEVICE)
    use_causal = cfg["causal"]
    l1, l2 = (lam1, lam2) if use_causal else (0.0, 0.0)
 
    stats = {}
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        loss, stats = compute_loss(model, batch, cfg, l1, l2, class_w)
        loss.backward()
        opt.step()
 
    res = {}
    res["train_acc"], res["train_f1"] = evaluate(model, train_graphs)
    for name, g in eval_sets.items():
        res[name + "_acc"], res[name + "_f1"] = evaluate(model, g)
    res.update({k: stats.get(k, np.nan) for k in ["l_sup", "l_unif", "l_caus", "l_env", "l_adv",
                                                  "att_node", "att_edge"]})
    return res
 
 
def choose_lambda(cfg, train_graphs, seed, node_dim, n_cls):
    """Fixed loss weights. The scripts in tuning/ replace this function to tune them on a validation split."""
    return LAMBDA1, LAMBDA2


def key_of(r):
    """Identify a result row by variant, experiment, threshold, direction and seed."""
    return (r["variant"], r["experiment"], float(r["threshold"]), r["direction"], int(r["seed"]))
 
 
def main():
    """Run all variants on the random split and the cross-environment splits and save the results."""
    t0 = time.time()
    df = load_merged_data()
    graphs, enc = build_dataset(df, num_eov_bins=NUM_BINS)
    node_dim, n_cls = graphs[0].x.shape[1], len(enc.classes_)
    temps = np.array([g.temp.item() for g in graphs])
    labels = [g.y.item() for g in graphs]
    print(f"{len(graphs)} graphs, {n_cls} classes, node dim {node_dim}, edge dim {EDGE_ATTR_DIM}, "
          f"device {DEVICE}, epochs {EPOCHS}, seeds {SEEDS}, standardize {STANDARDIZE}, "
          f"intervention {INTERVENTION}, lambdas ({LAMBDA1}, {LAMBDA2})")
    print(f"variants: {VARIANTS_TO_RUN}\n")
 
    rows, done = [], set()
    if os.path.exists(CSV_PATH):
        old = pd.read_csv(CSV_PATH)
        rows = old.to_dict("records")
        done = {key_of(r) for r in rows}
        print(f"resuming: {len(done)} finished runs found in {os.path.basename(CSV_PATH)}\n")
 
    def save():
        """Write the results collected so far to the CSV file."""
        pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
 
    n_new = 0
    for variant in VARIANTS_TO_RUN:
        cfg = VARIANTS[variant]
 
        if RUN_RANDOM:
            for seed in SEEDS:
                k = (variant, "random", 0.0, "all", seed)
                if k in done:
                    continue
                tr_idx, te_idx = train_test_split(range(len(graphs)), test_size=TEST_SIZE,
                                                  stratify=labels, random_state=seed)
                tr, te = [graphs[i] for i in tr_idx], [graphs[i] for i in te_idx]
                l1, l2 = choose_lambda(cfg, tr, seed, node_dim, n_cls)
                r = fit_and_eval(cfg, tr, {"test": te}, seed, node_dim, n_cls, l1, l2)
                rows.append(dict(variant=variant, experiment="random", threshold=0.0, direction="all",
                                 seed=seed, lam1=l1, lam2=l2, **r))
                done.add(k)
                n_new += 1
                print(f"[random] {variant:<14} seed {seed:>4} | train {r['train_acc']:.3f} | "
                      f"test acc {r['test_acc']:.3f} f1 {r['test_f1']:.3f}")
                save()
 
        if RUN_CROSS:
            for thr in THRESHOLDS_C:
                low = [g for g, t in zip(graphs, temps) if t < thr]
                high = [g for g, t in zip(graphs, temps) if t >= thr]
                for direction, (train_env, other_env) in (("train_low_test_high", (low, high)),
                                                          ("train_high_test_low", (high, low))):
                    accs = []
                    for seed in SEEDS:
                        k = (variant, "cross", float(thr), direction, seed)
                        if k in done:
                            continue
                        tr, matched = split_within_environment(train_env, MATCHED_TEST_SIZE, seed)
                        l1, l2 = choose_lambda(cfg, tr, seed, node_dim, n_cls)
                        r = fit_and_eval(cfg, tr, {"matched": matched, "mismatched": other_env},
                                         seed, node_dim, n_cls, l1, l2)
                        gap = r["matched_acc"] - r["mismatched_acc"] if not np.isnan(r["matched_acc"]) else np.nan
                        rows.append(dict(variant=variant, experiment="cross", threshold=float(thr),
                                         direction=direction, seed=seed, lam1=l1, lam2=l2, gap=gap, **r))
                        done.add(k)
                        n_new += 1
                        accs.append(r["mismatched_acc"])
                        save()
                    if accs:
                        print(f"[cross ] {variant:<14} thr {thr:>4} {direction:<20} "
                              f"mismatched acc {np.mean(accs):.3f} over {len(accs)} new seeds "
                              f"| elapsed {(time.time() - t0) / 60:.1f} min")
    save()
    print(f"\n{n_new} new runs in {(time.time() - t0) / 60:.1f} min -> {CSV_PATH}")
    report(pd.DataFrame(rows))
 
 
def boot_ci(v, n=5000, seed=0):
    """Bootstrap 95% confidence interval of the mean of v (resampling over seeds)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (n, len(v)))
    return np.percentile(v[idx].mean(axis=1), [2.5, 97.5])
 
 
def diff_line(name, per_seed):
    """Format the mean paired difference with its bootstrap interval. per_seed holds one difference per seed."""
    v = np.asarray(per_seed, dtype=float)
    v = v[~np.isnan(v)]
    lo, hi = boot_ci(v)
    return (f"  {name:<34} mean {v.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  SD {v.std(ddof=1):.3f}  "
            f"seeds +/-/0: {int((v > 0).sum())}/{int((v < 0).sum())}/{int((v == 0).sum())} (n={len(v)})")
 
 
def ms(x):
    """Format the mean and standard deviation of x as text."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return f"{x.mean():.3f} +/- {x.std(ddof=1):.3f}" if len(x) > 1 else f"{x.mean():.3f}"
 
 
PAIRS = [("cal_node", "nocausal_node"), ("cal_edge", "nocausal_edge"), ("cal_both", "nocausal_both"),
         ("cal_node", "vanilla"), ("cal_edge", "vanilla"), ("cal_both", "vanilla"),
         ("nocausal_node", "vanilla"), ("nocausal_edge", "vanilla"), ("nocausal_both", "vanilla"),
         ("vanilla_bal", "vanilla"), ("cal_node", "vanilla_bal"), ("cal_edge", "vanilla_bal"),
         ("cal_both", "vanilla_bal"), ("cal_edge", "cal_node"), ("cal_both", "cal_node"),
         ("cal_both", "cal_edge"),
         # trivial-branch variants
         ("cal_edge_env", "cal_edge"), ("cal_edge_envadv", "cal_edge"),
         ("cal_edge_envadv", "nocausal_edge_envadv"),      # effect of L_unif + L_caus once the branch is alive
         ("nocausal_edge_envadv", "nocausal_edge"),        # effect of L_env + L_adv alone
         ("cal_edge_envadv", "vanilla"), ("cal_edge_envadv", "cal_edge_env"),
         ("cal_edge_envadv_nowd", "cal_edge_envadv"), ("nocausal_edge_nowd", "nocausal_edge"),
         ("cal_both_envadv", "cal_both"), ("cal_both_envadv", "nocausal_both_envadv"),
         ("cal_both_envadv", "vanilla")]
 
 
def report(res):
    """Print the mean and standard deviation of every variant and the paired differences between variants."""
    pd.set_option("display.width", 250)
    order = [v for v in VARIANTS_TO_RUN if v in set(res.variant)]
    seeds = sorted(set(res.seed))
 
    rs = res[res.experiment == "random"]
    if len(rs):
        print("\n" + "=" * 100)
        print(f"RANDOM SPLIT, mean +/- SD (ddof=1) over {len(seeds)} seeds")
        print("=" * 100)
        print(f"  {'variant':<15}{'train acc':>18}{'test acc':>18}{'test F1':>18}{'gap':>8}")
        for v in order:
            r = rs[rs.variant == v]
            print(f"  {v:<15}{ms(r.train_acc):>18}{ms(r.test_acc):>18}{ms(r.test_f1):>18}"
                  f"{r.train_acc.mean() - r.test_acc.mean():>8.3f}")
        for metric in ["test_acc", "test_f1"]:
            print(f"\nPAIRED differences in {metric} (a - b), same seed -> same split; bootstrap over seeds")
            for a, b in PAIRS:
                if a in order and b in order:
                    A = rs[rs.variant == a].set_index("seed")[metric]
                    B = rs[rs.variant == b].set_index("seed")[metric]
                    print(diff_line(f"{a} - {b}", (A - B).dropna().values))
        cal = rs[rs.variant.str.startswith("cal_")]
        if len(cal):
            print("\nCAL diagnostics at the last epoch (mean over seeds): "
                  "L_caus - L_sup near 0 means the intervention term is inert; "
                  "l_env near ln(5)=1.61 means the trivial branch does not know the temperature")
            d = rs[rs.variant.isin([v for v in order if v.startswith("cal_")])].copy()
            d["l_caus_minus_l_sup"] = d.l_caus - d.l_sup
            dcols = [c for c in ["l_sup", "l_unif", "l_caus", "l_caus_minus_l_sup", "l_env", "l_adv",
                                 "att_node", "att_edge"] if c in d.columns]
            print(d.groupby("variant")[dcols].mean().loc[
                [v for v in order if v.startswith("cal_")]].round(4).to_string())
 
    cx = res[res.experiment == "cross"]
    if len(cx):
        print("\n" + "=" * 100)
        print(f"CROSS-ENVIRONMENT, mismatched accuracy, mean +/- SD over {len(seeds)} seeds")
        print("=" * 100)
        tab = cx.groupby(["threshold", "direction", "variant"]).mismatched_acc.agg(
            lambda x: ms(x)).unstack("variant")[order]
        print(tab.to_string())
        print("\nMean over the 6 settings:")
        print(cx.groupby("variant")[["matched_acc", "mismatched_acc", "mismatched_f1", "gap"]].mean()
              .loc[order].round(3).to_string())
        key = ["threshold", "direction", "seed"]
        for metric, label in [("mismatched_acc", "mismatched accuracy"), ("mismatched_f1", "mismatched macro-F1"),
                              ("gap", "gap (matched - mismatched)")]:
            print(f"\nPAIRED differences in {label} (a - b); settings averaged within seed, bootstrap over seeds")
            for a, b in PAIRS:
                if a in order and b in order:
                    A = cx[cx.variant == a].set_index(key)[metric]
                    B = cx[cx.variant == b].set_index(key)[metric]
                    d = (A - B).dropna().reset_index()
                    print(diff_line(f"{a} - {b}", d.groupby("seed")[metric].mean().values))
 
 
 
if __name__ == "__main__":
    main()
 
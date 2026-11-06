"""
Records the predicted class of every session and draws the confusion matrices of GIN (and optionally GIN+CAL and its control) at step 3.

Needs: compare_vanilla_cal_backward.py.
"""
from __future__ import annotations
import itertools
import os
import sys
 
import numpy as np
import pandas as pd
 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "core"))                      # shared modules (compare_vanilla_cal*.py)
HERE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "results"))   # all CSV results of gin_cal live here
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)
sys.path.insert(0, CORE_DIR)
# results go to a confusion_matrix folder inside gin_cal/results, created if missing
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(HERE, "confusion_matrix"))
os.makedirs(OUT_DIR, exist_ok=True)
import compare_vanilla_cal_backward as bw           # noqa: E402
 
m = bw.m
EXP = "A"
ARGS = sys.argv[1:]
STEP = int(next((a for a in ARGS if a.isdigit()), 3))
VARIANTS = ["vanilla", "cal_both", "nocausal_both"] if "all" in ARGS else ["vanilla"]
 
state = {"enc": None, "records": []}     # one entry per fit_and_eval call, in the order of the main loop
 
 
def summarise(variant, df, names):
    """Save and print the confusion matrix of one variant."""
    tag = f"{variant}_step{STEP}"
    df.to_csv(os.path.join(OUT_DIR, f"confusion_{tag}_predictions.csv"), index=False)
    cm = pd.crosstab(df["true"], df["pred"]).reindex(index=names, columns=names, fill_value=0)
    rn = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0)
    cm.to_csv(os.path.join(OUT_DIR, f"confusion_{tag}_counts.csv"))
    rn.round(3).to_csv(os.path.join(OUT_DIR, f"confusion_{tag}_rownorm.csv"))
 
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(f"\n{'=' * 100}\n{variant}, step {STEP}: counts (rows = true state, columns = predicted state)\n{'=' * 100}")
    print(cm.to_string())
    print("\nRow-normalised (share of the true state predicted as the column state):")
    print(rn.round(2).to_string())
    print("\nRecall per state:", {n: round(float(rn.loc[n, n]), 3) for n in names if not np.isnan(rn.loc[n, n])})
    print(f"Pooled accuracy over all predictions: {float((df['true'] == df['pred']).mean()):.3f}")
    off = rn.where(~np.eye(len(names), dtype=bool)).stack().sort_values(ascending=False).head(6)
    print("Largest errors (true -> predicted, share of the true state):")
    for (a, b), v in off.items():
        print(f"  {a} -> {b}: {v:.2f}")
 
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(rn.values, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("predicted state")
        ax.set_ylabel("true state")
        for i in range(len(names)):
            for j in range(len(names)):
                v = rn.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}\n({int(cm.values[i, j])})", ha="center", va="center", fontsize=7,
                            color="white" if v > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, label="share of the true state")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"confusion_{tag}.png"), dpi=200)
        plt.close(fig)
        print(f"saved confusion_{tag}.png")
    except ImportError:
        print("matplotlib not installed: no plot saved")
 
 
def main():
    """Train the models at one step and save the confusion matrices."""
    removed, node_keep, edge_keep = bw.kept_features(STEP)
    inner = bw.make_builder(EXP, STEP)
 
    def build(df, **kw):
        """Build the graphs and keep the label encoder."""
        graphs, enc = inner(df, **kw)
        state["enc"] = enc
        return graphs, enc
 
    m.build_dataset = build
    m.EDGE_ATTR_DIM = len(edge_keep)
    m.DATASET_LABEL = f"confusion matrix, step {STEP}"
    m.CSV_PATH = os.path.join(OUT_DIR, f"confusion_run_step{STEP}_by_seed.csv")      # fresh file, so nothing is skipped
    if os.path.exists(m.CSV_PATH):
        os.remove(m.CSV_PATH)
    m.VARIANTS_TO_RUN = VARIANTS
    m.RUN_RANDOM = False
    m.report = lambda res: None
    print(f"variants {VARIANTS}; step {STEP}; seeds {m.SEEDS}")
 
    orig_eval, orig_fit = m.evaluate, m.fit_and_eval
    calls = []
 
    def eval_hook(model, graphs):
        """Wrap evaluate() so that the predictions of every evaluated set are recorded."""
        out = orig_eval(model, graphs)
        if len(graphs):
            import torch
            from torch_geometric.data import Batch
            with torch.no_grad():
                model.eval()
                b = Batch.from_data_list(graphs).to(m.DEVICE)
                pred = model(b.x, b.edge_index, b.edge_attr, b.batch)["logits_c"].argmax(1).cpu().numpy()
            calls.append((b.y.view(-1).cpu().numpy(), pred, [float(g.temp.item()) for g in graphs]))
        else:
            calls.append((np.array([], dtype=int), np.array([], dtype=int), []))
        return out
 
    def fit_hook(*a, **kw):
        """Wrap fit_and_eval() so that the predictions on the mismatched test set are stored."""
        calls.clear()
        res = orig_fit(*a, **kw)
        # evaluate() order inside fit_and_eval: train set, then the eval sets in dict order (matched, mismatched)
        state["records"].append(calls[2])
        return res
 
    m.evaluate, m.fit_and_eval = eval_hook, fit_hook
    m.main()
 
    # the main loop visits variant, threshold, direction, seed in this fixed order
    order = list(itertools.product(VARIANTS, m.THRESHOLDS_C, ("train_low_test_high", "train_high_test_low"), m.SEEDS))
    recs = state["records"]
    assert len(recs) == len(order), f"{len(recs)} records but {len(order)} runs; was an old csv reused?"
    names = list(state["enc"].classes_)
    rows = {v: [] for v in VARIANTS}
    for (variant, thr, direction, seed), (y, p, t) in zip(order, recs):
        for yi, pi, ti in zip(y, p, t):
            rows[variant].append(dict(threshold=thr, direction=direction, seed=seed,
                                      true=names[yi], pred=names[pi], temp=ti))
    for variant in VARIANTS:
        summarise(variant, pd.DataFrame(rows[variant]), names)
 
 
if __name__ == "__main__":
    main()
 
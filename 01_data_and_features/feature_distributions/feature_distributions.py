"""
Describes the node and edge features over all sessions (range, spread and skewness), for the Results section on feature distributions.

Needs: lumo_features.csv and between_channel_features.csv; compare_vanilla_cal.py and feature_temperature_damage_analysis.py (03_models/gin_cal).
"""
from __future__ import annotations
import os
import sys
import warnings
 
import numpy as np
import pandas as pd
import matplotlib
 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import skew
 
HERE = os.path.dirname(os.path.abspath(__file__))
ASSESS_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "03_models", "gin_cal", "feature_assessment"))
OUT_DIR = os.path.join(HERE, "output_feature_distributions")
 
SKEW_LIMIT = 2.0       # skewness above this, for an all-positive feature, flags log10 as recommended
 
sys.path.insert(0, ASSESS_DIR)
sys.argv = [sys.argv[0]]            # make the imported module run in its default (full) mode
import feature_temperature_damage_analysis as fa
 
warnings.filterwarnings("ignore")
NODE_FEATURES = ["rms", "peak", "freq_hz"]
EDGE_FEATURES = ["xcov", "coh_b1", "coh_t1", "coh_b2", "trans_b1", "trans_t1", "trans_b2"]
SENSOR_LABELS = {0: "accelerometer x", 1: "accelerometer y", 2: "strain gauge"}
 
 
def describe(values):
    """Descriptive statistics of a flat array. Non-finite values are counted and then left out."""
    v = np.asarray(values, float).ravel()
    ok = np.isfinite(v)
    w = v[ok]
    if len(w) < 3:
        return dict(n_values=int(v.size), pct_missing=100.0 * (1.0 - ok.mean()))
    q1, med, q3 = np.percentile(w, [25, 50, 75])
    pos = w[w > 0]
    mean = w.mean()
    return dict(
        n_values=int(v.size),
        pct_missing=100.0 * (1.0 - ok.mean()),
        min=w.min(), q1=q1, median=med, q3=q3, max=w.max(),
        iqr=q3 - q1, mean=mean, sd=w.std(ddof=1),
        skewness=float(skew(w)),
        pct_positive=100.0 * (w > 0).mean(),
        orders_of_magnitude=float(np.log10(pos.max() / pos.min())) if len(pos) else np.nan,
        max_over_median=float(w.max() / med) if med != 0 else np.nan,
        mean_pulled_by_extremes=bool(mean > q3 or mean < q1),
    )
 
 
def log10_recommended(d):
    """True when a feature is strictly positive and strongly right-skewed."""
    if not np.isfinite(d.get("pct_positive", np.nan)) or d["pct_positive"] < 100.0:
        return False
    return d["skewness"] > SKEW_LIMIT
 
 
def load():
    """Load the graphs through feature_temperature_damage_analysis.py."""
    cwd = os.getcwd()
    os.chdir(ASSESS_DIR)
    try:
        graphs, class_names = fa.load_graphs()
    finally:
        os.chdir(cwd)
    xs, es, temp, y = fa.session_arrays(graphs)
    edge_index = graphs[0].edge_index.detach().cpu().numpy()      # (2, E), identical in every session
    return xs, es, temp, y, class_names, edge_index
 
 
def direction_check(es, edge_index):
    """Compare the two stored directions of every physical link (relative difference and median product)."""
    src, dst = edge_index
    index = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(src, dst))}
    pairs = [(k, index[(int(b), int(a))]) for (a, b), k in index.items()
             if (int(b), int(a)) in index and int(a) < int(b)]
    rows = []
    if not pairs:
        return pd.DataFrame(rows)
    f_idx = [p[0] for p in pairs]
    r_idx = [p[1] for p in pairs]
    for name in EDGE_FEATURES:
        a = edge_array(es, name)
        fwd, rev = a[:, f_idx], a[:, r_idx]
        rel = np.abs(fwd - rev) / (np.abs(fwd) + np.abs(rev) + 1e-300)
        rows.append(dict(feature=name, n_pairs=len(pairs), rel_diff=float(np.nanmedian(rel)),
                         median_product=float(np.nanmedian(fwd * rev))))
    return pd.DataFrame(rows)
 
 
def one_direction(edge_index):
    """Boolean mask that keeps the first stored direction of every pair of nodes (each physical link once)."""
    seen, keep = set(), np.zeros(edge_index.shape[1], bool)
    for k, (a, b) in enumerate(zip(edge_index[0], edge_index[1])):
        key = frozenset((int(a), int(b)))
        if key not in seen:
            seen.add(key)
            keep[k] = True
    return keep
 
 
def node_array(xs, name):
    """Return the values of one node feature for every session and node."""
    return xs[:, :, fa.NODE_NAMES.index(name)]                    # (sessions, nodes)
 
 
def edge_array(es, name):
    """Return the values of one edge feature for every session and edge."""
    return es[:, :, fa.EDGE_NAMES.index(name)]                    # (sessions, edges)
 
 
def main():
    """Compute and save the distribution statistics and histograms of the node and edge features."""
    os.makedirs(OUT_DIR, exist_ok=True)
    xs, es, temp, y, class_names, edge_index = load()
    n_sessions, n_nodes = xs.shape[0], xs.shape[1]
    healthy = np.array([class_names[i] == "Healthy" for i in y])
    # every physical link is stored in both directions: report how the two values relate, then keep one direction
    check = direction_check(es, edge_index)
    n_directed = edge_index.shape[1]
    keep = one_direction(edge_index)
    es, edge_index = es[:, keep], edge_index[:, keep]
    print(f"Edges stored: {n_directed} directed -> {es.shape[1]} physical links after keeping one direction each")
    if len(check):
        check.to_csv(os.path.join(OUT_DIR, "5_direction_check.csv"), index=False)
        print("Direction check:")
        print(check.to_string(index=False), "\n")
    node_type = xs[0, :, 3:6].argmax(1)                            # 0 accel x, 1 accel y, 2 strain gauge
    node_is_accel = node_type < 2
    edge_has_strain = (node_type[edge_index[0]] == 2) | (node_type[edge_index[1]] == 2)
 
    print(f"\n{n_sessions} sessions ({healthy.sum()} Healthy, {(~healthy).sum()} damaged), "
          f"{n_nodes} nodes ({node_is_accel.sum()} accelerometer, {(~node_is_accel).sum()} strain), "
          f"{es.shape[1]} edges ({(~edge_has_strain).sum()} accelerometer-accelerometer, "
          f"{edge_has_strain.sum()} strain-accelerometer)\n")
 
    # the groups: (level, feature, array of shape (sessions, items), item mask for each split)
    feats = [("node", n, node_array(xs, n)) for n in NODE_FEATURES] + \
            [("edge", n, edge_array(es, n)) for n in EDGE_FEATURES]
    splits = {
        "node": [("accelerometer", node_is_accel), ("strain gauge", ~node_is_accel)],
        "edge": [("accelerometer-accelerometer", ~edge_has_strain), ("strain-accelerometer", edge_has_strain)],
    }
 
    rows = []
    for level, name, a in feats:
        d = describe(a)
        d["log10_recommended"] = log10_recommended(d)
        rows.append(dict(level=level, feature=name, **d))
    overall = pd.DataFrame(rows)
 
    rows = []
    for level, name, a in feats:
        part = {}
        for label, mask in splits[level]:
            d = describe(a[:, mask])
            part[label] = d
            rows.append(dict(level=level, feature=name, sensor_or_edge_type=label,
                             n_items=int(mask.sum()), **d))
        l0, l1 = splits[level][0][0], splits[level][1][0]
        m0, m1 = part[l0]["median"], part[l1]["median"]
        # a ratio of medians is only meaningful when both medians are positive (not for signed features such as xcov)
        ratio = m0 / m1 if (m0 > 0 and m1 > 0) else np.nan
        rows.append(dict(level=level, feature=name, sensor_or_edge_type=f"median ratio {l0} / {l1}",
                         median=ratio))
    by_sensor = pd.DataFrame(rows)
 
    rows = []
    for level, name, a in feats:
        for label, mask in [("all", np.ones(a.shape[1], bool))] + splits[level]:
            for group, smask in (("Healthy", healthy), ("Damaged", ~healthy)):
                d = describe(a[smask][:, mask])
                rows.append(dict(level=level, feature=name, sensor_or_edge_type=label, group=group,
                                 n_sessions=int(smask.sum()),
                                 **{k: d.get(k, np.nan) for k in ("median", "q1", "q3", "iqr", "mean", "sd")}))
    by_health = pd.DataFrame(rows)
 
    rows = []
    for level, name, a in feats:
        d = describe(a)
        if not log10_recommended(d):
            continue
        l = np.log10(a[np.isfinite(a)])
        dl = describe(l)
        rows.append(dict(level=level, feature=name,
                         skewness_raw=d["skewness"], skewness_log10=dl["skewness"],
                         median_log10=dl["median"], q1_log10=dl["q1"], q3_log10=dl["q3"],
                         mean_log10=dl["mean"], sd_log10=dl["sd"], min_log10=dl["min"], max_log10=dl["max"]))
    log10_table = pd.DataFrame(rows)
 
    for fname, df in (("1_overall", overall), ("2_by_sensor_type", by_sensor),
                      ("3_by_health", by_health), ("4_log10", log10_table)):
        df.to_csv(os.path.join(OUT_DIR, f"{fname}.csv"), index=False)
 
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200,
                  "display.float_format", lambda v: f"{v:.3g}")
    print("=== 1. Overall (all sessions, pooled over nodes or edges) ===")
    print(overall[["level", "feature", "pct_missing", "min", "q1", "median", "q3", "max", "mean", "sd",
                   "skewness", "orders_of_magnitude", "mean_pulled_by_extremes", "log10_recommended"]]
          .to_string(index=False))
    print("\n=== 2. By sensor type (nodes) and edge type (edges) ===")
    print(by_sensor[["level", "feature", "sensor_or_edge_type", "min", "median", "q1", "q3", "max", "mean", "sd"]]
          .to_string(index=False))
    print("\n=== 3. Healthy vs damaged (median, Q1, Q3) ===")
    print(by_health[by_health.sensor_or_edge_type == "all"][["level", "feature", "group", "median", "q1", "q3"]]
          .to_string(index=False))
    print("\n=== 4. log10 statistics (strictly positive, strongly skewed features) ===")
    print(log10_table.to_string(index=False))
 
    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    for ax, (level, name, a) in zip(axes.ravel(), feats):
        v = a[np.isfinite(a)].ravel()
        if log10_recommended(describe(a)):
            ax.hist(np.log10(v), bins=50, color="#4C72B0")
            ax.set_xlabel(f"log10({name})")
        else:
            ax.hist(v, bins=50, color="#4C72B0")
            ax.set_xlabel(name)
        ax.set_title(f"{level}: {name}", fontsize=10)
        ax.set_yticks([])
    for ax in axes.ravel()[len(feats):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "feature_distributions.png"), dpi=150)
    print(f"\nSaved 4 CSV tables and feature_distributions.png in {OUT_DIR}")
 
 
if __name__ == "__main__":
    main()
 
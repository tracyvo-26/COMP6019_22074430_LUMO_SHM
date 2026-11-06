"""
Measures, for every feature, how much temperature and how much damage information it carries, and writes the order in which features are removed.

Needs: compare_vanilla_cal.py.
"""
from __future__ import annotations
import datetime
import os
import re
import sys
import time
import warnings
 
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
 
warnings.filterwarnings("ignore")
 
QUICK = len(sys.argv) > 1 and sys.argv[1].lower() == "quick"
N_TREES = 60 if QUICK else 200
REPEATS_TEMP = 1 if QUICK else 3          # repeats of the 5-fold CV for temperature
REPEATS_LABEL = 1 if QUICK else 2         # repeats of the 5-fold CV for the damage label
REPEATS_GROUP = 1 if QUICK else 2         # repeats of the grouped 5-fold CV
GROUP_LEVELS = ["day", "week"]
THRESHOLDS_C = [10.0, 15.0, 20.0]
SEED = 42
NODE_NAMES = ["rms", "peak", "freq_hz", "is_accel_x", "is_accel_y", "is_strain"]
EDGE_NAMES = ["xcov", "coh_b1", "coh_t1", "coh_b2", "trans_b1", "trans_t1", "trans_b2"]
NODE_MEASURES = ["rms", "peak", "freq_hz"]            # columns 0-2; columns 3-5 are the one-hot sensor type
OUT_CSV = "feature_temperature_damage_quick.csv" if QUICK else "feature_temperature_damage.csv"
OUT_SPEARMAN = "feature_temperature_spearman.csv"
 
INDIVIDUAL = [("node", n) for n in NODE_MEASURES] + [("edge", n) for n in EDGE_NAMES]
GROUPS = {
    "node_response":         [("node", n) for n in NODE_MEASURES],
    "edge_coherence":        [("edge", n) for n in ("coh_b1", "coh_t1", "coh_b2")],
    "edge_transmissibility": [("edge", n) for n in ("trans_b1", "trans_t1", "trans_b2")],
    "edge_xcov":             [("edge", "xcov")],
    "edge_all":              [("edge", n) for n in EDGE_NAMES],
    "all_features":          [("node", n) for n in NODE_MEASURES] + [("edge", n) for n in EDGE_NAMES],
    "mode_B1":               [("edge", "coh_b1"), ("edge", "trans_b1")],
    "mode_T1":               [("edge", "coh_t1"), ("edge", "trans_t1")],
    "mode_B2":               [("edge", "coh_b2"), ("edge", "trans_b2")],
}
 
 
def session_arrays(graphs):
    """Stack the graphs: node array (S, N, 6), edge array (S, E, 7), temperature, label."""
    xs = np.stack([g.x.detach().cpu().numpy() for g in graphs]).astype(float)
    es = np.stack([g.edge_attr.detach().cpu().numpy() for g in graphs]).astype(float)
    temp = np.array([float(g.temp.item()) for g in graphs])
    y = np.array([int(g.y.item()) for g in graphs])
    assert xs.shape[2] == len(NODE_NAMES), f"expected {len(NODE_NAMES)} node features, got {xs.shape[2]}"
    assert es.shape[2] == len(EDGE_NAMES), f"expected {len(EDGE_NAMES)} edge features, got {es.shape[2]}"
    return xs, es, temp, y
 
 
def fill_nan(a):
    """Replace NaN / inf by the column median (counted and reported)."""
    a = np.asarray(a, float).copy()
    bad = ~np.isfinite(a)
    if bad.any():
        a[bad] = np.nan
        med = np.nanmedian(a, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        idx = np.where(np.isnan(a))
        a[idx] = med[idx[1]]
    return a, int(bad.sum())
 
 
def feature_columns(xs, es, kind, name, rep):
    """The columns that one feature contributes, for the vector or pooled representation."""
    if kind == "node":
        j = NODE_NAMES.index(name)
        if rep == "vector":
            return xs[:, :, j]
        types = xs[0, :, 3:6].argmax(1)           # accel-x / accel-y / strain (same in every session)
        cols = [xs[:, types == t, j].mean(1) for t in range(3) if (types == t).any()]
        return np.stack(cols, axis=1)
    j = EDGE_NAMES.index(name)
    if rep == "vector":
        return es[:, :, j]
    return np.stack([es[:, :, j].mean(1), es[:, :, j].std(1)], axis=1)
 
 
def build_matrix(xs, es, members, rep):
    """Stack the columns of the selected features into one matrix and drop constant columns."""
    blocks = [feature_columns(xs, es, k, n, rep) for k, n in members]
    X, n_bad = fill_nan(np.concatenate(blocks, axis=1))
    # Drop columns that are (almost) constant across sessions: standardising them amplifies rounding noise and makes the
    # ridge R2 unstable. Random forests are unaffected.
    sd = X.std(0)
    keep = sd > np.maximum(1e-8, 1e-6 * np.abs(X.mean(0)))
    if keep.sum() == 0:
        keep[:] = True
    return X[:, keep], n_bad
 
 
def residualise_all(X, temp):
    """Remove a quadratic temperature trend fitted on all sessions from every column."""
    t = (temp - temp.mean()) / (temp.std() + 1e-9)
    D = np.column_stack([t, t ** 2])
    return X - LinearRegression().fit(D, X).predict(D)
 
 
def r2_temperature(X, temp, model_name):
    """Out-of-fold R2 of temperature predicted from the features by ridge regression or a random forest."""
    scores = []
    for r in range(REPEATS_TEMP):
        pred = np.zeros_like(temp)
        for tr, te in KFold(5, shuffle=True, random_state=SEED + r).split(X):
            if model_name == "ridge":
                mdl = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            else:
                mdl = RandomForestRegressor(n_estimators=N_TREES, min_samples_leaf=2, random_state=SEED, n_jobs=-1)
            pred[te] = mdl.fit(X[tr], temp[tr]).predict(X[te])
        scores.append(r2_score(temp, pred))
    return float(np.mean(scores)), float(np.std(scores))
 
 
def forest():
    """Create a random forest classifier with the settings used in the screens."""
    return RandomForestClassifier(n_estimators=N_TREES, min_samples_leaf=1, random_state=SEED, n_jobs=-1,
                                  class_weight=None)
 
 
def scores(y_true, y_pred):
    """Return macro F1 and accuracy for a set of predictions."""
    labs = np.unique(y_true)
    return (f1_score(y_true, y_pred, labels=labs, average="macro", zero_division=0),
            accuracy_score(y_true, y_pred))
 
 
def label_cv(X, y):
    """Macro F1 and accuracy of the damage label under repeated stratified 5-fold cross-validation."""
    f1s, accs = [], []
    for r in range(REPEATS_LABEL):
        pred = np.zeros_like(y)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(X, y):
            pred[te] = forest().fit(X[tr], y[tr]).predict(X[te])
        f, a = scores(y, pred)
        f1s.append(f)
        accs.append(a)
    return float(np.mean(f1s)), float(np.mean(accs))
 
 
def label_cross(X, y, temp):
    """Macro F1 and accuracy of the damage label when training on one temperature range and testing on the other."""
    f1s, accs = [], []
    for thr in THRESHOLDS_C:
        low, high = np.where(temp < thr)[0], np.where(temp >= thr)[0]
        for tr, te in ((low, high), (high, low)):
            if len(tr) < 5 or len(te) < 5 or len(np.unique(y[tr])) < 2:
                continue
            pred = forest().fit(X[tr], y[tr]).predict(X[te])
            f, a = scores(y[te], pred)
            f1s.append(f)
            accs.append(a)
    return float(np.mean(f1s)), float(np.mean(accs))
 
 
_FILE_RE = re.compile(r"SHMTS_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})")
 
 
def session_groups(graphs):
    """Time-block id per session ('day' or 'week'), or None if a graph has no usable source file."""
    days = []
    for g in graphs:
        f = getattr(g, "source_file", None)
        mt = _FILE_RE.search(str(f)) if f is not None else None
        if not mt:
            return None
        days.append(datetime.date(int(mt.group(1)), int(mt.group(2)), int(mt.group(3))).toordinal())
    days = np.array(days)
    return {"day": days, "week": (days - days.min()) // 7}
 
 
def group_folds(groups, n_splits, seed):
    """Split groups into folds that keep each group in one fold, with balanced fold sizes."""
    rng = np.random.default_rng(seed)
    uniq, counts = np.unique(groups, return_counts=True)
    n_splits = min(n_splits, len(uniq))
    if n_splits < 3:
        return []
    order = rng.permutation(len(uniq))
    order = order[np.argsort(-counts[order], kind="stable")]
    load = np.zeros(n_splits)
    assign = {}
    for k in order:
        f = int(np.argmin(load))
        assign[uniq[k]] = f
        load[f] += counts[k]
    fold = np.array([assign[g] for g in groups])
    return [(np.where(fold != f)[0], np.where(fold == f)[0]) for f in range(n_splits)]
 
 
def r2_temperature_grouped(X, temp, groups):
    """Out-of-fold R2 of a random forest, with time-grouped folds (NaN if too few groups)."""
    out = []
    for r in range(REPEATS_GROUP):
        folds = group_folds(groups, 5, SEED + r)
        if not folds:
            return float("nan")
        pred = np.zeros_like(temp)
        for tr, te in folds:
            mdl = RandomForestRegressor(n_estimators=N_TREES, min_samples_leaf=2, random_state=SEED, n_jobs=-1)
            pred[te] = mdl.fit(X[tr], temp[tr]).predict(X[te])
        out.append(r2_score(temp, pred))
    return float(np.mean(out))
 
 
def label_cv_grouped(X, y, groups):
    """Macro-F1 of the damage label, random forest, time-grouped folds (NaN if too few groups)."""
    out = []
    for r in range(REPEATS_GROUP):
        folds = group_folds(groups, 5, SEED + r)
        if not folds:
            return float("nan")
        pred = np.zeros_like(y)
        for tr, te in folds:
            if len(np.unique(y[tr])) < 2:
                pred[te] = y[tr][0]
                continue
            pred[te] = forest().fit(X[tr], y[tr]).predict(X[te])
        out.append(scores(y, pred)[0])
    return float(np.mean(out))
 
 
def analyse_unit(unit_name, unit_type, members, rep, xs, es, temp, y, healthy_idx, groups):
    """Compute the temperature and damage measures of one feature or feature group."""
    X, n_bad = build_matrix(xs, es, members, rep)
    r2_ridge, r2_ridge_sd = r2_temperature(X, temp, "ridge")
    r2_rf, r2_rf_sd = r2_temperature(X, temp, "rf")
    Xr = residualise_all(X, temp)
    f1_cv, acc_cv = label_cv(X, y)
    f1_cr, acc_cr = label_cross(X, y, temp)
    f1_crr, acc_crr = label_cross(Xr, y, temp)
    f1_cvr, _ = label_cv(Xr, y)
    temp_info = max(r2_ridge, r2_rf, 0.0)
    if healthy_idx is not None and len(healthy_idx) >= 15:
        Xh, th = X[healthy_idx], temp[healthy_idx]
        r2h_ridge, r2h_ridge_sd = r2_temperature(Xh, th, "ridge")
        r2h_rf, r2h_rf_sd = r2_temperature(Xh, th, "rf")
        temp_info_h = max(r2h_ridge, r2h_rf, 0.0)
    else:
        r2h_ridge = r2h_ridge_sd = r2h_rf = r2h_rf_sd = float("nan")
        temp_info_h = float("nan")
    grouped = {}
    for lvl in GROUP_LEVELS:
        nan = float("nan")
        if groups is None:
            grouped.update({f"r2_temp_rf_{lvl}": nan, f"r2_temp_healthy_rf_{lvl}": nan, f"f1_cv_{lvl}": nan})
            continue
        gl = groups[lvl]
        grouped[f"r2_temp_rf_{lvl}"] = r2_temperature_grouped(X, temp, gl)
        grouped[f"f1_cv_{lvl}"] = label_cv_grouped(X, y, gl)
        grouped[f"r2_temp_healthy_rf_{lvl}"] = (r2_temperature_grouped(X[healthy_idx], temp[healthy_idx], gl[healthy_idx])
                                                if healthy_idx is not None and len(healthy_idx) >= 15 else nan)
    wk = grouped.get("r2_temp_healthy_rf_week", float("nan"))
    return dict(unit=unit_name, type=unit_type, representation=rep, n_columns=X.shape[1], n_nonfinite_filled=n_bad,
                r2_temp_ridge=r2_ridge, r2_temp_ridge_sd=r2_ridge_sd, r2_temp_rf=r2_rf, r2_temp_rf_sd=r2_rf_sd,
                f1_cv=f1_cv, acc_cv=acc_cv, f1_cv_resid=f1_cvr,
                f1_cross=f1_cr, acc_cross=acc_cr, f1_cross_resid=f1_crr, acc_cross_resid=acc_crr,
                r2_temp_healthy_ridge=r2h_ridge, r2_temp_healthy_ridge_sd=r2h_ridge_sd,
                r2_temp_healthy_rf=r2h_rf, r2_temp_healthy_rf_sd=r2h_rf_sd,
                damage_minus_temp=f1_crr - temp_info, damage_minus_temp_healthy=f1_crr - temp_info_h,
                damage_minus_temp_healthy_week=f1_crr - max(wk, 0.0), **grouped)
 
 
def spearman_table(xs, es, temp):
    """Quick sanity check: |Spearman| of each feature's session mean with temperature (pooled over nodes / edges)."""
    rows = []
    for kind, name in INDIVIDUAL:
        v = feature_columns(xs, es, kind, name, "vector")
        v, _ = fill_nan(v)
        rho, p = spearmanr(v.mean(1), temp)
        rows.append(dict(feature=f"{kind}:{name}", spearman_with_temp=rho, p_value=p))
    return pd.DataFrame(rows)

def removal_order(res, rep="vector"):
    """Backward elimination order: features by decreasing Healthy-only temperature R2, the lowest never removed."""
    t = res[(res.type == "individual") & (res.representation == rep)].copy()
    t["kind"] = t["unit"].str.split(":").str[0]
    t["feature"] = t["unit"].str.split(":").str[1]
    t = t.sort_values("r2_temp_healthy_rf_week", ascending=False).reset_index(drop=True)
    return t[["feature", "kind", "r2_temp_healthy_rf_week"]].rename(columns={"r2_temp_healthy_rf_week": "r2"}) 

def load_graphs():
    """Load compare_vanilla_cal.py and build the graphs."""
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("cvc", os.path.join(here, "..", "core", "compare_vanilla_cal.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["cvc"] = m
    spec.loader.exec_module(m)
    df = m.load_merged_data()
    graphs, enc = m.build_dataset(df, num_eov_bins=m.NUM_BINS)
    return graphs, list(enc.classes_)
 
 
def run(graphs, class_names):
    """Compute the measures for every feature and feature group and save the tables."""
    xs, es, temp, y = session_arrays(graphs)
    print(f"{len(graphs)} sessions | node array {xs.shape} | edge array {es.shape} | "
          f"temperature {temp.min():.1f} to {temp.max():.1f} C | classes {class_names}")
    majority = pd.Series(y).value_counts(normalize=True).max()
    print(f"majority-class rate (accuracy reference): {majority:.3f}")
    healthy_idx = None
    if "Healthy" in class_names:
        healthy_idx = np.where(y == class_names.index("Healthy"))[0]
        print(f"Healthy sessions: {len(healthy_idx)}, temperature {temp[healthy_idx].min():.1f} to "
              f"{temp[healthy_idx].max():.1f} C (used for the Healthy-only temperature R2)")
    else:
        print("WARNING: no class named 'Healthy'; Healthy-only columns will be NaN")
    groups = session_groups(graphs)
    if groups is None:
        print("WARNING: graphs have no usable source_file (SHMTS_YYYYMMDDHHMM); grouped-CV columns will be NaN")
    else:
        for lvl in GROUP_LEVELS:
            u, c = np.unique(groups[lvl], return_counts=True)
            print(f"time groups '{lvl}': {len(u)} groups, sessions per group min {c.min()} / median {int(np.median(c))} / max {c.max()}")
        if healthy_idx is not None:
            for lvl in GROUP_LEVELS:
                print(f"  Healthy sessions fall in {len(np.unique(groups[lvl][healthy_idx]))} '{lvl}' groups")
    print()
 
    units = [(f"{k}:{n}", "individual", [(k, n)]) for k, n in INDIVIDUAL]
    units += [(g, "group", mem) for g, mem in GROUPS.items()]
    rows = []
    t0 = time.time()
    for i, (name, utype, members) in enumerate(units, 1):
        for rep in ("vector", "pooled"):
            r = analyse_unit(name, utype, members, rep, xs, es, temp, y, healthy_idx, groups)
            rows.append(r)
            print(f"[{i:2d}/{len(units)}] {name:28s} {rep:6s}  R2 temp ridge {r['r2_temp_ridge']:+.2f} rf {r['r2_temp_rf']:+.2f}"
                  f" | Healthy-only R2 rf {r['r2_temp_healthy_rf']:+.2f} (day {r['r2_temp_healthy_rf_day']:+.2f}, week {r['r2_temp_healthy_rf_week']:+.2f})"
                  f" | F1 cv {r['f1_cv']:.2f} (week {r['f1_cv_week']:.2f}) cross {r['f1_cross']:.2f} cross-resid {r['f1_cross_resid']:.2f}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(rows)
    res.insert(0, "n_sessions", len(graphs))
    res.to_csv(OUT_CSV, index=False)
    sp = spearman_table(xs, es, temp)
    sp.to_csv(OUT_SPEARMAN, index=False)

    order = removal_order(res)
    order.to_csv("backward_removal_order.csv", index=False)
    kept = order.feature.iloc[-1]                       # lowest R2: never removed
    print(f"\nREMOVAL_ORDER for compare_vanilla_cal_backward.py (n = {len(graphs)}; {kept} is never removed):")
    print("REMOVAL_ORDER = [" + ", ".join(f'("{r.feature}", "{r.kind}", {r.r2:.2f})'
                                          for r in order.itertuples() if r.feature != kept) + "]")
 
    show = ["unit", "r2_temp_healthy_rf", "r2_temp_healthy_rf_day", "r2_temp_healthy_rf_week", "f1_cv", "f1_cv_week",
            "f1_cross", "f1_cross_resid", "damage_minus_temp_healthy_week"]
    sort_col = "damage_minus_temp_healthy_week" if res["damage_minus_temp_healthy_week"].notna().any() else "damage_minus_temp_healthy"
    for rep in ("vector", "pooled"):
        for utype in ("individual", "group"):
            t = res[(res.representation == rep) & (res.type == utype)].sort_values(sort_col, ascending=False)
            print(f"\n=== {utype}, {rep} representation, sorted by {sort_col} (best first) ===")
            print(t[show].round(3).to_string(index=False))
    print("\n=== Spearman correlation of each feature's session mean with temperature (sorted by |rho|) ===")
    print(sp.round(3).sort_values("spearman_with_temp", key=abs, ascending=False).to_string(index=False))
    print(f"\nSaved {OUT_CSV} and {OUT_SPEARMAN}")
 
 
if __name__ == "__main__":
    g, cls = load_graphs()
    run(g, cls)
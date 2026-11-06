"""
Tests whether the spatial features can tell the damage states apart in a new temperature range, using a random forest.

Needs: spatial_features.csv (from spatial_features.py).
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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, r2_score

warnings.filterwarnings("ignore")

ARGS = sys.argv[1:]
QUICK = "quick" in ARGS
COMP_DEGREE = 1
N_TREES = 60 if QUICK else 200
REPEATS = 1 if QUICK else 2
THRESHOLDS_C = [10.0, 15.0, 20.0]
MIN_HEALTHY_FIT = 10
SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
IN_CSV = next((a for a in ARGS if a.lower().endswith(".csv")), os.path.join(HERE, "spatial_features.csv"))
OUT_CSV = os.path.join(HERE, "spatial_feature_screen_quick.csv" if QUICK else "spatial_feature_screen.csv")
MODES = ["B1x", "B1y", "B2x", "B2y"]
_RE = re.compile(r"SHMTS_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})")


def week_groups(sessions):
    """Assign each session to a 7-day block from the date in its file name."""
    days = []
    for s in sessions:
        m = _RE.search(str(s))
        if not m:
            return None
        days.append(datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3))).toordinal())
    days = np.array(days)
    return (days - days.min()) // 7


def group_folds(groups, n_splits, seed):
    """Split groups into folds so that each group stays inside one fold and fold sizes are balanced."""
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
        j = int(np.argmin(load))
        assign[uniq[k]] = j
        load[j] += counts[k]
    fold = np.array([assign[g] for g in groups])
    return [(np.where(fold != j)[0], np.where(fold == j)[0]) for j in range(n_splits)]


def fit_comp(X, T, healthy):
    """Fit a polynomial of temperature to each column using Healthy training rows only."""
    mu, sd = T[healthy].mean() if healthy.any() else 0.0, (T[healthy].std() if healthy.any() else 1.0) + 1e-9
    coefs = []
    for j in range(X.shape[1]):
        ok = healthy & np.isfinite(X[:, j])
        if ok.sum() < MIN_HEALTHY_FIT:
            coefs.append(None)
            continue
        coefs.append(np.polyfit((T[ok] - mu) / sd, X[ok, j], COMP_DEGREE))
    return mu, sd, coefs


def apply_comp(X, T, comp):
    """Subtract the fitted temperature polynomial from every column of X."""
    mu, sd, coefs = comp
    out = X.copy()
    t = (T - mu) / sd
    for j, c in enumerate(coefs):
        out[:, j] = np.nan if c is None else X[:, j] - np.polyval(c, t)
    return out


def prepare(unit, Xall, T, healthy, tr, te):
    """Compensate (if requested), impute and standardise the train and test matrices of one split."""
    Xtr, Xte = Xall[tr], Xall[te]
    if unit["comp"]:
        comp = fit_comp(Xtr, T[tr], healthy[tr])
        Xtr, Xte = apply_comp(Xtr, T[tr], comp), apply_comp(Xte, T[te], comp)
    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(Xtr)
    if Xtr.shape[1] == 0:
        return None
    return Xtr, imp.transform(Xte)


def forest():
    """Create a random forest classifier with the settings used in the screens."""
    return RandomForestClassifier(n_estimators=N_TREES, random_state=SEED, n_jobs=-1)


def macro_acc(y_true, y_pred):
    """Return macro F1 and accuracy for a set of predictions."""
    labs = np.unique(y_true)
    return (f1_score(y_true, y_pred, labels=labs, average="macro", zero_division=0),
            accuracy_score(y_true, y_pred))


def grouped_r2(unit, Xall, T, healthy, weeks, subset):
    """Out-of-fold R2 of temperature under repeated folds that keep 7-day blocks together."""
    out = []
    for r in range(REPEATS):
        folds = group_folds(weeks[subset], 5, SEED + r)
        if not folds:
            return float("nan")
        pred = np.full(len(subset), np.nan)
        for tr, te in folds:
            p = prepare(unit, Xall, T, healthy, subset[tr], subset[te])
            if p is None:
                return float("nan")
            mdl = RandomForestRegressor(n_estimators=N_TREES, min_samples_leaf=2, random_state=SEED, n_jobs=-1)
            pred[te] = mdl.fit(p[0], T[subset[tr]]).predict(p[1])
        out.append(r2_score(T[subset], pred))
    return float(np.mean(out))


def grouped_f1(unit, Xall, T, y, healthy, weeks):
    """Macro F1 of the damage label under repeated folds that keep 7-day blocks together."""
    out = []
    for r in range(REPEATS):
        folds = group_folds(weeks, 5, SEED + r)
        if not folds:
            return float("nan")
        pred = np.zeros_like(y)
        for tr, te in folds:
            p = prepare(unit, Xall, T, healthy, tr, te)
            if p is None or len(np.unique(y[tr])) < 2:
                pred[te] = y[tr][0]
                continue
            pred[te] = forest().fit(p[0], y[tr]).predict(p[1])
        out.append(macro_acc(y, pred)[0])
    return float(np.mean(out))


def cross_env_detail(unit, Xall, T, y, healthy):
    """Like cross_env, but also returns the predictions pooled over all splits."""
    f1s, accs, yt, yp = [], [], [], []
    for thr in THRESHOLDS_C:
        low, high = np.where(T < thr)[0], np.where(T >= thr)[0]
        for tr, te in ((low, high), (high, low)):
            if len(tr) < 5 or len(te) < 5 or len(np.unique(y[tr])) < 2:
                continue
            p = prepare(unit, Xall, T, healthy, tr, te)
            if p is None:
                continue
            pred = forest().fit(p[0], y[tr]).predict(p[1])
            f, a = macro_acc(y[te], pred)
            f1s.append(f)
            accs.append(a)
            yt.extend(y[te])
            yp.extend(pred)
    if not f1s:
        return float("nan"), float("nan"), np.array([]), np.array([])
    return float(np.mean(f1s)), float(np.mean(accs)), np.array(yt), np.array(yp)


def load():
    """Load the spatial feature table and add the mean curvature of the interior levels."""
    df = pd.read_csv(IN_CSV).drop_duplicates("session").reset_index(drop=True)
    print(f"input: {IN_CSV}\n{len(df)} sessions")
    interior = [f"L{i}" for i in range(2, 9)]
    for m in MODES:
        df[f"curvmean_{m}"] = df[[f"curv_{m}_{l}" for l in interior]].abs().mean(axis=1)
    return df


def build_units(df):
    """Define the spatial feature sets (amplitudes, curvatures and their means) to be screened."""
    amp = {m: [c for c in df.columns if c.startswith(f"amp_{m}_")] for m in MODES}
    cur = {m: [c for c in df.columns if c.startswith(f"curv_{m}_")] for m in MODES}
    allamp = sum(amp.values(), [])
    allcur = sum(cur.values(), [])
    groups = {"amp_all": allamp, "curv_all": allcur, "ampcurv_all": allamp + allcur,
              "curvmean_all": [f"curvmean_{m}" for m in MODES]}
    for m in MODES:
        groups[f"amp_{m}"] = amp[m]
        groups[f"curv_{m}"] = cur[m]
    units = [dict(name="temperature_only (control)", cols=None, comp=False, kind="control")]
    for g, cols in groups.items():
        units.append(dict(name=g, cols=cols, comp=False, kind="group"))
        units.append(dict(name=g + "__comp", cols=cols, comp=True, kind="group"))
    return units


def common(df):
    """Return temperature, classes, labels, Healthy mask and 7-day blocks from the table."""
    T = df["temperature_degC"].to_numpy(float)
    classes = sorted(df["damage_label"].unique())
    y = np.array([classes.index(c) for c in df["damage_label"]])
    healthy = (df["damage_label"] == "Healthy").to_numpy()
    weeks = week_groups(df["session"])
    if weeks is None:
        raise SystemExit("session names are not SHMTS_YYYYMMDDHHMM")
    return T, classes, y, healthy, weeks


def main():
    """Screen each spatial feature set for temperature content and cross-environment damage detection and save the table."""
    df = load()
    T, classes, y, healthy, weeks = common(df)
    hid = np.where(healthy)[0]
    print(f"classes {classes}; Healthy sessions {len(hid)}; temperature {T.min():.1f} to {T.max():.1f} C")
    rows, t0 = [], time.time()
    units = build_units(df)
    for i, u in enumerate(units, 1):
        Xall = T[:, None] if u["cols"] is None else df[u["cols"]].to_numpy(float)
        r = dict(unit=u["name"], kind=u["kind"], compensated=u["comp"], n_columns=Xall.shape[1])
        r["r2_temp_healthy_week"] = grouped_r2(u, Xall, T, healthy, weeks, hid)
        r["f1_cv_week"] = grouped_f1(u, Xall, T, y, healthy, weeks)
        r["f1_cross"], r["acc_cross"], _, _ = cross_env_detail(u, Xall, T, y, healthy)
        rows.append(r)
        print(f"[{i:2d}/{len(units)}] {u['name']:28s} R2 temp Healthy-week {r['r2_temp_healthy_week']:+.2f} | "
              f"F1 cv-week {r['f1_cv_week']:.2f} cross {r['f1_cross']:.2f} acc {r['acc_cross']:.2f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(rows)
    ref = res.loc[res.kind == "control", "f1_cross"].iloc[0]
    res["f1_cross_vs_T"] = res["f1_cross"] - ref
    res.insert(0, "n_sessions", len(df))
    res.to_csv(OUT_CSV, index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(f"\ncontrol: temperature alone gives cross-environment macro-F1 {ref:.3f}")
    print(res.sort_values("f1_cross", ascending=False)[
        ["unit", "r2_temp_healthy_week", "f1_cv_week", "f1_cross", "acc_cross", "f1_cross_vs_T"]]
        .round(3).to_string(index=False))
    print(f"\nSaved {OUT_CSV}")


def checks():
    """A per-class F1, B complete cases only, C detection-flags-only control (same logic as modal_feature_screen.py)."""
    df = load()
    T, classes, y, healthy, weeks = common(df)
    det = [f"det_{m}" for m in MODES]
    flags = (1.0 - df[det].to_numpy(float))
    f1c, accc, _, _ = cross_env_detail(dict(comp=False), flags, T, y, healthy)
    print(f"C  detection flags only: cross-environment macro-F1 {f1c:.3f}\n")
    keep = ["amp_all", "amp_all__comp", "curv_all", "ampcurv_all", "amp_B2y", "amp_B2y__comp", "amp_B2x", "amp_B1x",
            "amp_B1y"]
    rows_cls, rows_cc = [], []
    for u in build_units(df):
        if u["name"] not in keep:
            continue
        Xall = df[u["cols"]].to_numpy(float)
        f1, acc, yt, yp = cross_env_detail(u, Xall, T, y, healthy)
        per = f1_score(yt, yp, labels=list(range(len(classes))), average=None, zero_division=0)
        rows_cls.append(dict(unit=u["name"], macro_F1=f1, **{c: v for c, v in zip(classes, per)}))
        ok = np.isfinite(Xall).all(1)
        if ok.sum() >= 30 and len(np.unique(y[ok])) >= 3:
            f1cc = cross_env_detail(u, Xall[ok], T[ok], y[ok], healthy[ok])[0]
        else:
            f1cc = float("nan")
        rows_cc.append(dict(unit=u["name"], n_complete=int(ok.sum()), macro_F1_all=f1, macro_F1_complete_cases=f1cc,
                            classes_present=int(pd.Series(df["damage_label"][ok]).nunique())))
        print(f"{u['name']:16s} all {f1:.3f} | complete cases ({ok.sum()}) {f1cc:.3f}", flush=True)
    print("\nA  per-class F1 pooled over the six settings")
    print(pd.DataFrame(rows_cls).round(2).to_string(index=False))
    out = os.path.join(HERE, "spatial_feature_screen_checks.csv")
    pd.DataFrame(rows_cls).merge(pd.DataFrame(rows_cc), on="unit").to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    checks() if "check" in ARGS else main()

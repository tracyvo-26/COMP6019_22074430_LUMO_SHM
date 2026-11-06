
"""
Tests whether the modal frequencies (and ratios and splits of them) can tell the damage states apart in a new temperature range, with and without a temperature correction. This is the frequency baseline.

Needs: ssi_modal_frequencies.csv (from 02_modal_identification/SSI-COV).
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
COMP_DEGREE = 2 if "deg2" in ARGS else 1
N_TREES = 60 if QUICK else 200
REPEATS = 1 if QUICK else 2               # repeats of the grouped 5-fold CV
THRESHOLDS_C = [10.0, 15.0, 20.0]
MIN_HEALTHY_FIT = 10                      # fewer finite Healthy training rows than this -> the column is not compensated
SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
FREQ_CSV_CANDIDATES = [
    os.path.join(HERE, "ssi_modal_frequencies.csv"),
    os.path.join(HERE, "..", "..", "02_modal_identification", "SSI-COV", "output_batch", "blockrows20_final",
                 "ssi_modal_frequencies.csv"),
]
OUT_CSV = os.path.join(HERE, "modal_feature_screen_quick.csv" if QUICK else "modal_feature_screen.csv")
 
 
def find_csv():
    """Locate the frequency table, either from the command line or from the default candidate paths."""
    given = next((a for a in ARGS if a.lower().endswith(".csv")), None)
    if given:
        if not os.path.exists(given):
            raise SystemExit(f"csv not found: {given}")
        return given
    for p in FREQ_CSV_CANDIDATES:
        if os.path.exists(p):
            return os.path.normpath(p)
    raise SystemExit("ssi_modal_frequencies.csv not found")
 
 
def load_table():
    """Load the SSI-COV frequency table and add the derived frequency, ratio and split columns."""
    path = find_csv()
    df = pd.read_csv(path)
    need = ["session", "damage_label", "temperature_degC", "freq_B1-x", "freq_B1-y", "freq_B2-x", "freq_B2-y"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"{path} lacks columns {miss}; found {list(df.columns)}")
    df = df.drop_duplicates("session").reset_index(drop=True)
    print(f"input: {path}")
    print(f"{len(df)} sessions")
    f = {k: df[f"freq_{k}"].astype(float) for k in ("B1-x", "B1-y", "B2-x", "B2-y")}
    base = pd.DataFrame({
        "f_B1x": f["B1-x"], "f_B1y": f["B1-y"], "f_B2x": f["B2-x"], "f_B2y": f["B2-y"],
        "r_B2B1_x": f["B2-x"] / f["B1-x"], "r_B2B1_y": f["B2-y"] / f["B1-y"],
        "split_B1": 2 * (f["B1-x"] - f["B1-y"]) / (f["B1-x"] + f["B1-y"]),
        "split_B2": 2 * (f["B2-x"] - f["B2-y"]) / (f["B2-x"] + f["B2-y"]),
    })
    return df, base
 
 
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
    """Folds that keep every group wholly inside one fold (as in feature_temperature_damage_analysis.py)."""
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
    """Per column: polynomial of the (standardised) temperature fitted on the finite Healthy rows of X."""
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
    """Compensate (if requested) and median-fill the train and test matrices of one split, or return None if no column is usable."""
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
    """Out-of-fold R2 of temperature (random forest), folds grouped by 7-day block, on the sessions in `subset`."""
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
 
 
def cross_env(unit, Xall, T, y, healthy):
    """Macro F1 and accuracy when training on one temperature range and testing on the other, over all thresholds and both directions."""
    f1s, accs = [], []
    for thr in THRESHOLDS_C:
        low, high = np.where(T < thr)[0], np.where(T >= thr)[0]
        for tr, te in ((low, high), (high, low)):
            if len(tr) < 5 or len(te) < 5 or len(np.unique(y[tr])) < 2:
                continue
            p = prepare(unit, Xall, T, healthy, tr, te)
            if p is None:
                continue
            f, a = macro_acc(y[te], forest().fit(p[0], y[tr]).predict(p[1]))
            f1s.append(f)
            accs.append(a)
    if not f1s:
        return float("nan"), float("nan")
    return float(np.mean(f1s)), float(np.mean(accs))
 
 
def build_units():
    """Define the feature sets (single features and groups) to be screened."""
    raw = ["f_B1x", "f_B1y", "f_B2x", "f_B2y"]
    ratios = ["r_B2B1_x", "r_B2B1_y"]
    splits = ["split_B1", "split_B2"]
    groups = {"freq4": raw, "ratios2": ratios, "splits2": splits, "ratios_splits4": ratios + splits,
              "all8": raw + ratios + splits}
    units = [dict(name="temperature_only (control)", cols=None, comp=False, kind="control")]
    for c in raw + ratios + splits:
        units.append(dict(name=c, cols=[c], comp=False, kind="individual"))
        units.append(dict(name=c + "__comp", cols=[c], comp=True, kind="individual"))
    for g, cols in groups.items():
        units.append(dict(name=g, cols=cols, comp=False, kind="group"))
        units.append(dict(name=g + "__comp", cols=cols, comp=True, kind="group"))
    return units
 
 
def main():
    """Screen each feature set for temperature content and cross-environment damage detection and save the table."""
    df, base = load_table()
    T = df["temperature_degC"].to_numpy(float)
    classes = sorted(df["damage_label"].unique())
    y = np.array([classes.index(c) for c in df["damage_label"]])
    healthy = (df["damage_label"] == "Healthy").to_numpy()
    weeks = week_groups(df["session"])
    if weeks is None:
        raise SystemExit("session names are not SHMTS_YYYYMMDDHHMM; cannot form the 7-day blocks")
    hid = np.where(healthy)[0]
 
    print(f"classes {classes}; chance macro-F1 about {1 / len(classes):.2f}")
    print(f"temperature {T.min():.1f} to {T.max():.1f} C; Healthy sessions {len(hid)} "
          f"({T[hid].min():.1f} to {T[hid].max():.1f} C) in {len(np.unique(weeks[hid]))} weekly blocks; "
          f"{len(np.unique(weeks))} weekly blocks in all")
    print(f"temperature compensation: polynomial degree {COMP_DEGREE} on the Healthy training sessions of each split")
    print("share of sessions with the feature present:")
    print("  " + ", ".join(f"{c} {base[c].notna().mean():.2f}" for c in base.columns))
    same = ((df["freq_B1-x"] == df["freq_B1-y"]) & df["freq_B1-x"].notna()).sum()
    both = (df["freq_B1-x"].notna() & df["freq_B1-y"].notna()).sum()
    print(f"sessions where B1-x and B1-y have the identical frequency: {same} of {both} with both detected"
          + ("   <-- the same pole was given to both labels; split_B1 is 0 there" if same else ""))
    print("  (B1-x / B1-y are assigned by closeness to the nominal frequencies, not by the direction of the shape)\n")
 
    rows = []
    t0 = time.time()
    units = build_units()
    for i, u in enumerate(units, 1):
        Xall = T[:, None] if u["cols"] is None else base[u["cols"]].to_numpy(float)
        r = dict(unit=u["name"], kind=u["kind"], compensated=u["comp"], n_columns=Xall.shape[1])
        r["r2_temp_healthy_week"] = grouped_r2(u, Xall, T, healthy, weeks, hid)
        r["r2_temp_week"] = grouped_r2(u, Xall, T, healthy, weeks, np.arange(len(T)))
        r["f1_cv_week"] = grouped_f1(u, Xall, T, y, healthy, weeks)
        r["f1_cross"], r["acc_cross"] = cross_env(u, Xall, T, y, healthy)
        rows.append(r)
        print(f"[{i:2d}/{len(units)}] {u['name']:30s} R2 temp Healthy-week {r['r2_temp_healthy_week']:+.2f} "
              f"all-week {r['r2_temp_week']:+.2f} | F1 cv-week {r['f1_cv_week']:.2f} cross {r['f1_cross']:.2f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
 
    res = pd.DataFrame(rows)
    ref = res.loc[res.kind == "control", "f1_cross"].iloc[0]
    res["damage_minus_temp"] = res["f1_cross"] - res["r2_temp_healthy_week"].clip(lower=0).fillna(0)
    res["f1_cross_vs_T"] = res["f1_cross"] - ref
    res.insert(0, "n_sessions", len(df))
    res.to_csv(OUT_CSV, index=False)
 
    show = ["unit", "r2_temp_healthy_week", "r2_temp_week", "f1_cv_week", "f1_cross", "acc_cross",
            "f1_cross_vs_T", "damage_minus_temp"]
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(f"\ncontrol: temperature alone gives cross-environment macro-F1 {ref:.3f}")
    for kind in ("individual", "group"):
        t = res[res.kind == kind].sort_values("damage_minus_temp", ascending=False)
        print(f"\n=== {kind} features, sorted by damage_minus_temp (best first) ===")
        print(t[show].round(3).to_string(index=False))
    print(f"\nSaved {OUT_CSV}")
 
 
def cross_env_detail(unit, Xall, T, y, healthy):
    """Like cross_env, but also returns the predictions pooled over the six settings (for per-class scores)."""
    f1s, yt, yp = [], [], []
    for thr in THRESHOLDS_C:
        low, high = np.where(T < thr)[0], np.where(T >= thr)[0]
        for tr, te in ((low, high), (high, low)):
            if len(tr) < 5 or len(te) < 5 or len(np.unique(y[tr])) < 2:
                continue
            p = prepare(unit, Xall, T, healthy, tr, te)
            if p is None:
                continue
            pred = forest().fit(p[0], y[tr]).predict(p[1])
            f1s.append(macro_acc(y[te], pred)[0])
            yt.extend(y[te])
            yp.extend(pred)
    return (float(np.mean(f1s)) if f1s else float("nan")), np.array(yt), np.array(yp)
 
 
def checks():
    """Run four checks on whether the high F1 of the compensated frequencies depends on how missing modes are handled."""
    df, base = load_table()
    T = df["temperature_degC"].to_numpy(float)
    classes = sorted(df["damage_label"].unique())
    y = np.array([classes.index(c) for c in df["damage_label"]])
    healthy = (df["damage_label"] == "Healthy").to_numpy()
    raw = ["f_B1x", "f_B1y", "f_B2x", "f_B2y"]
    sets = [("freq4__comp", raw, True), ("freq4", raw, False), ("B1pair__comp", raw[:2], True),
            ("B2pair__comp", raw[2:], True), ("all8__comp", list(base.columns), True)]
    pd.set_option("display.width", 220, "display.max_columns", 30)
 
    flags = base[raw].isna().to_numpy(float)
    unit = dict(name="detection_flags_only", cols=None, comp=False)
    f1c, yt, yp = cross_env_detail(unit, flags, T, y, healthy)
    print(f"C  detection flags only (no frequency values): cross-environment macro-F1 {f1c:.3f}  "
          f"(this is what the missing-mode pattern alone can give)\n")
 
    rows_cls, rows_cc = [], []
    for name, cols, comp in sets:
        unit = dict(name=name, cols=cols, comp=comp)
        Xall = base[cols].to_numpy(float)
        f1, yt, yp = cross_env_detail(unit, Xall, T, y, healthy)
        per = f1_score(yt, yp, labels=list(range(len(classes))), average=None, zero_division=0)
        rows_cls.append(dict(unit=name, macro_F1=f1, **{c: v for c, v in zip(classes, per)}))
        ok = np.isfinite(Xall).all(1)
        if ok.sum() >= 30 and len(np.unique(y[ok])) >= 3:
            f1cc, ytc, ypc = cross_env_detail(unit, Xall[ok], T[ok], y[ok], healthy[ok])
        else:
            f1cc = float("nan")
        n_by = pd.Series(df["damage_label"][ok]).value_counts()
        rows_cc.append(dict(unit=name, n_complete=int(ok.sum()), macro_F1_all=f1, macro_F1_complete_cases=f1cc,
                            classes_present=int(n_by.size)))
        print(f"{name:14s} all sessions {f1:.3f} | complete cases only ({ok.sum()} sessions, {n_by.size} classes) {f1cc:.3f}",
              flush=True)
    print("\nA  per-class F1, pooled over the six cross-environment settings (all sessions)")
    print(pd.DataFrame(rows_cls).round(2).to_string(index=False))
    print("\nB  all sessions against complete cases")
    print(pd.DataFrame(rows_cc).round(3).to_string(index=False))
    out = os.path.join(HERE, "modal_feature_screen_checks.csv")
    pd.DataFrame(rows_cls).merge(pd.DataFrame(rows_cc), on="unit", suffixes=("", "_cc")).to_csv(out, index=False)
    print(f"\nSaved {out}")
 
 
if __name__ == "__main__":
    if "check" in ARGS:
        checks()
    else:
        main()
 
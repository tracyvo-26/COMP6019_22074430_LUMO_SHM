"""
Builds the mode-shape amplitude and curvature features of every session from the SSI-COV mode shapes, and writes spatial_features.csv.

Needs: ssi_mode_shapes.csv and ssi_modal_frequencies.csv (from 02_modal_identification/SSI-COV).
"""
import os
import numpy as np
import pandas as pd
 
BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "01_data_and_features", "spatial_features"))
INPUT_CSV = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "02_modal_identification", "SSI-COV", "output_batch", "blockrows20_final", "ssi_mode_shapes.csv"))
OUTPUT_CSV = os.path.join(BASE, "spatial_features.csv")
FREQ_CSV = os.path.join(os.path.dirname(INPUT_CSV), "ssi_modal_frequencies.csv")   # all processed sessions
 
# Real measurement heights (m), ML1 (top) ... ML9, from LUMO Table 1
HEIGHTS = np.array([8.95, 8.00, 7.00, 5.95, 5.00, 4.00, 2.95, 2.00, 1.00])
MODES = ["B1-x", "B1-y", "B2-x", "B2-y"]
LEVELS = list(range(1, 10))
 
 
def curvature(phi, z):
    """Second derivative of a mode shape at the interior levels (non-uniform central differences)."""
    out = []
    for i in range(1, len(phi) - 1):
        h1 = abs(z[i] - z[i - 1])      # spacing to the neighbour above
        h2 = abs(z[i + 1] - z[i])      # spacing to the neighbour below
        d_up = (phi[i] - phi[i - 1]) / h1
        d_dn = (phi[i + 1] - phi[i]) / h2
        out.append(2.0 * (d_dn - d_up) / (h1 + h2) * -1.0)
        # factor -1: derivative taken along increasing height, whereas
        # arrays run top -> bottom. Keeps the usual sign convention d2phi/dz2.
    return np.array(out)
 
 
def _reference_shapes(df):
    """Reference shape of each mode: principal eigenvector over all complete sessions, which fixes the sign."""
    refs = {}
    for mode in MODES:
        vecs = []
        for _, g in df[(df["mode_label"] == mode) & (df["direction"] == mode[-1])].groupby("session"):
            if g["level"].nunique() == 9:
                v = g.sort_values("level")["amplitude"].to_numpy(float)
                n = np.linalg.norm(v)
                if n > 0:
                    vecs.append(v / n)
        V = np.array(vecs)
        w, U = np.linalg.eigh(V.T @ V)
        ref = U[:, -1]
        if ref.sum() < 0:
            ref = -ref
        refs[mode] = ref
    return refs
 
 
def main(path, out_path="spatial_features.csv"):
    """Build the amplitude, curvature and detection features of every session and save them."""
    df = pd.read_csv(path)
    sessions = df[["session", "damage_label", "temperature_degC"]].drop_duplicates("session")
    refs = _reference_shapes(df)
    rows = []
    for _, meta in sessions.iterrows():
        rec = {"session": meta["session"], "damage_label": meta["damage_label"],
               "temperature_degC": meta["temperature_degC"]}
        sub_s = df[df["session"] == meta["session"]]
        for mode in MODES:
            d = mode[-1]
            sub = sub_s[(sub_s["mode_label"] == mode) & (sub_s["direction"] == d)]
            tag = mode.replace("-", "")
            ok = sub["level"].nunique() == 9
            rec[f"det_{tag}"] = int(ok)
            if not ok:
                for L in LEVELS:
                    rec[f"amp_{tag}_L{L}"] = np.nan
                for L in LEVELS[1:-1]:
                    rec[f"curv_{tag}_L{L}"] = np.nan
                rec[f"freq_{tag}"] = np.nan
                continue
            sub = sub.sort_values("level")
            phi = sub["amplitude"].to_numpy(float)
            # sign alignment to the mode's reference shape (sign-invariant)
            if phi @ refs[mode] < 0:
                phi = -phi
            m = np.max(np.abs(phi))
            phi = phi / m if m > 0 else phi
            curv = curvature(phi, HEIGHTS)
            rec[f"freq_{tag}"] = float(sub["detected_freq_Hz"].iloc[0])
            for L, a in zip(LEVELS, phi):
                rec[f"amp_{tag}_L{L}"] = a
            for L, c in zip(LEVELS[1:-1], curv):
                rec[f"curv_{tag}_L{L}"] = c
        rows.append(rec)
 
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} sessions -> {out_path}")
 
    print("\nDetection flags by damage class (mean):")
    print(out.groupby("damage_label")[[f"det_{m.replace('-', '')}" for m in MODES]].mean().round(2))
 
    for m in ("B1x", "B2x"):
        cols = [f"curv_{m}_L{L}" for L in LEVELS[1:-1]]
        print(f"\nMean |curvature| per interior level, mode {m}, by damage class:")
        print(out.groupby("damage_label")[cols].apply(lambda g: g.abs().mean()).round(3))
 
 
def run_diagnostics(out, out_dir):
    """Run the robust checks of the spatial features (statistics, boxplots, temperature dependence, matched comparisons)."""
    from scipy import stats
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    order = ["Healthy", "DAM3_010", "DAM4_010", "DAM6_010",
             "DAM3_111", "DAM4_111", "DAM6_111"]
    interior = [f"L{L}" for L in LEVELS[1:-1]]
 
    # per-session summary: mean |curvature| over interior levels
    for m in ("B1x", "B1y", "B2x", "B2y"):
        out[f"curvmean_{m}"] = out[[f"curv_{m}_{l}" for l in interior]].abs().mean(axis=1)
 
    print("\n" + "=" * 70)
    print("1. SESSION COUNT CHECK")
    print("=" * 70)
    if os.path.exists(FREQ_CSV):
        allsess = pd.read_csv(FREQ_CSV)
        col = "session" if "session" in allsess.columns else allsess.columns[0]
        missing = allsess[~allsess[col].isin(out["session"])]
        print(f"Sessions in frequency CSV: {allsess[col].nunique()}, "
              f"in spatial features: {out['session'].nunique()}")
        if len(missing):
            keep = [c for c in (col, "damage_label", "temperature_degC") if c in missing.columns]
            print("Missing sessions (no mode detected at all):")
            print(missing[keep].to_string(index=False))
    else:
        print(f"FREQ_CSV not found ({FREQ_CSV}); skipped.")
 
    print("\n" + "=" * 70)
    print("2. MEDIAN [IQR] of per-session mean |curvature| (n = detected sessions)")
    print("=" * 70)
    for m in ("B1x", "B2x"):
        c = f"curvmean_{m}"
        rows = []
        for cl in order:
            v = out.loc[out["damage_label"] == cl, c].dropna()
            if len(v) == 0:
                rows.append((cl, 0, np.nan, np.nan, np.nan))
            else:
                rows.append((cl, len(v), v.median(), v.quantile(.25), v.quantile(.75)))
        t = pd.DataFrame(rows, columns=["class", "n", "median", "q25", "q75"])
        print(f"\nMode {m}")
        print(t.round(4).to_string(index=False))
        groups = [out.loc[out["damage_label"] == cl, c].dropna() for cl in order]
        groups = [g for g in groups if len(g) > 2]
        if len(groups) > 1:
            h, p = stats.kruskal(*groups)
            print(f"Kruskal-Wallis across classes: H={h:.2f}, p={p:.4g}")
 
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for r, m in enumerate(("B1x", "B2x")):
        for k, lvl in enumerate(("L4", "L5", "L6")):
            ax = axes[r, k]
            data = [out.loc[out["damage_label"] == cl, f"curv_{m}_{lvl}"].abs().dropna()
                    for cl in order]
            ax.boxplot(data, showfliers=True)
            ax.set_xticks(range(1, len(order) + 1))
            ax.set_xticklabels(order)
            ax.set_title(f"{m}  |curvature| at {lvl}")
            ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "spatial_curvature_boxplots.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\nSaved boxplots -> {fig_path}")
 
    print("\n" + "=" * 70)
    print("3. SPEARMAN correlation: mean |curvature| vs temperature")
    print("=" * 70)
    rows = []
    for m in ("B1x", "B2x"):
        c = f"curvmean_{m}"
        for scope in ["ALL"] + order:
            d = out if scope == "ALL" else out[out["damage_label"] == scope]
            d = d[[c, "temperature_degC"]].dropna()
            if len(d) >= 8:
                rho, p = stats.spearmanr(d["temperature_degC"], d[c])
                rows.append((m, scope, len(d), rho, p))
    print(pd.DataFrame(rows, columns=["mode", "scope", "n", "rho", "p"]).round(4).to_string(index=False))
 
    print("\n" + "=" * 70)
    print("4. TEMPERATURE-MATCHED comparison (Healthy restricted to the damage class's temp range)")
    print("=" * 70)
    rows = []
    healthy = out[out["damage_label"] == "Healthy"]
    for m in ("B1x", "B2x"):
        c = f"curvmean_{m}"
        for cl in order[1:]:
            dmg = out[out["damage_label"] == cl].dropna(subset=[c, "temperature_degC"])
            if len(dmg) < 3:
                continue
            lo, hi = dmg["temperature_degC"].min(), dmg["temperature_degC"].max()
            hm = healthy[(healthy["temperature_degC"] >= lo) &
                         (healthy["temperature_degC"] <= hi)].dropna(subset=[c])
            if len(hm) < 3:
                rows.append((m, cl, f"{lo:.1f}-{hi:.1f}", len(hm), len(dmg), np.nan, np.nan, np.nan, np.nan))
                continue
            u, p = stats.mannwhitneyu(dmg[c], hm[c], alternative="two-sided")
            cliff = 2 * u / (len(dmg) * len(hm)) - 1   # >0: damage higher
            rows.append((m, cl, f"{lo:.1f}-{hi:.1f}", len(hm), len(dmg),
                         hm[c].median(), dmg[c].median(), cliff, p))
    print(pd.DataFrame(rows, columns=["mode", "class", "temp_range", "n_healthy",
                                      "n_damage", "med_healthy", "med_damage",
                                      "cliffs_delta", "p"]).round(4).to_string(index=False))
    print("Cliff's delta > 0: damage curvature higher than temperature-matched Healthy; |d|<0.15 negligible, >0.47 large.")
    print("Small n and many tests: treat p-values as descriptive, not confirmatory.")
 
 
# STAGE 2: validation of the spatial features before graph construction
#   A. temperature-residualised comparison (Healthy-fitted)
#   B. per-level (L2-L8) analysis -> localisation hypothesis
#   C. deviation-from-Healthy-reference features (+ 1-MAC)
#   D. simple classifier (does it add information beyond RMS/peak?)
 
# Optional: CSV with the existing per-session features (RMS, peak, ...).
# Needs one row per session and a key column matching `session`
# (e.g. "SHMTS_202011010004.mat"). None skips the baseline comparison.
RUN_VALIDATION = False      # True = run residual / per-level / classifier checks (slow, optional)
BASELINE_CSV = None
BASELINE_KEY = "session"
BASELINE_EXCLUDE = ["session", "damage_label", "temperature_degC",
                    "label", "file", "filename"]
 
TEMP_POLY_DEGREE = 1        # 1 = linear in temperature, 2 = quadratic
INCLUDE_DET_FLAGS = False   # det_* flags can leak the label (DAM6_111 has no B2)
CV_SEEDS = [0, 1, 2, 3, 4]
N_TEMP_BINS = 5             # temperature-grouped CV
RF_TREES = 150
CLASS_ORDER = ["Healthy", "DAM3_010", "DAM4_010", "DAM6_010",
               "DAM3_111", "DAM4_111", "DAM6_111"]
SPATIAL_MODES = ["B1x", "B1y", "B2x", "B2y"]
 
 
def _spatial_cols(modes=SPATIAL_MODES, det=False):
    """Names of the spatial feature columns for the given modes."""
    cols = []
    for m in modes:
        cols += [f"amp_{m}_L{L}" for L in LEVELS]
        cols += [f"curv_{m}_L{L}" for L in LEVELS[1:-1]]
        if det:
            cols.append(f"det_{m}")
    return cols
 
 
def temp_residuals(df, cols, fit_mask=None, deg=None):
    """Residuals from a Huber polynomial fit of temperature on Healthy rows, cross-fitted for the rows used in the fit."""
    deg = TEMP_POLY_DEGREE if deg is None else deg
    T = df["temperature_degC"].to_numpy(float)
    if fit_mask is None:
        fit_mask = (df["damage_label"] == "Healthy").to_numpy()
    fit_mask = np.asarray(fit_mask, bool)
    res = pd.DataFrame(np.nan, index=df.index, columns=cols)
    if fit_mask.sum() < deg + 3:
        return res
    t0 = np.nanmean(T[fit_mask])
    X = np.column_stack([((T - t0) / 10.0) ** k for k in range(deg + 1)])
 
    def huber_fit(Xf, yf, iters=30):
        """Robust (Huber) linear fit by iteratively reweighted least squares."""
        w = np.ones(len(yf))
        b = np.zeros(Xf.shape[1])
        for _ in range(iters):
            sw = np.sqrt(w)
            b = np.linalg.lstsq(Xf * sw[:, None], yf * sw, rcond=None)[0]
            r = yf - Xf @ b
            s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
            u = np.abs(r) / (1.345 * s)
            w = np.where(u <= 1, 1.0, 1.0 / u)
        return b
 
    for c in cols:
        y = df[c].to_numpy(float)
        ok = ~np.isnan(y) & ~np.isnan(T)
        fit = ok & fit_mask
        if fit.sum() < deg + 3:
            continue
        beta = huber_fit(X[fit], y[fit])
        r = y - X @ beta
        idx = np.where(fit)[0]
        k = min(10, len(idx))
        fold = np.arange(len(idx)) % k
        for f in range(k):
            tr = idx[fold != f]
            te = idx[fold == f]
            b = huber_fit(X[tr], y[tr])
            r[te] = y[te] - X[te] @ b
        r[~ok] = np.nan
        res[c] = r
    return res
 
 
def cliffs(a, b):
    """Cliff's delta and Mann-Whitney p-value of two samples."""
    from scipy import stats
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    b = np.asarray(b, float); b = b[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return 2 * u / (len(a) * len(b)) - 1, p
 
 
def add_deviation_scalars(out):
    """Deviation from the Healthy reference shape (median over Healthy)."""
    healthy = out["damage_label"] == "Healthy"
    for m in SPATIAL_MODES:
        a_cols = [f"amp_{m}_L{L}" for L in LEVELS]
        c_cols = [f"curv_{m}_L{L}" for L in LEVELS[1:-1]]
        ref_a = out.loc[healthy, a_cols].median().to_numpy()
        ref_c = out.loc[healthy, c_cols].median().to_numpy()
        A = out[a_cols].to_numpy(float)
        C = out[c_cols].to_numpy(float)
        out[f"devamp_{m}"] = np.linalg.norm(A - ref_a, axis=1)
        out[f"devcurv_{m}"] = np.linalg.norm(C - ref_c, axis=1)
        mac = (A @ ref_a) ** 2 / (np.sum(A * A, axis=1) * (ref_a @ ref_a))
        out[f"omm_{m}"] = 1 - mac          # 1 - MAC vs Healthy reference
    return out
 
 
def stage_A_C_residual_comparison(out):
    """Compare damaged and Healthy sessions on raw and temperature-residualised features."""
    print("\n" + "=" * 70)
    print("5. TEMPERATURE-RESIDUALISED comparison vs Healthy (Cliff's delta, p)")
    print("   raw = as is;  res = after removing Healthy-fitted temperature trend")
    print("=" * 70)
    out = add_deviation_scalars(out)
    scal = []
    for m in SPATIAL_MODES:
        scal += [f"curvmean_{m}", f"devcurv_{m}", f"devamp_{m}", f"omm_{m}"]
    R = temp_residuals(out, scal)
    healthy = (out["damage_label"] == "Healthy").to_numpy()
 
    # sanity: residual of Healthy should no longer correlate with temperature
    from scipy import stats
    chk = []
    for c in scal:
        d = pd.DataFrame({"t": out.loc[healthy, "temperature_degC"], "r": R.loc[healthy, c]}).dropna()
        if len(d) > 8:
            chk.append((c, stats.spearmanr(d["t"], d["r"])[0],
                        stats.spearmanr(d["t"], out.loc[d.index, c])[0]))
    print("Spearman with temperature within Healthy (raw -> residual):")
    print(pd.DataFrame(chk, columns=["feature", "rho_resid", "rho_raw"]).round(3).to_string(index=False))
 
    rows = []
    for c in scal:
        for cl in CLASS_ORDER[1:]:
            dm = (out["damage_label"] == cl).to_numpy()
            d_raw, p_raw = cliffs(out.loc[dm, c], out.loc[healthy, c])
            d_res, p_res = cliffs(R.loc[dm, c], R.loc[healthy, c])
            rows.append((c, cl, d_raw, p_raw, d_res, p_res))
    t = pd.DataFrame(rows, columns=["feature", "class", "d_raw", "p_raw", "d_res", "p_res"])
    pd.set_option("display.max_rows", 500)
    print(t.round(3).to_string(index=False))
    n_tests = t["p_res"].notna().sum()
    print(f"\n{n_tests} tests; Bonferroni threshold on residualised p = {0.05 / max(n_tests, 1):.4f}")
    sig = t[(t["p_res"] < 0.05 / max(n_tests, 1))]
    print("Residualised effects surviving Bonferroni:")
    print(sig.round(4).to_string(index=False) if len(sig) else "  none")
    return out
 
 
def stage_B_per_level(out, out_dir):
    """Compare damaged and Healthy sessions level by level on the temperature-residualised curvature."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    print("\n" + "=" * 70)
    print("6. PER-LEVEL curvature, temperature-residualised, Cliff's delta vs Healthy")
    print("   (damage heights are not hard-coded)")
    print("=" * 70)
    healthy = (out["damage_label"] == "Healthy").to_numpy()
    classes = CLASS_ORDER[1:]
    interior = LEVELS[1:-1]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, m in zip(axes.ravel(), SPATIAL_MODES):
        cols = [f"curv_{m}_L{L}" for L in interior]
        R = temp_residuals(out, cols)
        delta = pd.DataFrame(np.nan, index=classes, columns=[f"L{L}" for L in interior])
        pval = delta.copy()
        for cl in classes:
            dm = (out["damage_label"] == cl).to_numpy()
            for L, c in zip(interior, cols):
                d, p = cliffs(R.loc[dm, c], R.loc[healthy, c])
                delta.loc[cl, f"L{L}"] = d
                pval.loc[cl, f"L{L}"] = p
        print(f"\nMode {m}: Cliff's delta (rows = class, cols = measurement level; "
              f"heights m: " + ", ".join(f"L{L}={HEIGHTS[L-1]:.2f}" for L in interior) + ")")
        print(delta.round(2).to_string())
        for cl in classes:
            row = delta.loc[cl].abs()
            if row.notna().any():
                L = row.idxmax()
                print(f"  {cl:<9} strongest at {L} ({HEIGHTS[int(L[1:]) - 1]:.2f} m), "
                      f"delta={delta.loc[cl, L]:+.2f}, p={pval.loc[cl, L]:.3g}")
        im = ax.imshow(delta.to_numpy(float), vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(range(len(interior)))
        ax.set_xticklabels([f"L{L}\n{HEIGHTS[L-1]:.1f}m" for L in interior])
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes)
        for i in range(len(classes)):
            for j in range(len(interior)):
                p = pval.iloc[i, j]
                if pd.notna(p) and p < 0.05 / delta.size:
                    ax.text(j, i, "**", ha="center", va="center")
                elif pd.notna(p) and p < 0.05:
                    ax.text(j, i, "*", ha="center", va="center")
        ax.set_title(f"{m}: residualised curvature, Cliff's delta")
    fig.colorbar(im, ax=axes, shrink=0.8)
    path = os.path.join(out_dir, "spatial_per_level_delta.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n* p<0.05 (uncorrected), ** passes Bonferroni. Saved -> {path}")
 
 
def stage_D_classifier(out, out_dir):
    """Test whether the spatial features add information beyond the baseline features, with cross-validated classifiers."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, GroupKFold
    from sklearn.metrics import f1_score, balanced_accuracy_score
 
    print("\n" + "=" * 70)
    print("7. CLASSIFIER CHECK (does the spatial block add information?)")
    print("=" * 70)
    df = out.reset_index(drop=True).copy()
    spatial = _spatial_cols(det=INCLUDE_DET_FLAGS)
 
    base_cols = []
    if BASELINE_CSV and os.path.exists(BASELINE_CSV):
        b = pd.read_csv(BASELINE_CSV)
        print(f"Baseline CSV: {len(b)} rows, columns = {list(b.columns)[:25]}"
              + (" ..." if len(b.columns) > 25 else ""))
        # auto-detect the session key: a column whose values look like SHMTS_*.mat
        key = BASELINE_KEY if BASELINE_KEY in b.columns else None
        if key is None:
            for c in b.columns:
                if b[c].dtype == object and b[c].astype(str).str.contains("SHMTS_").any():
                    key = c
                    break
        if key is not None:
            b = b.copy()
            b["_key"] = b[key].astype(str).str.extract(r"(SHMTS_\d+)")[0]
            df["_key"] = df["session"].astype(str).str.extract(r"(SHMTS_\d+)")[0]
            dup = b["_key"].duplicated().sum()
            if dup:
                print(f"WARNING: {dup} duplicated session keys in baseline CSV "
                      f"(per-channel/long format?). Keeping first row per session; "
                      f"aggregate to one row per session for a fair comparison.")
            b = b.drop_duplicates("_key")
            base_cols = [c for c in b.select_dtypes("number").columns
                         if c not in BASELINE_EXCLUDE and c not in df.columns]
            df = df.merge(b[["_key"] + base_cols], on="_key", how="left")
            matched = int(df[base_cols].notna().any(axis=1).sum()) if base_cols else 0
            print(f"Using session key '{key}': {len(base_cols)} baseline columns, "
                  f"matched {matched}/{len(df)} sessions")
            if matched < 0.8 * len(df):
                print("Too few matches; baseline skipped.")
                base_cols = []
        else:
            print("No column containing 'SHMTS_...' found in the baseline CSV; baseline skipped.")
    else:
        print("BASELINE_CSV not set: comparing spatial vs temperature-only instead.")
 
    spatial_b1 = _spatial_cols(["B1x", "B1y"], det=False)
    spatial_b2 = _spatial_cols(["B2x", "B2y"], det=False)
    # spec = (name, baseline_cols, spatial_cols or None, "raw"/"res"/"none")
    fsets = [("temp_only", None, None, "none"),
             ("spatial_raw", None, spatial, "raw"),
             ("spatial_res", None, spatial, "res"),
             ("spatialB1_raw", None, spatial_b1, "raw"),     # B1 only (no missing-mode pattern)
             ("spatialB2_raw", None, spatial_b2, "raw")]     # B2 only (DAM6_111 has none)
    if base_cols:
        fsets = [("baseline", base_cols, None, "none")] + fsets + [
            ("baseline+spatial_raw", base_cols, spatial, "raw"),
            ("baseline+spatial_res", base_cols, spatial, "res"),
            ("baseline+spatialB1_raw", base_cols, spatial_b1, "raw")]
 
    y_multi = df["damage_label"].to_numpy()
    y_bin = (df["damage_label"] != "Healthy").astype(int).to_numpy()
    temp_bin = pd.qcut(df["temperature_degC"], N_TEMP_BINS, labels=False, duplicates="drop").to_numpy()
    # recording day from the file name (SHMTS_YYYYMMDDhhmm): neighbouring 10-min
    # sessions of the same day/campaign share conditions -> group them in CV
    day = df["session"].astype(str).str.extract(r"SHMTS_(\d{8})")[0].fillna("unknown").to_numpy()
 
    def make_X(tr, te, spec):
        """Build the train and test matrices of a feature specification."""
        name, bcols, scols, mode = spec
        parts_tr, parts_te = [], []
        if name == "temp_only":
            return df.loc[tr, ["temperature_degC"]].to_numpy(float), df.loc[te, ["temperature_degC"]].to_numpy(float)
        if bcols:
            parts_tr.append(df.loc[tr, bcols].to_numpy(float))
            parts_te.append(df.loc[te, bcols].to_numpy(float))
        if scols:
            if mode == "raw":
                parts_tr.append(df.loc[tr, scols].to_numpy(float))
                parts_te.append(df.loc[te, scols].to_numpy(float))
            else:
                sub = df.iloc[np.concatenate([tr, te])]
                fit_mask = np.r_[(df["damage_label"].iloc[tr] == "Healthy").to_numpy(),
                                 np.zeros(len(te), bool)]
                R = temp_residuals(sub, scols, fit_mask=fit_mask)
                parts_tr.append(R.iloc[:len(tr)].to_numpy(float))
                parts_te.append(R.iloc[len(tr):].to_numpy(float))
        return np.hstack(parts_tr), np.hstack(parts_te)
 
    def models(seed):
        """Return the random forest and logistic regression models used in the classifier test."""
        return {
            "RF": make_pipeline(SimpleImputer(strategy="median"),
                                RandomForestClassifier(n_estimators=RF_TREES, min_samples_leaf=2,
                                                       class_weight="balanced_subsample",
                                                       random_state=seed, n_jobs=-1)),
            "LR": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                LogisticRegression(max_iter=3000, C=0.3,
                                                   class_weight="balanced")),
        }
 
    records = []
    recall_pred = {}          # pooled RF multiclass predictions (seed 0) for per-class recall
    for scheme in ("random5fold", "daygroup", "tempgroup"):
        for seed in CV_SEEDS:
            if scheme == "random5fold":
                splits = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(df, y_multi))
            elif scheme == "daygroup":
                splits = list(GroupKFold(5).split(df, y_multi, day))
            else:
                splits = list(GroupKFold(len(np.unique(temp_bin))).split(df, y_multi, temp_bin))
            pred = {(f[0], t, mname): np.empty(len(df), dtype=object)
                    for f in fsets for t in ("binary", "multiclass") for mname in ("RF", "LR")}
            for tr, te in splits:
                for spec in fsets:
                    Xtr, Xte = make_X(tr, te, spec)
                    for tname, y in (("binary", y_bin), ("multiclass", y_multi)):
                        for mname, mdl in models(seed).items():
                            mdl.fit(Xtr, y[tr])
                            pred[(spec[0], tname, mname)][te] = mdl.predict(Xte)
            for (fs, tname, mname), p in pred.items():
                y = y_bin if tname == "binary" else y_multi
                p = p.astype(int) if tname == "binary" else p.astype(str)
                records.append((scheme, tname, fs, mname, seed,
                                f1_score(y, p, average="macro", zero_division=0),
                                balanced_accuracy_score(y, p)))
                if seed == CV_SEEDS[0] and tname == "multiclass" and mname == "RF":
                    recall_pred[(scheme, fs)] = p
 
        # label-permutation control: same pipeline, shuffled labels -> should sit at chance
        if scheme in ("random5fold", "daygroup"):
            spec = [s for s in fsets if s[0] == "spatial_raw"][0]
            for seed in CV_SEEDS[:3]:
                rng = np.random.default_rng(100 + seed)
                yb_p, ym_p = rng.permutation(y_bin), rng.permutation(y_multi)
                if scheme == "random5fold":
                    splits = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(df, ym_p))
                else:
                    splits = list(GroupKFold(5).split(df, ym_p, day))
                pb = np.empty(len(df), dtype=object)
                pm = np.empty(len(df), dtype=object)
                for tr, te in splits:
                    Xtr, Xte = make_X(tr, te, spec)
                    pb[te] = models(seed)["RF"].fit(Xtr, yb_p[tr]).predict(Xte)
                    pm[te] = models(seed)["RF"].fit(Xtr, ym_p[tr]).predict(Xte)
                for tname, y, p in (("binary", yb_p, pb.astype(int)), ("multiclass", ym_p, pm.astype(str))):
                    records.append((scheme, tname, "PERMUTED_labels(spatial_raw)", "RF", seed,
                                    f1_score(y, p, average="macro", zero_division=0),
                                    balanced_accuracy_score(y, p)))
 
    # per-class recall (RF, multiclass, seed 0): who is actually being recognised?
    from sklearn.metrics import recall_score
    print("\nPer-class recall, RF multiclass, seed 0 (spatial_raw vs B1-only vs B2-only):")
    rows = []
    for scheme in ("random5fold", "daygroup", "tempgroup"):
        for fs in ("spatial_raw", "spatialB1_raw", "spatialB2_raw"):
            p = recall_pred.get((scheme, fs))
            if p is None:
                continue
            rec = recall_score(y_multi, p, labels=CLASS_ORDER, average=None, zero_division=0)
            rows.append([scheme, fs] + list(np.round(rec, 2)))
    print(pd.DataFrame(rows, columns=["cv", "features"] + CLASS_ORDER).to_string(index=False))
    res = pd.DataFrame(records, columns=["scheme", "target", "features", "model",
                                         "seed", "macro_f1", "bal_acc"])
    res.to_csv(os.path.join(out_dir, "spatial_classifier_results_by_seed.csv"), index=False)
    summ = (res.groupby(["scheme", "target", "features", "model"])[["macro_f1", "bal_acc"]]
            .agg(["mean", "std"]).round(3))
    summ.to_csv(os.path.join(out_dir, "spatial_classifier_results.csv"))
    pd.set_option("display.width", 220)
    for (scheme, tname), g in summ.groupby(level=[0, 1]):
        print(f"\n--- CV: {scheme} | target: {tname} (mean, std over {len(CV_SEEDS)} seeds) ---")
        print(g.droplevel([0, 1]).to_string())
 
 
def run_validation(out, out_dir):
    """Run the residual, per-level and classifier checks of the spatial features."""
    out = stage_A_C_residual_comparison(out)
    stage_B_per_level(out, out_dir)
    stage_D_classifier(out, out_dir)
 
 
if __name__ == "__main__":
    main(INPUT_CSV, OUTPUT_CSV)
    out_df = pd.read_csv(OUTPUT_CSV)
    run_diagnostics(out_df, os.path.dirname(OUTPUT_CSV))   # simple descriptive stats
    if RUN_VALIDATION:                                     # heavy optional checks
        run_validation(out_df, os.path.dirname(OUTPUT_CSV))
 
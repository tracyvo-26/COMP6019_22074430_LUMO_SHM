
"""
Compares the FDD mode frequencies with temperature and with damage state.

Needs: the FDD frequency table (fdd_modal_frequencies_nperseg16384.csv) written by the FDD scripts.
"""
 
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
 
CSV_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "02_modal_identification", "FDD", "output_optimized_nperseg", "fdd_modal_frequencies_nperseg16384.csv"))
OUT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "02_modal_identification", "FDD", "temp_analysis", "16384"))
os.makedirs(OUT_DIR, exist_ok=True)
 
df = pd.read_csv(CSV_PATH)
df["freq_B1"] = df["freq_B1-x"]  # B1-y is identical to B1-x, keep one merged column
 
GROUPS = ["Healthy", "DAM3_010", "DAM3_111", "DAM4_010", "DAM4_111", "DAM6_010", "DAM6_111"]
FEATURES = [
    ("freq_B1", "B1 (modes 1-2, merged, not yet separable)"),
    ("freq_T1", "T1 (mode 3, torsional)"),
    ("freq_B2-y", "B2-y (modes 4-5, y direction)"),
    ("freq_B2-x", "B2-x (modes 4-5, x direction)"),
]
COLORS = {
    "Healthy": "tab:gray",
    "DAM3_010": "tab:blue", "DAM3_111": "tab:cyan",
    "DAM4_010": "tab:orange", "DAM4_111": "tab:red",
    "DAM6_010": "tab:green", "DAM6_111": "tab:purple",
}
MARKERS = {
    "Healthy": "o",
    "DAM3_010": "s", "DAM3_111": "s",
    "DAM4_010": "^", "DAM4_111": "^",
    "DAM6_010": "D", "DAM6_111": "D",
}
 
# Part 1: regression, frequency vs temperature, per damage state
 
 
def fit(sub, feat_col):
    """Linear regression of a feature on temperature for one subset."""
    valid = sub[[feat_col, "temperature_degC"]].dropna()
    x = valid["temperature_degC"].values
    y = valid[feat_col].values
    if len(x) < 3 or np.std(x) == 0:
        return None
    slope, intercept, r, p, se = stats.linregress(x, y)
    return dict(n=len(x), temp_min=round(x.min(), 1), temp_max=round(x.max(), 1),
                slope=slope, r=r, r2=r ** 2, p=p)
 
 
rows = []
for g in GROUPS:
    sub = df[df["damage_label"] == g]
    for feat_col, feat_label in FEATURES:
        res = fit(sub, feat_col)
        n_detected = sub[feat_col].notna().sum()
        if res:
            rows.append(dict(group=g, feature=feat_label, n_total=len(sub),
                              n_detected=n_detected, **res))
        else:
            rows.append(dict(group=g, feature=feat_label, n_total=len(sub),
                              n_detected=n_detected, n=None, temp_min=None,
                              temp_max=None, slope=None, r=None, r2=None, p=None))
 
sub = df[df["damage_label"].isin(["DAM3_010", "DAM3_111"])]
for feat_col, feat_label in FEATURES:
    res = fit(sub, feat_col)
    n_detected = sub[feat_col].notna().sum()
    if res:
        rows.append(dict(group="DAM3_combined", feature=feat_label, n_total=len(sub),
                          n_detected=n_detected, **res))
 
regression_out = pd.DataFrame(rows)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", None)
print("=== Regression results: FDD frequency vs temperature ===")
print(regression_out.to_string(index=False))
 
regression_csv = f"{OUT_DIR}/fdd_temperature_regression.csv"
regression_out.to_csv(regression_csv, index=False)
print(f"\nSaved to {regression_csv}")
 
# Part 2: Healthy-vs-Damaged effect size within matching temperature bins
 
TEMP_BINS = [-float("inf"), 0, 10, 20, 30, float("inf")]
TEMP_BIN_LABELS = ["<0", "0-10", "10-20", "20-30", ">30"]
 
df["temp_bin"] = pd.cut(df["temperature_degC"], bins=TEMP_BINS, labels=TEMP_BIN_LABELS)
df["is_healthy"] = df["damage_label"].str.lower() == "healthy"
 
effect_rows = []
for feat_col, feat_label in FEATURES:
    for temp_bin, group in df.groupby("temp_bin", observed=True):
        healthy_vals = group.loc[group["is_healthy"], feat_col].dropna()
        damaged_vals = group.loc[~group["is_healthy"], feat_col].dropna()
        if len(healthy_vals) < 2 or len(damaged_vals) < 2:
            continue
        pooled_std = np.std(pd.concat([healthy_vals, damaged_vals]))
        if pooled_std == 0 or np.isnan(pooled_std):
            continue
        effect_size = abs(damaged_vals.mean() - healthy_vals.mean()) / pooled_std
        effect_rows.append({"feature": feat_label, "temp_bin": str(temp_bin),
                             "n_healthy": len(healthy_vals), "n_damaged": len(damaged_vals),
                             "effect_size": effect_size})
 
effect_out = pd.DataFrame(effect_rows)
if effect_out.empty:
    print("\nNo temperature bin had both Healthy and Damaged sessions with a detected "
          "frequency to compare -- worth reporting as a coverage gap in its own right.")
else:
    ranked = (effect_out.groupby("feature")["effect_size"]
              .agg(["mean", "std", "count"]).sort_values("mean", ascending=False))
    print("\n=== Healthy-vs-Damaged effect size by mode, within matching temperature bins ===")
    print(ranked)
    effect_csv = f"{OUT_DIR}/fdd_damage_effect_size.csv"
    effect_out.to_csv(effect_csv, index=False)
    print(f"\nSaved to {effect_csv}")
 
# Part 3: scatter plots, one per mode, all damage states overlaid
 
for feat_col, feat_label in FEATURES:
    plt.figure(figsize=(8, 5))
    for g in GROUPS:
        sub = df[df["damage_label"] == g].dropna(subset=[feat_col])
        if len(sub) == 0:
            continue
        plt.scatter(sub["temperature_degC"], sub[feat_col], label=g,
                     color=COLORS[g], marker=MARKERS[g], s=60,
                     edgecolor="black", linewidth=0.5)
    plt.xlabel("Temperature (°C)")
    plt.ylabel(f"{feat_label} frequency (Hz)")
    plt.title(f"{feat_label} vs Temperature")
    plt.legend(title="Damage state", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fname = f"{OUT_DIR}/{feat_col}_vs_temperature.png"
    plt.savefig(fname, dpi=150)
    print(f"saved {fname}")
    plt.close()

"""
Scans between_channel_features.csv for sessions whose feature values look like outliers.

Needs: between_channel_features.csv (from between_channel_features.py).
"""
 
import numpy as np
import pandas as pd
import os
 
CSV_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "01_data_and_features", "between_channel and temp explore", "between_channel_features", "between_channel_features.csv"))
OUT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "01_data_and_features", "between_channel and temp explore", "between_channel_features")) 
Z_THRESHOLD = 2.0
 
df = pd.read_csv(CSV_PATH)
 
FEATURES = ["xcov", "coh_b1", "coh_t1", "coh_b2", "trans_b1", "trans_t1", "trans_b2"]
GROUPS = ["Healthy", "DAM3_010", "DAM3_111", "DAM4_010", "DAM4_111", "DAM6_010", "DAM6_111"]
 
# pool each feature across all edges into one value per session first
for feat in FEATURES:
    cols = [c for c in df.columns if c.endswith(f"_{feat}")]
    df[f"pooled_{feat}"] = df[cols].mean(axis=1)
 
rows = []
for feat in FEATURES:
    pooled_col = f"pooled_{feat}"
    # z-score against the WHOLE distribution, not per damage group
    mean, std = df[pooled_col].mean(), df[pooled_col].std()
    df[f"z_{feat}"] = (df[pooled_col] - mean) / std
 
    for g in GROUPS:
        sub = df[df["damage_label"] == g]
        z = sub[f"z_{feat}"]
        flagged = sub.loc[z.abs() >= Z_THRESHOLD, ["session", f"z_{feat}"]]
        rows.append({
            "damage_state": g,
            "feature": feat,
            "n": len(sub),
            "z_min": round(z.min(), 2) if len(sub) else None,
            "z_max": round(z.max(), 2) if len(sub) else None,
            "n_outliers": len(flagged),
            "outlier_sessions": "; ".join(
                f"{row.session} (z={row[f'z_{feat}']:.2f})" for _, row in flagged.iterrows()
            ),
        })
 
summary = pd.DataFrame(rows)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)
print(summary.to_string(index=False))

summary.to_csv(f"{OUT_DIR}/outlier_scan_full.csv", index=False) 
print(f"\nSaved full outlier scan")
 
print("\n=== Outlier count by feature (summed across all damage states) ===")
print(summary.groupby("feature")["n_outliers"].sum().sort_values(ascending=False).to_string())
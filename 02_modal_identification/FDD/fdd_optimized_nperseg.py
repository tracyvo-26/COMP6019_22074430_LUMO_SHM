"""
Tries several window lengths for FDD and keeps the one that detects the modes best in every damage state.

Needs: fdd_modal_identification.py.
"""
 
import os
import numpy as np
import pandas as pd
 
from fdd_modal_identification import (
    DEFAULT_PATH, FMAX, PEAK_TOLERANCE, ACCEL_CHANNELS, KNOWN_MODES,
    extract_fdd_features, plot_average_mode_shapes,
)
 
NPERSEG_CANDIDATES = [4096, 16384, 65536]
AUTO_SELECT = True          # False -> use MANUAL_NPERSEG instead of auto-picking
MANUAL_NPERSEG = 16384      # only used when AUTO_SELECT is False
OUT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "02_modal_identification", "FDD", "output_optimized_nperseg"))
 
os.makedirs(OUT_DIR, exist_ok=True)
 
# B1-y and B1-x are excluded from scoring because FDD merges the two B1 modes.
MODES_TO_SCORE = [label for _, label in KNOWN_MODES[:5] if not label.startswith("B1")]
 
 
def detection_rate_tables(freq_df, nperseg):
    """Compute the overall and per damage state detection rates of each mode."""
    freq_cols = [c for c in freq_df.columns if c.startswith("freq_")]
 
    overall_rows = []
    for col in freq_cols:
        mode = col.replace("freq_", "")
        overall_rows.append({
            "nperseg": nperseg, "mode": mode,
            "n_sessions": len(freq_df), "detection_rate": freq_df[col].notna().mean(),
        })
 
    by_damage_rows = []
    for damage_label, group in freq_df.groupby("damage_label"):
        for col in freq_cols:
            mode = col.replace("freq_", "")
            by_damage_rows.append({
                "nperseg": nperseg, "damage_label": damage_label, "mode": mode,
                "n_sessions": len(group), "detection_rate": group[col].notna().mean(),
            })
 
    return pd.DataFrame(overall_rows), pd.DataFrame(by_damage_rows)
 
 
def score_nperseg(by_damage_df, nperseg):
    """Summarise the detection rates of one window length by their minimum and mean."""
    sub = by_damage_df[(by_damage_df["nperseg"] == nperseg) &
                        (by_damage_df["mode"].isin(MODES_TO_SCORE))]
    return {
        "nperseg": nperseg,
        "min_detection_rate": sub["detection_rate"].min(),
        "mean_detection_rate": sub["detection_rate"].mean(),
    }
 
 
def main():
    """Scan the candidate window lengths, choose the one with the best detection rates and run the full extraction with it."""
    all_overall = []
    all_by_damage = []
 
    for nperseg in NPERSEG_CANDIDATES:
        # frequencies only needed for this scan, mode shapes are computed
        # once at the end for whichever nperseg gets chosen, no point
        # paying that cost for every candidate here
        freq_df, _ = extract_fdd_features(root_path=DEFAULT_PATH, fmax=FMAX,
                                           tolerance=PEAK_TOLERANCE, nperseg=nperseg)
        print(f"\n=== Scanned nperseg={nperseg} across {len(freq_df)} sessions ===")
        freq_df.to_csv(f"{OUT_DIR}/fdd_modal_frequencies_nperseg{nperseg}.csv", index=False)
 
        overall, by_damage = detection_rate_tables(freq_df, nperseg)
        all_overall.append(overall)
        all_by_damage.append(by_damage)
 
    overall_df = pd.concat(all_overall, ignore_index=True)
    by_damage_df = pd.concat(all_by_damage, ignore_index=True)
 
    overall_csv = f"{OUT_DIR}/detection_rate_by_mode.csv"
    by_damage_csv = f"{OUT_DIR}/detection_rate_by_mode_and_damage.csv"
    overall_df.to_csv(overall_csv, index=False)
    by_damage_df.to_csv(by_damage_csv, index=False)
 
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", None)
 
    print("\n=== Detection rate by mode, all nperseg settings ===")
    print(overall_df.pivot(index="mode", columns="nperseg", values="detection_rate").to_string())
    print(f"Saved -> {overall_csv}")
 
    print("\n=== Detection rate by mode AND damage state, all nperseg settings ===")
    print(by_damage_df.pivot_table(index=["damage_label", "mode"], columns="nperseg",
                                    values="detection_rate").to_string())
    print(f"Saved -> {by_damage_csv}")
 
    scores = pd.DataFrame([score_nperseg(by_damage_df, n) for n in NPERSEG_CANDIDATES])
    scores = scores.sort_values(["min_detection_rate", "mean_detection_rate"], ascending=False)
    print("\n=== nperseg ranking (B1 excluded, worst-case detection rate first) ===")
    print(scores.to_string(index=False))
    scores.to_csv(f"{OUT_DIR}/nperseg_scores.csv", index=False)
 
    if AUTO_SELECT:
        chosen_nperseg = int(scores.iloc[0]["nperseg"])
        print(f"\nAuto-selected nperseg={chosen_nperseg} "
              f"(highest worst-case detection rate across damage groups and modes, B1 excluded).")
    else:
        chosen_nperseg = MANUAL_NPERSEG
        print(f"\nAUTO_SELECT is False, using MANUAL_NPERSEG={chosen_nperseg} as set above.")
 
    # Full extraction (frequencies + mode shapes + plots) at the
    # chosen nperseg, same as fdd_modal_identification.py's __main__
    final_dir = f"{OUT_DIR}/nperseg{chosen_nperseg}_final"
    os.makedirs(final_dir, exist_ok=True)
 
    print(f"\n=== Running full extraction at nperseg={chosen_nperseg} -> {final_dir} ===")
    freq_df, shape_df = extract_fdd_features(root_path=DEFAULT_PATH, fmax=FMAX,
                                              tolerance=PEAK_TOLERANCE, nperseg=chosen_nperseg)
    freq_df.to_csv(f"{final_dir}/fdd_modal_frequencies.csv", index=False)
    shape_df.to_csv(f"{final_dir}/fdd_mode_shapes.csv", index=False)
 
    freq_cols = [c for c in freq_df.columns if c.startswith("freq_")]
    print("\n=== Final detection rate per mode, chosen nperseg ===")
    print(freq_df[freq_cols].notna().mean().to_string())
 
    plot_average_mode_shapes(shape_df, out_dir=final_dir)
    print(f"\nDone. All outputs saved under {os.path.abspath(OUT_DIR)}/")
 
 
if __name__ == "__main__":
    main()
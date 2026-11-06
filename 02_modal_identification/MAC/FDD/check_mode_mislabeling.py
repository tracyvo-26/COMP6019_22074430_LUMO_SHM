"""
Checks whether sessions with a low MAC value are explained by a mode being matched to the wrong peak.

Needs: mac_validation.py (its output csv), fdd_mode_shapes.csv and fdd_modal_identification.py.
"""
 
import os
import pandas as pd
import sys

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "02_modal_identification", "FDD")))
 
from fdd_modal_identification import KNOWN_MODES  # same table both methods use
 
MAC_CSV_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "02_modal_identification", "MAC", "FDD", "output_validation", "fdd_mac_vs_reference.csv"))
SHAPE_CSV_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "02_modal_identification", "FDD", "output_optimized_nperseg", "nperseg16384_final", "fdd_mode_shapes.csv"))
METHOD_LABEL = "FDD"
LOW_MAC_THRESHOLD = 0.3   # sessions at or below this MAC are checked for mislabeling
OUT_CSV = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "02_modal_identification", "MAC", "FDD", "output_mislabeling", "mislabeling_check.csv"))
 
MODE_FREQS = {label: freq for freq, label in KNOWN_MODES}
 
 
def nearest_other_mode(freq, own_label):
    """Return the known modes other than the assigned one, with their frequency distances."""
    own_dist = abs(freq - MODE_FREQS[own_label])
    best_other_label, best_other_dist = None, float("inf")
    for label, target in MODE_FREQS.items():
        if label == own_label:
            continue
        d = abs(freq - target)
        if d < best_other_dist:
            best_other_label, best_other_dist = label, d
    return own_dist, best_other_label, best_other_dist
 
 
def main():
    """For sessions with a low MAC value, check whether the detected frequency is closer to a different known mode."""
    mac_df = pd.read_csv(MAC_CSV_PATH)
    shape_df = pd.read_csv(SHAPE_CSV_PATH)
 
    # one row per session per mode with its detected frequency (shape_df has
    # one row per channel, so just take the first, detected_freq_Hz is the
    # same for every channel within a session+mode)
    freq_lookup = (shape_df.groupby(["session", "mode_label"])["detected_freq_Hz"]
                   .first().reset_index())
 
    merged = mac_df.merge(freq_lookup, on=["session", "mode_label"], how="left")
    low_mac = merged[merged["mac"] <= LOW_MAC_THRESHOLD].copy()
 
    if low_mac.empty:
        print(f"No sessions at or below MAC={LOW_MAC_THRESHOLD} for {METHOD_LABEL}, nothing to check.")
        return
 
    rows = []
    for _, r in low_mac.iterrows():
        own_dist, other_label, other_dist = nearest_other_mode(r["detected_freq_Hz"], r["mode_label"])
        likely_mislabeled = other_dist < own_dist
        rows.append({
            "session": r["session"], "damage_label": r["damage_label"],
            "mode_label": r["mode_label"], "mac": r["mac"],
            "detected_freq_Hz": r["detected_freq_Hz"],
            "distance_to_own_target_Hz": own_dist,
            "closer_to_mode": other_label,
            "distance_to_that_mode_Hz": other_dist,
            "likely_mislabeled": likely_mislabeled,
        })
 
    result = pd.DataFrame(rows).sort_values("mac")
    result.to_csv(OUT_CSV, index=False)
 
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", None)
    print(f"=== {METHOD_LABEL}: {len(result)} sessions with MAC <= {LOW_MAC_THRESHOLD} ===\n")
    print(result.to_string(index=False))
 
    n_mislabeled = result["likely_mislabeled"].sum()
    print(f"\n{n_mislabeled} of {len(result)} low-MAC sessions ({n_mislabeled/len(result):.0%}) "
          f"were actually closer to a different known mode than the one they were labeled as.")
    print(f"Saved -> {OUT_CSV}")
 
 
 
if __name__ == "__main__":
    main()
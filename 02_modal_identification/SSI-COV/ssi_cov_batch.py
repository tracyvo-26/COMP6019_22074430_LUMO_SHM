"""
Runs SSI-COV on all sessions with several settings, keeps the best setting, and writes ssi_modal_frequencies.csv and ssi_mode_shapes.csv.

Needs: ssi_cov_identification.py.
"""

import os
import numpy as np
import pandas as pd

from ssi_cov_identification import (
    DEFAULT_PATH, FMAX, PEAK_TOLERANCE, ACCEL_CHANNELS, KNOWN_MODES, FS, TARGET_FS,
    extract_ssi_features, plot_average_mode_shapes, run_single_session_diagnostics,
    load_session, damage_label_from_path, run_stabilization, label_pole,
)

BLOCK_ROWS_CANDIDATES = [20, 40, 80]
AUTO_SELECT = True
MANUAL_BLOCK_ROWS = 40
OUT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "02_modal_identification", "SSI-COV", "output_batch"))

os.makedirs(OUT_DIR, exist_ok=True)

MODES_TO_SCORE = [label for _, label in KNOWN_MODES[:5] if not label.startswith("B1")]


def detection_rate_tables(freq_df, block_rows):
    """Compute the overall and per damage state detection rates of each mode."""
    freq_cols = [c for c in freq_df.columns if c.startswith("freq_")]

    overall_rows = []
    for col in freq_cols:
        mode = col.replace("freq_", "")
        overall_rows.append({
            "block_rows": block_rows, "mode": mode,
            "n_sessions": len(freq_df), "detection_rate": freq_df[col].notna().mean(),
        })

    by_damage_rows = []
    for damage_label, group in freq_df.groupby("damage_label"):
        for col in freq_cols:
            mode = col.replace("freq_", "")
            by_damage_rows.append({
                "block_rows": block_rows, "damage_label": damage_label, "mode": mode,
                "n_sessions": len(group), "detection_rate": group[col].notna().mean(),
            })

    return pd.DataFrame(overall_rows), pd.DataFrame(by_damage_rows)


def score_block_rows(by_damage_df, block_rows):
    """Summarise the detection rates of one block-row setting by their minimum and mean."""
    sub = by_damage_df[(by_damage_df["block_rows"] == block_rows) &
                        (by_damage_df["mode"].isin(MODES_TO_SCORE))]
    return {
        "block_rows": block_rows,
        "min_detection_rate": sub["detection_rate"].min(),
        "mean_detection_rate": sub["detection_rate"].mean(),
    }


def main():
    """Scan the candidate block-row settings, choose the one with the best detection rates and run the full extraction with it."""
    all_overall = []
    all_by_damage = []

    for s in BLOCK_ROWS_CANDIDATES:
        print(f"\n=== Scanning BLOCK_ROWS={s} across all sessions ===")
        freq_df, damping_df, _ = extract_ssi_features(root_path=DEFAULT_PATH, s=s,
                                                        tolerance=PEAK_TOLERANCE)
        freq_df.to_csv(f"{OUT_DIR}/ssi_modal_frequencies_blockrows{s}.csv", index=False)
        damping_df.to_csv(f"{OUT_DIR}/ssi_modal_damping_blockrows{s}.csv", index=False)

        overall, by_damage = detection_rate_tables(freq_df, s)
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

    print("\n=== Detection rate by mode, all BLOCK_ROWS settings ===")
    print(overall_df.pivot(index="mode", columns="block_rows", values="detection_rate").to_string())
    print(f"Saved -> {overall_csv}")

    print("\n=== Detection rate by mode AND damage state, all BLOCK_ROWS settings ===")
    print(by_damage_df.pivot_table(index=["damage_label", "mode"], columns="block_rows",
                                    values="detection_rate").to_string())
    print(f"Saved -> {by_damage_csv}")

    scores = pd.DataFrame([score_block_rows(by_damage_df, s) for s in BLOCK_ROWS_CANDIDATES])
    scores = scores.sort_values(["min_detection_rate", "mean_detection_rate"], ascending=False)
    print("\n=== BLOCK_ROWS ranking (worst-case detection rate first) ===")
    print(scores.to_string(index=False))
    scores.to_csv(f"{OUT_DIR}/block_rows_scores.csv", index=False)

    if AUTO_SELECT:
        chosen_s = int(scores.iloc[0]["block_rows"])
        print(f"\nAuto-selected BLOCK_ROWS={chosen_s} "
              f"(highest worst-case detection rate across damage groups and modes).")
    else:
        chosen_s = MANUAL_BLOCK_ROWS
        print(f"\nAUTO_SELECT is False, using MANUAL_BLOCK_ROWS={chosen_s} as set above.")

    # Full extraction (frequencies, damping, mode shapes, plots) at the
    # chosen BLOCK_ROWS
    final_dir = f"{OUT_DIR}/blockrows{chosen_s}_final"
    os.makedirs(final_dir, exist_ok=True)

    print(f"\n=== Running full extraction at BLOCK_ROWS={chosen_s} -> {final_dir} ===")
    freq_df, damping_df, shape_df = extract_ssi_features(root_path=DEFAULT_PATH, s=chosen_s,
                                                           tolerance=PEAK_TOLERANCE)
    freq_df.to_csv(f"{final_dir}/ssi_modal_frequencies.csv", index=False)
    damping_df.to_csv(f"{final_dir}/ssi_modal_damping.csv", index=False)
    shape_df.to_csv(f"{final_dir}/ssi_mode_shapes.csv", index=False)

    freq_cols = [c for c in freq_df.columns if c.startswith("freq_")]
    print("\n=== Final detection rate per mode, chosen BLOCK_ROWS ===")
    print(freq_df[freq_cols].notna().mean().to_string())

    plot_average_mode_shapes(shape_df, out_dir=final_dir)

    # one example stabilization diagram, from the first session found, so
    # a visual sanity check is saved alongside the batch results
    sample_path = None
    for dirpath, _, filenames in os.walk(DEFAULT_PATH):
        for fname in filenames:
            if fname.endswith(".mat"):
                sample_path = os.path.join(dirpath, fname)
                break
        if sample_path:
            break
    if sample_path:
        print(f"\nSaving one example stabilization diagram from {sample_path}")
        run_single_session_diagnostics(sample_path, s=chosen_s, out_dir=final_dir)

    print(f"\nDone. All outputs saved under {os.path.abspath(OUT_DIR)}/")


if __name__ == "__main__":
    main()
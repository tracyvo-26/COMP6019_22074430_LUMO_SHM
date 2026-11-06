
"""
Compares each session's SSI-COV mode shape with a reference shape (MAC value) and plots it against temperature, to see whether the shapes stay stable.

Needs: ssi_mode_shapes.csv (from ssi_cov_batch.py).
"""
 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
SHAPE_CSV_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "02_modal_identification", "SSI-COV", "output_batch", "blockrows20_final", "ssi_mode_shapes.csv"))
METHOD_LABEL = "SSI-COV"          # used in output filenames and plot titles, e.g. "FDD" or "SSI-COV"
REFERENCE_GROUP = "Healthy"   # which damage_label to build the reference shape from
REFERENCE_MODE = "average"    # "average" or "single_session"
REFERENCE_SESSION = None      # only used if REFERENCE_MODE == "single_session", e.g. a specific session filename
OUT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "02_modal_identification", "MAC", "SSI-COV", "output_validation"))
 
os.makedirs(OUT_DIR, exist_ok=True)
 
 
def mac(shape1, shape2):
    """Modal Assurance Criterion between two mode shape vectors (0 to 1)."""
    num = np.abs(np.dot(shape1, shape2)) ** 2
    den = np.dot(shape1, shape1) * np.dot(shape2, shape2)
    return num / den if den > 0 else np.nan
 
 
def session_shape_matrix(df, mode_label):
    """Pivot the long-format shape data of one mode into a session by channel matrix."""
    sub = df[df["mode_label"] == mode_label]
    pivot = sub.pivot_table(index="session", columns="channel", values="amplitude")
    meta = sub.groupby("session")[["damage_label", "temperature_degC"]].first()
    return pivot, meta
 
 
def build_reference_shape(pivot, meta, reference_group=REFERENCE_GROUP,
                           mode=REFERENCE_MODE, single_session=REFERENCE_SESSION):
    """Build the reference shape of one mode by sign-aligning and averaging the reference sessions."""
    ref_sessions = meta[meta["damage_label"].str.lower() == reference_group.lower()].index
    ref_sessions = [s for s in ref_sessions if s in pivot.index]
 
    if mode == "single_session":
        if single_session is None or single_session not in ref_sessions:
            raise ValueError(f"REFERENCE_SESSION must be set to one of: {ref_sessions}")
        return pivot.loc[single_session].values
 
    if not ref_sessions:
        return None
    anchor = pivot.loc[ref_sessions[0]].values
    aligned = [anchor]
    for s in ref_sessions[1:]:
        vec = pivot.loc[s].values
        if np.dot(vec, anchor) < 0:
            vec = -vec
        aligned.append(vec)
    return np.mean(aligned, axis=0)
 
 
def main():
    """Compute the MAC of every session against the reference shape of each mode and plot it against temperature."""
    df = pd.read_csv(SHAPE_CSV_PATH)
    modes = df["mode_label"].unique()
 
    all_results = []
    for mode_label in modes:
        pivot, meta = session_shape_matrix(df, mode_label)
        reference = build_reference_shape(pivot, meta)
        if reference is None:
            print(f"skip {mode_label}: no {REFERENCE_GROUP} sessions with this mode detected")
            continue
 
        for session in pivot.index:
            vec = pivot.loc[session].values
            if np.isnan(vec).any():
                continue  # some channels missing for this session, skip rather than guess
            score = mac(vec, reference)
            all_results.append({
                "mode_label": mode_label, "session": session,
                "damage_label": meta.loc[session, "damage_label"],
                "temperature_degC": meta.loc[session, "temperature_degC"],
                "mac": score,
            })
        print(f"{mode_label}: reference built from {REFERENCE_GROUP} "
              f"({REFERENCE_MODE}), MAC computed for {len(pivot)} sessions")
 
    results = pd.DataFrame(all_results)
    out_csv = f"{OUT_DIR}/{METHOD_LABEL.lower()}_mac_vs_reference.csv"
    results.to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv}")
 
    print("\n=== MAC summary by mode ===")
    print(results.groupby("mode_label")["mac"].agg(["min", "mean", "max", "count"]).to_string())
 
    print("\n=== MAC summary by mode and damage state ===")
    print(results.groupby(["mode_label", "damage_label"])["mac"]
          .agg(["min", "mean", "count"]).to_string())
 
    # Plots: MAC vs temperature per mode, same style as Wernitz et al.
    # Figure 18, colour-coded by damage state
    damage_labels = results["damage_label"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(damage_labels)))
    color_map = dict(zip(damage_labels, colors))
 
    for mode_label, group in results.groupby("mode_label"):
        fig, ax = plt.subplots(figsize=(9, 5))
        for damage_label, sub in group.groupby("damage_label"):
            ax.scatter(sub["temperature_degC"], sub["mac"], label=damage_label,
                       color=color_map[damage_label], alpha=0.8, s=40)
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("MAC vs reference shape")
        ax.set_title(f"{METHOD_LABEL}: MAC vs temperature, {mode_label} "
                     f"(reference: {REFERENCE_GROUP} {REFERENCE_MODE})")
        ax.set_ylim(0, 1.05)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fname = f"{OUT_DIR}/{METHOD_LABEL.lower()}_mac_vs_temp_{mode_label}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"saved {fname}")
 
 
if __name__ == "__main__":
    main()
 
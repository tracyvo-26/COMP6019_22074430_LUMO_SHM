"""
Computes the between-channel features (cross-covariance, coherence and transmissibility) for every pair of connected sensors in every session, and writes them to between_channel_features.csv.

Needs: the raw LUMO data folder (LUMO_DATA_DIR).
"""
 
import os
import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import coherence, csd, welch
import matplotlib.pyplot as plt
 
DEFAULT_PATH = os.environ.get("LUMO_DATA_DIR", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data")))          # folder containing the session subfolders
FS = 1651.61                   # LUMO sampling rate (Hz)
OUT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "01_data_and_features", "between_channel and temp explore", "between_channel_features")) 
 
MODE_WINDOWS = [
    (2.5, 3.1, "b1"),     # modes 1-2: first bending pair, 2.76/2.81 Hz
    (13.1, 13.7, "t1"),   # mode 3: first torsional mode, 13.40 Hz
    (15.6, 16.6, "b2"),   # modes 4-5: second bending pair, 15.94/16.28 Hz
]
 
def build_edge_list(n_levels=9, strain_channels=("strain01", "strain02", "strain03")):
    """Build the edge list of the 21-node graph: x chain, y chain, same-level x-y edges and the fan-out to the strain nodes."""
    pairs = []
    # x-direction chain
    for i in range(1, n_levels):
        pairs.append((f"accel{i:02d}x", f"accel{i+1:02d}x"))
    # y-direction chain
    for i in range(1, n_levels):
        pairs.append((f"accel{i:02d}y", f"accel{i+1:02d}y"))
    # same-level x-y edges
    for i in range(1, n_levels + 1):
        pairs.append((f"accel{i:02d}x", f"accel{i:02d}y"))
    # fan-out from ML9 (closest accel level to base) to strain nodes
    for s in strain_channels:
        pairs.append((f"accel{n_levels:02d}x", s))
        pairs.append((f"accel{n_levels:02d}y", s))
    return pairs
 
 
CHANNEL_PAIRS = build_edge_list()

def _decode_string_cell(f, dataset):
    """Decode a cell array of HDF5 string references into a list of Python strings."""
    strings = []
    for ref in dataset[:, 0] if dataset.ndim == 2 else dataset[:]:
        obj = f[ref]
        chars = np.asarray(obj[()]).flatten()
        strings.append("".join(chr(c) for c in chars if c != 0))
    return strings 

def load_session(mat_path):
    """Load one LUMO .mat file, return dict of channel_name -> signal array + metadata."""
    try:
        d = sio.loadmat(mat_path, simplify_cells=True)["Dat"]
        names = list(d["ChannelNames"])
        data = d["Data"]  # (samples, channels)
        signals = {name: data[:, i] for i, name in enumerate(names)}
    except NotImplementedError:
        # v7.3 / HDF5 format, used by the additional archive-sourced sessions
        with h5py.File(mat_path, "r") as f:
            names = _decode_string_cell(f, f["Dat/ChannelNames"])
            data = np.asarray(f["Dat/Data"])  # (channels, samples), opposite orientation
            signals = {name: data[i, :] for i, name in enumerate(names)}
    temp = None
    if "temp01" in signals:
        temp = float(np.mean(signals["temp01"]))
    return signals, temp
 
 
def cross_covariance(x, y):
    """Covariance of two zero-mean channel signals over the session."""
    x = x - np.mean(x)
    y = y - np.mean(y)
    return float(np.mean(x * y))
 
 
def band_coherence(x, y, fs=FS, fmin=2.5, fmax=3.1):
    """Mean coherence between x and y over one fmin-fmax window."""
    f, Cxy = coherence(x, y, fs=fs, nperseg=16384)
    mask = (f >= fmin) & (f <= fmax)
    return float(np.mean(Cxy[mask])) if mask.any() else np.nan
 
 
def band_transmissibility(x, y, fs=FS, fmin=2.5, fmax=3.1):
    """PSD(y)/PSD(x) ratio over one fmin-fmax window -- how response propagates from x to y."""
    f, Pxx = welch(x, fs=fs, nperseg=16384)
    _, Pyy = welch(y, fs=fs, nperseg=16384)
    mask = (f >= fmin) & (f <= fmax)
    if not mask.any() or np.mean(Pxx[mask]) == 0:
        return np.nan
    return float(np.mean(Pyy[mask]) / np.mean(Pxx[mask]))
 
 
def damage_label_from_path(mat_path, root_path):
    """Return the damage label from the folder name of a .mat file, unwrapping an 'additional' subfolder."""
    parent = os.path.basename(os.path.dirname(mat_path))
    if parent.lower() == "additional":
        parent = os.path.basename(os.path.dirname(os.path.dirname(mat_path)))
    parts = parent.split("_", 1)
    if len(parts) > 1 and parts[0].isdigit():
        return parts[1]
    return parent
 
 
def extract_features(root_path=DEFAULT_PATH, pairs=CHANNEL_PAIRS):
    """Compute the between-channel features for every session under root_path and return one row per session."""
    rows = []
    n_mat_found = 0
 
    for dirpath, _, filenames in os.walk(root_path):
        for fname in filenames:
            if not fname.endswith(".mat"):
                continue
            n_mat_found += 1
            mat_path = os.path.join(dirpath, fname)
            damage_label = damage_label_from_path(mat_path, root_path)
 
            try:
                signals, temp = load_session(mat_path)
            except Exception as e:
                print(f"skip {mat_path}: {e}")
                continue
            if temp is None:
                print(f"skip {mat_path}: no temp01 channel found")
                continue
 
            row = {"session": fname, "damage_label": damage_label, "temperature_degC": temp}
            for ch_a, ch_b in pairs:
                if ch_a not in signals or ch_b not in signals:
                    continue
                x, y = signals[ch_a], signals[ch_b]
                pair_name = f"{ch_a}_{ch_b}"
                # cross-covariance is time-domain only, no frequency window applies
                row[f"{pair_name}_xcov"] = cross_covariance(x, y)
                # coherence and transmissibility: one value per mode window
                for fmin, fmax, label in MODE_WINDOWS:
                    row[f"{pair_name}_coh_{label}"] = band_coherence(x, y, fmin=fmin, fmax=fmax)
                    row[f"{pair_name}_trans_{label}"] = band_transmissibility(x, y, fmin=fmin, fmax=fmax)
            rows.append(row)
 
    print(f"Found {n_mat_found} .mat files under {root_path}; "
          f"successfully processed {len(rows)}.")
    if n_mat_found == 0:
        print(f"No .mat files found under {os.path.abspath(root_path)}")
 
    return pd.DataFrame(rows)
 
 
def plot_features_vs_temperature(df, pairs=CHANNEL_PAIRS, outdir="feature_plots"):
    """Plot each pooled feature against temperature, coloured by damage state."""
    if df.empty:
        print("DataFrame is empty, nothing to plot.")
        return
    os.makedirs(outdir, exist_ok=True)
    is_healthy = df["damage_label"].str.lower() == "healthy"
 
    feat_names = ["xcov"] + [f"{feat}_{label}" for feat in ["coh", "trans"] for _, _, label in MODE_WINDOWS]
    for ch_a, ch_b in pairs:
        pair_name = f"{ch_a}_{ch_b}"
        for feat in feat_names:
            col = f"{pair_name}_{feat}"
            if col not in df.columns:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(df.loc[is_healthy, "temperature_degC"], df.loc[is_healthy, col],
                       label="Healthy", alpha=0.7)
            ax.scatter(df.loc[~is_healthy, "temperature_degC"], df.loc[~is_healthy, col],
                       label="Damaged", alpha=0.7, marker="x")
            ax.set_xlabel("Temperature (°C)")
            ax.set_ylabel(f"{feat} ({pair_name})")
            ax.set_title(f"{feat} vs. temperature: {pair_name}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"{col}.png"), dpi=150)
            plt.close(fig)
 
    print(f"Saved plots to ./{outdir}/")
 
 
def _pooled_feature_columns(df, pairs, feat):
    """Every edge column that exists for this feature type, e.g. all *_xcov columns."""
    return [f"{a}_{b}_{feat}" for a, b in pairs if f"{a}_{b}_{feat}" in df.columns]
 
 
def summarize_temperature_sensitivity(df, pairs=CHANNEL_PAIRS):
    """Correlate each edge-pooled feature of the Healthy sessions with temperature."""
    healthy = df[df["damage_label"].str.lower() == "healthy"].copy()
    records = []
    feat_types = ["xcov"] + [f"{feat}_{label}" for feat in ["coh", "trans"] for _, _, label in MODE_WINDOWS]
 
    for feat in feat_types:
        feat_cols = _pooled_feature_columns(healthy, pairs, feat)
        if not feat_cols:
            continue
        pooled = healthy[feat_cols].mean(axis=1)
        valid_mask = pooled.notna() & healthy["temperature_degC"].notna()
        n = int(valid_mask.sum())
        if n < 3:
            records.append({"feature_type": feat, "n": n, "corr_with_temp": np.nan, "n_edges_pooled": len(feat_cols)})
            continue
        r = healthy.loc[valid_mask, "temperature_degC"].corr(pooled[valid_mask])
        records.append({"feature_type": feat, "n": n, "corr_with_temp": r, "n_edges_pooled": len(feat_cols)})
 
    summary = pd.DataFrame(records).sort_values("corr_with_temp", key=abs, ascending=False)
    print("\nCorrelation with temperature by feature type (Healthy sessions only, averaged across edges first):")
    print(summary.to_string(index=False))
    print("\nHigher |corr_with_temp| = more temperature-driven, lower = less confounded by environment.")
    return summary
 
 
TEMP_BINS = [-float("inf"), 0, 10, 20, 30, float("inf")]
TEMP_BIN_LABELS = ["<0", "0-10", "10-20", "20-30", ">30"]
 
 
def summarize_damage_sensitivity(df, pairs=CHANNEL_PAIRS):
    """Compare damaged and Healthy sessions on each edge-pooled feature within matching temperature bins."""
    df = df.copy()
    df["temp_bin"] = pd.cut(df["temperature_degC"], bins=TEMP_BINS, labels=TEMP_BIN_LABELS)
    df["is_healthy"] = df["damage_label"].str.lower() == "healthy"
 
    records = []
    feat_types = ["xcov"] + [f"{feat}_{label}" for feat in ["coh", "trans"] for _, _, label in MODE_WINDOWS]
    for feat in feat_types:
        feat_cols = _pooled_feature_columns(df, pairs, feat)
        if not feat_cols:
            continue
        df[f"pooled_{feat}"] = df[feat_cols].mean(axis=1)
 
        for temp_bin, group in df.groupby("temp_bin", observed=True):
            healthy_vals = group.loc[group["is_healthy"], f"pooled_{feat}"].dropna()
            damaged_vals = group.loc[~group["is_healthy"], f"pooled_{feat}"].dropna()
            # only compare where both groups actually have data in this bin
            if len(healthy_vals) < 2 or len(damaged_vals) < 2:
                continue
            pooled_std = np.std(pd.concat([healthy_vals, damaged_vals]))
            if pooled_std == 0 or np.isnan(pooled_std):
                continue
            effect_size = abs(damaged_vals.mean() - healthy_vals.mean()) / pooled_std
            records.append({
                "feature_type": feat, "temp_bin": str(temp_bin),
                "n_healthy": len(healthy_vals), "n_damaged": len(damaged_vals),
                "effect_size": effect_size,
            })
 
    summary = pd.DataFrame(records)
    if summary.empty:
        print("No temperature bin had both Healthy and Damaged sessions with enough data "
              "to compare -- this itself is worth reporting as a positivity-style gap.")
        return summary, None
 
    ranked = (summary.groupby("feature_type")["effect_size"]
              .agg(["mean", "std", "count"])
              .sort_values("mean", ascending=False))
    print("\nHealthy-vs-Damaged effect size by feature type (pooled across edges, within matching temperature bins):")
    print(ranked)
    print("\nHigher mean = feature separates damage from healthy more strongly, "
          "within the same temperature context.")
    return summary, ranked
 
 
if __name__ == "__main__":
    df = extract_features()
    df.to_csv(f"{OUT_DIR}/between_channel_features.csv", index=False)
    print(f"Extracted features for {len(df)} sessions -> between_channel_features.csv")
    plot_features_vs_temperature(df, outdir=f"{OUT_DIR}/feature_plots")
    summarize_temperature_sensitivity(df)
    summarize_damage_sensitivity(df)
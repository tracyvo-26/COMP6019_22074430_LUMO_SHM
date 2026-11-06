"""
Finds the tower's natural frequencies and mode shapes from the accelerometer data alone, using Frequency Domain Decomposition (FDD). Other FDD and SSI-COV scripts reuse its functions.

Needs: the raw LUMO data folder (LUMO_DATA_DIR).
"""
 
import os
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import csd, find_peaks
import matplotlib.pyplot as plt
import h5py 
 
DEFAULT_PATH = os.environ.get("LUMO_DATA_DIR", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")))
SAMPLE_SESSION = None  # one .mat path for the __main__ check, or None to use the first one found under DEFAULT_PATH
OUT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "02_modal_identification", "FDD", "output_identification"))
FMAX = 20.0        # only search up to 20 Hz, where modes 1-5 show real peaks
PEAK_TOLERANCE = 0.5  # Hz, how close a detected peak must be to a known mode to count as a match
 
FS = 1651.61
NPERSEG = 16384
 
# All 18 accelerometer channels (9 levels x 2 directions). Strain channels
# are left out, mode shapes are conventionally estimated from the
# accelerometer network only.
ACCEL_CHANNELS = [f"accel{i:02d}{d}" for i in range(1, 10) for d in ("x", "y")]
 
# Wernitz et al., Table 6: known mode frequencies, for labelling peaks
KNOWN_MODES = [
    (2.76, "B1-y"), (2.81, "B1-x"), (13.40, "T1"),
    (15.94, "B2-y"), (16.28, "B2-x"), (39.78, "T2"),
    (40.76, "B3-y"), (42.14, "B3-x"), (66.40, "T3"),
    (69.15, "B4-y"), (71.51, "B4-x"), (92.40, "T4"),
    (100.98, "B5-y"), (104.33, "B5-x"), (117.16, "T5"),
]
 
def _decode_string_cell(f, dataset):
    """Decode a cell array of HDF5 string references into a list of Python strings."""
    strings = []
    for ref in dataset[:, 0] if dataset.ndim == 2 else dataset[:]:
        obj = f[ref]
        chars = np.asarray(obj[()]).flatten()
        strings.append("".join(chr(c) for c in chars if c != 0))
    return strings

def load_session(mat_path):
    """Load one session from either the .mat or the v7.3 HDF5 format and return its signals and metadata."""
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
    temp = float(np.mean(signals["temp01"])) if "temp01" in signals else None
    return signals, temp
 
 
def damage_label_from_path(mat_path):
    """Return the damage label from the folder name of a .mat file, unwrapping an 'additional' subfolder."""
    parent = os.path.basename(os.path.dirname(mat_path))
    if parent.lower() == "additional":
        parent = os.path.basename(os.path.dirname(os.path.dirname(mat_path)))
    parts = parent.split("_", 1)
    if len(parts) > 1 and parts[0].isdigit():
        return parts[1]
    return parent
 
 
def build_cpsd_matrix(signals, channels, fs=FS, nperseg=NPERSEG):
    """Return the frequency bins and the cross power spectral density matrix at every frequency."""
    n = len(channels)
    freqs = None
    G = None
 
    for i in range(n):
        for j in range(i, n):  # CPSD matrix is Hermitian, only compute half
            f, Sij = csd(signals[channels[i]], signals[channels[j]], fs=fs, nperseg=nperseg)
            if G is None:
                freqs = f
                G = np.zeros((len(f), n, n), dtype=complex)
            G[:, i, j] = Sij
            if i != j:
                G[:, j, i] = np.conj(Sij)  # Hermitian symmetry
 
    return freqs, G
 
 
def run_fdd(signals, channels=ACCEL_CHANNELS, fmax=20.0, nperseg=NPERSEG):
    """Run FDD on one session and return the frequencies, first singular values and mode shapes."""
    freqs, G = build_cpsd_matrix(signals, channels, nperseg=nperseg)
 
    # keep only up to fmax, since that's where modes 1-5 live and this
    # keeps the SVD loop fast; widen to search higher modes
    mask = freqs <= fmax
    freqs = freqs[mask]
    G = G[mask]
 
    n_freqs = len(freqs)
    s1 = np.zeros(n_freqs)          # first (largest) singular value at each frequency
    mode_shapes = np.zeros((n_freqs, len(channels)), dtype=complex)
 
    for k in range(n_freqs):
        U, S, _ = np.linalg.svd(G[k])
        s1[k] = S[0]
        mode_shapes[k, :] = U[:, 0]  # first singular vector = candidate mode shape
 
    return freqs, s1, mode_shapes
 
 
def find_mode_peaks(freqs, s1, min_prominence_db=3.0, distance_hz=0.5):
    """Return the indices of peaks in the first singular value spectrum."""
    s1_db = 10 * np.log10(s1 / s1.max())
    distance_bins = max(1, int(distance_hz / (freqs[1] - freqs[0])))
    peak_idx, _ = find_peaks(s1_db, prominence=min_prominence_db, distance=distance_bins)
    return peak_idx
 
 
def label_peak(freq, tolerance=0.5):
    """Label a peak frequency with the closest known mode within the tolerance."""
    for known_freq, label in KNOWN_MODES:
        if abs(freq - known_freq) <= tolerance:
            return label
    return "unlabelled"
 
 
def plot_singular_value_spectrum(freqs, s1, peak_idx):
    """Plot the first singular value spectrum with the detected peaks labelled."""
    s1_db = 10 * np.log10(s1 / s1.max())
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freqs, s1_db, linewidth=1.0)
    ax.scatter(freqs[peak_idx], s1_db[peak_idx], color="red", zorder=5)
    for idx in peak_idx:
        label = label_peak(freqs[idx])
        ax.annotate(f"{freqs[idx]:.2f} Hz\n({label})", (freqs[idx], s1_db[idx]),
                    textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("First singular value (dB, normalized)")
    ax.set_title("FDD: first singular value spectrum, detected peaks marked")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fdd_singular_value_spectrum.png", dpi=150)
    print(f"saved {OUT_DIR}/fdd_singular_value_spectrum.png")
 
 
def plot_mode_shape(channels, mode_shape, freq, label):
    """Plot the normalised real part of a mode shape against channel index, for x and y."""
    real_shape = np.real(mode_shape) / np.max(np.abs(np.real(mode_shape)))
    x_vals = [real_shape[i] for i, ch in enumerate(channels) if ch.endswith("x")]
    y_vals = [real_shape[i] for i, ch in enumerate(channels) if ch.endswith("y")]
    levels = list(range(1, len(x_vals) + 1))
 
    fig, ax = plt.subplots(figsize=(4, 6))
    ax.plot(x_vals, levels, marker="o", label="x direction")
    ax.plot(y_vals, levels, marker="s", label="y direction")
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Normalized mode shape amplitude")
    ax.set_ylabel("Level (ML1 to ML9)")
    ax.set_title(f"Mode shape at {freq:.2f} Hz ({label})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fname = f"{OUT_DIR}/fdd_mode_shape_{label}.png"
    fig.savefig(fname, dpi=150)
    print(f"saved {fname}")
 
 
def run_single_session_diagnostics(mat_path, fmax=FMAX):
    """Run FDD on one session and print and plot the detected peaks."""
    signals, _ = load_session(mat_path)
    freqs, s1, mode_shapes = run_fdd(signals, fmax=fmax)
    peak_idx = find_mode_peaks(freqs, s1)
 
    print(f"Detected {len(peak_idx)} peaks:")
    for idx in peak_idx:
        label = label_peak(freqs[idx])
        print(f"  {freqs[idx]:.3f} Hz  ->  {label}")
 
    plot_singular_value_spectrum(freqs, s1, peak_idx)
    for idx in peak_idx:
        label = label_peak(freqs[idx])
        plot_mode_shape(ACCEL_CHANNELS, mode_shapes[idx], freqs[idx], label)
 
 
def assign_peaks_to_modes(candidate_freqs, modes_to_track, tolerance):
    """Match each tracked mode to at most one candidate peak, closest pairs first."""
    pairs = []
    for ci, cf in enumerate(candidate_freqs):
        for target_freq, label in modes_to_track:
            d = abs(cf - target_freq)
            if d <= tolerance:
                pairs.append((d, ci, label))
    pairs.sort(key=lambda x: x[0])
 
    assigned = {}
    used_candidates = set()
    used_labels = set()
    for d, ci, label in pairs:
        if label in used_labels or ci in used_candidates:
            continue
        assigned[label] = ci
        used_candidates.add(ci)
        used_labels.add(label)
 
    for _, label in modes_to_track:
        assigned.setdefault(label, None)
    return assigned
 
 
def normalize_shape(mode_shape, reference_channel_idx=0):
    """Return the real part of a mode shape, sign-fixed to a reference channel and scaled to a maximum of 1."""
    real_shape = np.real(mode_shape)
    if real_shape[reference_channel_idx] < 0:
        real_shape = -real_shape
    peak = np.max(np.abs(real_shape))
    if peak == 0:
        return real_shape
    return real_shape / peak

def scan_sessions(root_path):
    """Return [(path, fname, damage_label)] for every .mat session, skipping macOS metadata files."""
    found = []
    for dirpath, _, filenames in os.walk(root_path):
        for fname in sorted(filenames):
            if fname.endswith(".mat") and not fname.startswith("._"):
                path = os.path.join(dirpath, fname)
                found.append((path, fname, damage_label_from_path(path)))
    return found

def print_read_summary(found, processed, skipped, out_dir=OUT_DIR):
    """Print how many files were found, read and skipped for each damage state."""
    from collections import Counter
    n_found = Counter(label for _, _, label in found)
    n_read = Counter(label for _, fname, label in found if fname in processed)
    print(f"\n{'=' * 60}")
    print(f"{'damage state':<14}{'found':>8}{'read':>8}{'skipped':>10}")
    print("-" * 40)
    for label in sorted(n_found):
        print(f"{label:<14}{n_found[label]:>8}{n_read[label]:>8}{n_found[label] - n_read[label]:>10}")
    print("-" * 40)
    print(f"{'TOTAL':<14}{len(found):>8}{len(processed):>8}{len(found) - len(processed):>10}")
    print(f"{'=' * 60}")
    if skipped:
        os.makedirs(out_dir, exist_ok=True)
        path = f"{out_dir}/skipped_sessions.csv"
        pd.DataFrame(skipped, columns=["session", "reason"]).to_csv(path, index=False)
        print(f"{len(skipped)} sessions skipped, reasons saved to {path}")
        for fname, reason in skipped[:10]:
            print(f"  {fname}: {reason}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

def count_sessions_by_class(found, root_path):
    """Print the number of session files found for each damage state."""
    from collections import Counter
    counts = Counter(label for _, _, label in found)
    n_unique = len({fname for _, fname, _ in found})
    print(f"\n{'=' * 60}")
    print(f"Pre-flight scan: {len(found)} .mat files under {root_path}")
    print(f"Unique file names: {n_unique}"
          + ("" if n_unique == len(found) else f"  (WARNING: {len(found) - n_unique} duplicates)"))
    print(f"{'=' * 60}")
    for label, n in sorted(counts.items()):
        print(f"  {label:<12} {n}")
    print(f"{'=' * 60}\n")
    return counts
 
def extract_fdd_features(root_path=DEFAULT_PATH, fmax=FMAX, tolerance=PEAK_TOLERANCE,
                          channels=ACCEL_CHANNELS, nperseg=NPERSEG):
    """Run FDD on every session and return the frequency table and the mode shape table."""
    found = scan_sessions(root_path)
    count_sessions_by_class(found, root_path)

    freq_rows, shape_rows = [], []
    processed, skipped = set(), []
    modes_to_track = KNOWN_MODES[:5]  # B1-y, B1-x, T1, B2-y, B2-x

    for mat_path, fname, damage_label in found:
        try:
            signals, temp = load_session(mat_path)
        except Exception as e:
            skipped.append((fname, f"load error: {e}"))
            continue
        if temp is None:
            skipped.append((fname, "no temp01 channel"))
            continue
        missing = [ch for ch in channels if ch not in signals]
        if missing:
            skipped.append((fname, f"missing channels: {missing[:3]}"))
            continue

        freqs, s1, mode_shapes = run_fdd(signals, channels=channels, fmax=fmax, nperseg=nperseg)
        peak_idx = find_mode_peaks(freqs, s1)
        peak_freqs = freqs[peak_idx]
        assignment = assign_peaks_to_modes(peak_freqs, modes_to_track, tolerance)

        freq_row = {"session": fname, "damage_label": damage_label, "temperature_degC": temp}
        for known_freq, label in modes_to_track:
            pos = assignment[label]
            if pos is None:
                freq_row[f"freq_{label}"] = np.nan
                continue
            closest = peak_freqs[pos]
            freq_row[f"freq_{label}"] = closest
            shape = normalize_shape(mode_shapes[peak_idx[pos], :])
            for ch, amp in zip(channels, shape):
                shape_rows.append({
                    "session": fname, "damage_label": damage_label,
                    "temperature_degC": temp, "mode_label": label,
                    "detected_freq_Hz": closest, "channel": ch,
                    "direction": ch[-1],
                    "level": int(ch.replace("accel", "").rstrip("xy")),
                    "amplitude": amp,
                })
        freq_rows.append(freq_row)
        processed.add(fname)
        print(f"processed {fname} ({len(processed)}/{len(found)})")

    print_read_summary(found, processed, skipped)
    return pd.DataFrame(freq_rows), pd.DataFrame(shape_rows)
 
 
def plot_average_mode_shapes(shape_df, out_dir=OUT_DIR):
    """Plot the average mode shape of each damage state for every mode."""
    os.makedirs(out_dir, exist_ok=True)
    for label in shape_df["mode_label"].unique():
        sub = shape_df[shape_df["mode_label"] == label]
        fig, axes = plt.subplots(1, 2, figsize=(9, 6), sharey=True)
        for ax, direction in zip(axes, ["x", "y"]):
            dsub = sub[sub["direction"] == direction]
            for damage_label, group in dsub.groupby("damage_label"):
                avg_by_level = group.groupby("level")["amplitude"].mean().sort_index()
                ax.plot(avg_by_level.values, avg_by_level.index, marker="o",
                        label=damage_label, alpha=0.8)
            ax.axvline(0, color="gray", linewidth=0.5)
            ax.set_xlabel(f"{direction} direction amplitude")
            ax.set_title(f"{direction} direction")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("Level (ML1 to ML9)")
        axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        fig.suptitle(f"Average mode shape at {label}, by damage state")
        fig.tight_layout()
        fname = f"{out_dir}/fdd_avg_mode_shape_{label}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"saved {fname}")
 
 
def _find_one_sample_session(root_path):
    """Return the first .mat file found under root_path, or None."""
    for dirpath, _, filenames in os.walk(root_path):
        for fname in filenames:
            if fname.endswith(".mat"):
                return os.path.join(dirpath, fname)
    return None
 
 
if __name__ == "__main__":
    sample_path = SAMPLE_SESSION or _find_one_sample_session(DEFAULT_PATH)
    if sample_path is None:
        print(f"No .mat files found under {os.path.abspath(DEFAULT_PATH)}, "
              f"nothing to check.")
    else:
        print(f"Running a single-session check on {sample_path}\n"
              f"(for the full batch, use fdd_optimized_nperseg.py instead)\n")
        run_single_session_diagnostics(sample_path)
 
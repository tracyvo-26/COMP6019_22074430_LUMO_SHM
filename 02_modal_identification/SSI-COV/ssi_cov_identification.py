"""
Finds the natural frequencies and mode shapes with covariance-driven Stochastic Subspace Identification (SSI-COV), as a cross-check of FDD.

Needs: fdd_modal_identification.py and the raw LUMO data folder.
"""
 
import os
import numpy as np
import pandas as pd
from scipy.signal import decimate
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "02_modal_identification", "FDD")))

from fdd_modal_identification import (
    FS, ACCEL_CHANNELS, KNOWN_MODES, load_session, damage_label_from_path,
    normalize_shape, assign_peaks_to_modes,
)
 
DEFAULT_PATH = os.environ.get("LUMO_DATA_DIR", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")))
SAMPLE_SESSION = None       # one .mat path for the __main__ check, or None to use the first one found
OUT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "02_modal_identification", "SSI-COV", "output_identification"))
TARGET_FS = 50.0            # decimate down to this rate, well above Nyquist for the <20 Hz modes of interest
FMAX = 20.0                 # only keep poles up to this frequency, matches FDD's scope
BLOCK_ROWS = 40             # "s" in the Toeplitz matrix, this project's equivalent of FDD's nperseg
ORDER_MIN, ORDER_MAX, ORDER_STEP = 2, 60, 2   # model orders to scan for the stabilization diagram
FREQ_TOL = 0.01             # 1%, how much a pole's frequency may drift between orders and still count as stable
DAMPING_TOL = 0.05          # 5%, same idea for damping ratio
MAC_TOL = 0.98              # mode shape similarity threshold (0-1) for two poles to count as the same
MIN_CHAIN_LENGTH = 3        # a pole must stay stable across this many consecutive orders to be trusted
MAX_DAMPING = 0.15          # discard poles with unrealistically high damping, almost always numerical artifacts
PEAK_TOLERANCE = 0.5        # Hz, how close a stabilized pole must be to a known mode to count as a match
 
 
def decimate_signals(signals, channels, fs=FS, target_fs=TARGET_FS):
    """Decimate every channel to close to target_fs in steps of at most 10."""
    q_total = fs / target_fs
    stages = []
    remaining = q_total
    while remaining > 1.001:
        q = min(10, remaining)
        stages.append(int(round(q)))
        remaining /= q
    decimated = {}
    for ch in channels:
        sig = signals[ch]
        for q in stages:
            sig = decimate(sig, q, zero_phase=True)
        decimated[ch] = sig
    new_fs = fs / np.prod(stages) if stages else fs
    return decimated, new_fs
 
 
def compute_correlation_matrices(signals, channels, max_lag):
    """Biased correlation matrices R[k] = mean(y(t+k) y(t)^T) for lags 0 to max_lag."""
    n = len(channels)
    y = np.stack([signals[ch] - np.mean(signals[ch]) for ch in channels])  # n x N, zero-meaned
    N = y.shape[1]
    R = np.zeros((max_lag + 1, n, n))
    for k in range(max_lag + 1):
        # R[k, i, j] = mean over t of y_i(t+k) * y_j(t)
        R[k] = (y[:, k:] @ y[:, :N - k].T) / N
    return R
 
 
def build_toeplitz(R, s, n_channels):
    """Block Toeplitz matrix of the correlation matrices R[1] to R[2s-1]."""
    T = np.zeros((s * n_channels, s * n_channels))
    for a in range(s):
        for b in range(s):
            lag = s + a - b
            T[a * n_channels:(a + 1) * n_channels, b * n_channels:(b + 1) * n_channels] = R[lag]
    return T
 
 
def svd_toeplitz(T):
    """SVD of the block Toeplitz matrix, computed once per session and block-row setting."""
    U, S, _ = np.linalg.svd(T)
    return U, S
 
 
def mac(shape1, shape2):
    """Modal Assurance Criterion between two mode shape vectors (0 to 1)."""
    num = np.abs(np.vdot(shape1, shape2)) ** 2
    den = np.vdot(shape1, shape1).real * np.vdot(shape2, shape2).real
    return num / den if den > 0 else 0.0
 
 
def poles_at_order(U, S, order, n_channels, s, fs, fmax=FMAX, max_damping=MAX_DAMPING):
    """Return the physically plausible poles of the state-space model at one model order."""
    O = U[:, :order] * np.sqrt(S[:order])          # observability matrix
    O_up = O[:-n_channels, :]
    O_down = O[n_channels:, :]
    A = np.linalg.pinv(O_up) @ O_down
    C = O[:n_channels, :]
 
    eigvals, eigvecs = np.linalg.eig(A)
    dt = 1.0 / fs
 
    poles = []
    seen_freqs = []
    for k in range(len(eigvals)):
        mu = eigvals[k]
        if abs(mu) <= 0 or abs(mu) >= 1:
            continue  # unstable or non-oscillatory pole, discard
        lam = np.log(mu) / dt
        freq = abs(lam.imag) / (2 * np.pi)
        if freq <= 1e-6 or freq > fmax:
            continue
        damping = -lam.real / abs(lam)
        if not (0 < damping < max_damping):
            continue
        # complex conjugate pairs give the same freq/damping twice, keep one
        if any(abs(freq - f) < 1e-6 for f in seen_freqs):
            continue
        seen_freqs.append(freq)
        shape = C @ eigvecs[:, k]
        poles.append({"freq": freq, "damping": damping, "shape": shape})
    return poles
 
 
def run_stabilization(signals, channels=ACCEL_CHANNELS, fs=FS, target_fs=TARGET_FS,
                       s=BLOCK_ROWS, order_min=ORDER_MIN, order_max=ORDER_MAX,
                       order_step=ORDER_STEP, freq_tol=FREQ_TOL, damping_tol=DAMPING_TOL,
                       mac_tol=MAC_TOL, min_chain_length=MIN_CHAIN_LENGTH, fmax=FMAX):
    """Return the poles that stay stable over at least min_chain_length consecutive model orders."""
    decimated, dec_fs = decimate_signals(signals, channels, fs=fs, target_fs=target_fs)
    max_lag = 2 * s - 1
    R = compute_correlation_matrices(decimated, channels, max_lag)
    T = build_toeplitz(R, s, len(channels))
    U, S = svd_toeplitz(T)
 
    orders = list(range(order_min, min(order_max, T.shape[0]) + 1, order_step))
    poles_by_order = {o: poles_at_order(U, S, o, len(channels), s, dec_fs, fmax=fmax) for o in orders}
 
    # chain-building: for each pole at the highest order, walk backwards
    # through decreasing orders looking for a matching pole at each step
    chains = []
    for pole in poles_by_order[orders[-1]]:
        chain = [pole]
        for o in reversed(orders[:-1]):
            match = None
            for cand in poles_by_order[o]:
                freq_ok = abs(cand["freq"] - chain[-1]["freq"]) / chain[-1]["freq"] < freq_tol
                damp_ok = abs(cand["damping"] - chain[-1]["damping"]) / chain[-1]["damping"] < damping_tol
                mac_ok = mac(cand["shape"], chain[-1]["shape"]) > mac_tol
                if freq_ok and damp_ok and mac_ok:
                    match = cand
                    break
            if match is None:
                break
            chain.append(match)
        chains.append({"freq": pole["freq"], "damping": pole["damping"], "shape": pole["shape"],
                        "chain_length": len(chain)})
 
    stabilized = [c for c in chains if c["chain_length"] >= min_chain_length]
    return stabilized, poles_by_order, orders
 
 
def label_pole(freq, tolerance=PEAK_TOLERANCE):
    """Label a pole frequency with the closest known mode within the tolerance."""
    for known_freq, label in KNOWN_MODES:
        if abs(freq - known_freq) <= tolerance:
            return label
    return "unlabelled"
 
 
def plot_stabilization_diagram(poles_by_order, orders, stabilized, out_dir=OUT_DIR):
    """Plot the stabilisation diagram with the stabilised poles marked."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for o in orders:
        freqs = [p["freq"] for p in poles_by_order[o]]
        ax.scatter(freqs, [o] * len(freqs), color="lightgray", s=15)
    for c in stabilized:
        ax.scatter(c["freq"], orders[-1], color="red", s=40, zorder=5)
        label = label_pole(c["freq"])
        ax.annotate(f"{c['freq']:.2f} Hz\n({label})", (c["freq"], orders[-1]),
                    textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Model order")
    ax.set_title("SSI-COV stabilization diagram (red = stabilized poles)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fname = f"{out_dir}/ssi_stabilization_diagram.png"
    fig.savefig(fname, dpi=150)
    print(f"saved {fname}")
 
 
def run_single_session_diagnostics(mat_path, s=BLOCK_ROWS, out_dir=OUT_DIR):
    """Run SSI-COV on one session and plot its stabilisation diagram."""
    signals, _ = load_session(mat_path)
    stabilized, poles_by_order, orders = run_stabilization(signals, s=s)
 
    print(f"Found {len(stabilized)} stabilized poles:")
    for c in sorted(stabilized, key=lambda c: c["freq"]):
        label = label_pole(c["freq"])
        print(f"  {c['freq']:.3f} Hz, damping {c['damping']:.4f}, "
              f"stable across {c['chain_length']} orders -> {label}")
 
    plot_stabilization_diagram(poles_by_order, orders, stabilized, out_dir=out_dir)

def count_sessions_by_class(root_path):
    """Print the number of .mat files found for each damage state."""
    from collections import Counter
    counts = Counter()
    total = 0
    for dirpath, _, filenames in os.walk(root_path):
        for fname in filenames:
            if fname.endswith(".mat"):
                total += 1
                label = damage_label_from_path(os.path.join(dirpath, fname))
                counts[label] += 1

    print(f"\n{'=' * 60}")
    print(f"Pre-flight scan: {total} .mat files found under {root_path}")
    print(f"{'=' * 60}")
    for label, n in sorted(counts.items()):
        print(f"  {label:<12} {n}")
    print(f"{'=' * 60}\n")
    return total, counts
 
def extract_ssi_features(root_path=DEFAULT_PATH, s=BLOCK_ROWS, tolerance=PEAK_TOLERANCE,
                          channels=ACCEL_CHANNELS, fs=FS, target_fs=TARGET_FS):
    """Run SSI-COV on every session and return the frequency, shape and damping tables for modes 1 to 5."""
    count_sessions_by_class(root_path)
    freq_rows = []
    damping_rows = []
    shape_rows = []
    n_mat_found = 0
    modes_to_track = KNOWN_MODES[:5]
 
    for dirpath, _, filenames in os.walk(root_path):
        for fname in filenames:
            if not fname.endswith(".mat"):
                continue
            n_mat_found += 1
            mat_path = os.path.join(dirpath, fname)
            damage_label = damage_label_from_path(mat_path)
 
            try:
                signals, temp = load_session(mat_path)
            except Exception as e:
                print(f"skip {mat_path}: {e}")
                continue
            if temp is None:
                print(f"skip {mat_path}: no temp01 channel found")
                continue
 
            stabilized, _, _ = run_stabilization(signals, channels=channels, fs=fs,
                                                  target_fs=target_fs, s=s)
            stab_freqs = np.array([c["freq"] for c in stabilized])
 
            assignment = assign_peaks_to_modes(stab_freqs, modes_to_track, tolerance)
 
            freq_row = {"session": fname, "damage_label": damage_label, "temperature_degC": temp}
            damping_row = {"session": fname, "damage_label": damage_label, "temperature_degC": temp}
            for known_freq, label in modes_to_track:
                pos = assignment[label]
                if pos is None:
                    freq_row[f"freq_{label}"] = np.nan
                    damping_row[f"damping_{label}"] = np.nan
                    continue
                closest = stab_freqs[pos]
                freq_row[f"freq_{label}"] = closest
                damping_row[f"damping_{label}"] = stabilized[pos]["damping"]
 
                shape = normalize_shape(stabilized[pos]["shape"])
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
            damping_rows.append(damping_row)
            print(f"processed {fname} ({len(freq_rows)} so far)")
 
    print(f"\nFound {n_mat_found} .mat files under {root_path}; successfully processed {len(freq_rows)}.")
    return pd.DataFrame(freq_rows), pd.DataFrame(damping_rows), pd.DataFrame(shape_rows)
 
 
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
        fig.suptitle(f"Average SSI-COV mode shape at {label}, by damage state")
        fig.tight_layout()
        fname = f"{out_dir}/ssi_avg_mode_shape_{label}.png"
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
              f"(for the full batch, use ssi_cov_batch.py instead)\n")
        run_single_session_diagnostics(sample_path)
 
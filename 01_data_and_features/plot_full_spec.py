"""
Plots the coherence and power spectrum of one session over the full frequency range, with the known mode frequencies marked.

Needs: the raw LUMO data folder (LUMO_DATA_DIR).
"""
 
import numpy as np
import scipy.io as sio
from scipy.signal import coherence, welch
import matplotlib.pyplot as plt
 
MAT_PATH = "data/exemplary_datasets_dam3_010/12_DAM3_010/SHMTS_202106260005.mat"
OUT_DIR = "."
 
FS = 1651.61
 
# Wernitz et al., Table 6: 15 mode frequencies (average natural frequency column)
MODES = [
    (2.76, "B1-y"), (2.81, "B1-x"), (13.40, "T1"),
    (15.94, "B2-y"), (16.28, "B2-x"), (39.78, "T2"),
    (40.76, "B3-y"), (42.14, "B3-x"), (66.40, "T3"),
    (69.15, "B4-y"), (71.51, "B4-x"), (92.40, "T4"),
    (100.98, "B5-y"), (104.33, "B5-x"), (117.16, "T5"),
]
 
CHAIN_EDGE = ("accel05x", "accel06x")
XY_EDGE = ("accel05x", "accel05y")
 
 
def load_session(mat_path):
    """Load one session and return its signals by channel name."""
    d = sio.loadmat(mat_path, simplify_cells=True)["Dat"]
    names = list(d["ChannelNames"])
    data = d["Data"]
    return {name: data[:, i] for i, name in enumerate(names)}
 
 
def mark_modes(ax, fmax):
    """Draw the known mode frequencies as vertical lines on an axis."""
    for freq, label in MODES:
        if freq > fmax:
            continue
        ax.axvline(freq, color="red", alpha=0.3, linestyle="--", linewidth=1)
    # annotate a few so the plot isn't cluttered
    ymin, ymax = ax.get_ylim()
    for freq, label in MODES:
        if freq > fmax:
            continue
        ax.text(freq, ymax, label, rotation=90, fontsize=7, color="red",
                 va="top", ha="right")
 
 
def plot_coherence_full_range(signals):
    """Plot coherence over the full frequency range for a chain edge and a same-level x-y edge."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
 
    for ax, (ch_a, ch_b), title in [
        (axes[0], CHAIN_EDGE, f"Chain edge ({CHAIN_EDGE[0]}-{CHAIN_EDGE[1]}), expect strong peaks"),
        (axes[1], XY_EDGE, f"Same-level x-y edge ({XY_EDGE[0]}-{XY_EDGE[1]}), expect mostly flat/near-zero"),
    ]:
        x, y = signals[ch_a], signals[ch_b]
        f, Cxy = coherence(x, y, fs=FS, nperseg=4096)
        ax.plot(f, Cxy, linewidth=0.8, color="tab:blue")
        ax.set_title(title)
        ax.set_ylabel("Coherence")
        ax.set_ylim(0, 1.05)
        mark_modes(ax, f.max())
        ax.grid(alpha=0.2)
 
    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle("Full-range coherence, 0 to Nyquist, with the 15 known mode frequencies marked (red dashed)")
    fig.tight_layout()
    fname = f"{OUT_DIR}/full_range_coherence.png"
    fig.savefig(fname, dpi=150)
    print(f"saved {fname}")
 
 
def plot_psd_full_range(signals, channel="accel05x"):
    """Plot the power spectral density of one channel over the full frequency range."""
    x = signals[channel]
    f, Pxx = welch(x, fs=FS, nperseg=4096)
 
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.semilogy(f, Pxx, linewidth=0.8, color="tab:green")
    ax.set_title(f"Power spectral density, {channel}, full range, log scale")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (log scale)")
    mark_modes(ax, f.max())
    ax.grid(alpha=0.2, which="both")
    fig.tight_layout()
    fname = f"{OUT_DIR}/full_range_psd_{channel}.png"
    fig.savefig(fname, dpi=150)
    print(f"saved {fname}")
 
 
def plot_coherence_zoomed(signals, fmax=20):
    """Plot coherence for a chain edge and a same-level x-y edge up to fmax."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
 
    for ax, (ch_a, ch_b), title in [
        (axes[0], CHAIN_EDGE, f"Chain edge ({CHAIN_EDGE[0]}-{CHAIN_EDGE[1]})"),
        (axes[1], XY_EDGE, f"Same-level x-y edge ({XY_EDGE[0]}-{XY_EDGE[1]})"),
    ]:
        x, y = signals[ch_a], signals[ch_b]
        f, Cxy = coherence(x, y, fs=FS, nperseg=4096)
        mask = f <= fmax
        ax.plot(f[mask], Cxy[mask], linewidth=1.2, color="tab:blue", marker="o", markersize=2)
        ax.set_title(title)
        ax.set_ylabel("Coherence")
        ax.set_ylim(0, 1.05)
        mark_modes(ax, fmax)
        ax.grid(alpha=0.2)
 
    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle(f"Zoomed coherence, 0 to {fmax} Hz (modes 1-5 live here)")
    fig.tight_layout()
    fname = f"{OUT_DIR}/zoomed_coherence_0_{fmax}hz.png"
    fig.savefig(fname, dpi=150)
    print(f"saved {fname}")
 
 
if __name__ == "__main__":
    signals = load_session(MAT_PATH)
    plot_coherence_full_range(signals)
    plot_psd_full_range(signals)
    plot_coherence_zoomed(signals, fmax=20) 
"""
Tests whether FDD can separate the two close B1 modes (B1-x and B1-y) or merges them into one peak.

Needs: fdd_modal_identification.py and the raw LUMO data folder.
"""
 
import os
import numpy as np
import pandas as pd
 
from fdd_modal_identification import (
    DEFAULT_PATH, FMAX, ACCEL_CHANNELS, KNOWN_MODES,
    load_session, damage_label_from_path, run_fdd,
)
from scipy.signal import find_peaks
 
NPERSEG_FOR_TEST = 65536   
DISTANCE_TOLERANCE_CANDIDATES = [
    (0.5, 0.5),     # current default, included as a baseline for comparison
    (0.1, 0.1),
    (0.05, 0.05),
    (0.03, 0.03),
    (0.02, 0.02),
]
OUT_DIR = "b1_separation_test"
 
os.makedirs(OUT_DIR, exist_ok=True)
 
 
def find_peaks_with_params(freqs, s1, distance_hz, min_prominence_db=3.0):
    """Return the indices of spectrum peaks for a given minimum distance and prominence."""
    s1_db = 10 * np.log10(s1 / s1.max())
    distance_bins = max(1, int(distance_hz / (freqs[1] - freqs[0])))
    peak_idx, _ = find_peaks(s1_db, prominence=min_prominence_db, distance=distance_bins)
    return peak_idx
 
 
def label_with_tolerance(freq, tolerance):
    """Return all known modes within the given tolerance of a frequency."""
    return [label for known_freq, label in KNOWN_MODES if abs(freq - known_freq) <= tolerance]
 
 
def main():
    # Step 1 (expensive, done once): compute freqs/s1 for every session
    # at one fixed nperseg
    """Count, for several distance and tolerance settings, how often FDD resolves B1-x and B1-y as two separate peaks."""
    cached = []  # list of (session, damage_label, freqs, s1)
    n_mat_found = 0
    for dirpath, _, filenames in os.walk(DEFAULT_PATH):
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
                continue
            freqs, s1, _ = run_fdd(signals, channels=ACCEL_CHANNELS, fmax=FMAX, nperseg=NPERSEG_FOR_TEST)
            cached.append((fname, damage_label, freqs, s1))
            print(f"computed spectrum for {fname} ({len(cached)} so far)")
 
    print(f"\nFound {n_mat_found} .mat files; computed spectra for {len(cached)}. "
          f"This part is the expensive one, everything below is cheap.\n")
 
    # Step 2 (cheap): for each (distance_hz, tolerance) pair, check how
    # often B1-y and B1-x come out as two DISTINCT detected peaks
    results = []
    for distance_hz, tolerance in DISTANCE_TOLERANCE_CANDIDATES:
        n_separated = 0
        n_sessions = len(cached)
        for fname, damage_label, freqs, s1 in cached:
            peak_idx = find_peaks_with_params(freqs, s1, distance_hz)
            peak_freqs = freqs[peak_idx]
 
            # does some peak match ONLY B1-y (not also matching B1-x), and
            # a DIFFERENT peak match ONLY B1-x? that's genuine separation
            b1y_matches = [f for f in peak_freqs if "B1-y" in label_with_tolerance(f, tolerance)
                           and "B1-x" not in label_with_tolerance(f, tolerance)]
            b1x_matches = [f for f in peak_freqs if "B1-x" in label_with_tolerance(f, tolerance)
                           and "B1-y" not in label_with_tolerance(f, tolerance)]
            if b1y_matches and b1x_matches:
                n_separated += 1
 
        rate = n_separated / n_sessions if n_sessions else float("nan")
        print(f"distance_hz={distance_hz}, tolerance={tolerance}: "
              f"B1 separated in {n_separated}/{n_sessions} sessions ({rate:.1%})")
        results.append({"distance_hz": distance_hz, "tolerance": tolerance,
                         "n_separated": n_separated, "n_sessions": n_sessions,
                         "separation_rate": rate})
 
    results_df = pd.DataFrame(results)
    out_csv = f"{OUT_DIR}/b1_separation_by_params.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv}")
    print("\nIf separation_rate stays 0 even at the tightest settings, that confirms "
          "it's a genuine data/SNR limitation, not a parameter problem. If it rises "
          "above 0 as distance_hz/tolerance shrink, B1 is separable and the current "
          "defaults (0.5/0.5) were simply hiding it.")
 
 
if __name__ == "__main__":
    main()
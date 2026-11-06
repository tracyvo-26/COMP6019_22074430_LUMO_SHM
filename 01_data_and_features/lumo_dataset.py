"""
Reads the raw LUMO recordings and writes one row of summary features per recording session (lumo_features.csv).

Needs: the raw LUMO data folder (LUMO_DATA_DIR).
"""
from __future__ import annotations
import os
import re
import numpy as np
import pandas as pd
import scipy.io as sio
from datetime import datetime, timedelta
 
ACCEL_CHANNELS = [f"accel{i:02d}{axis}" for i in range(1, 10) for axis in ("x", "y")]
STRAIN_CHANNELS = ["strain01", "strain02", "strain03"]
TEMP_CHANNEL = "temp01"
_FILENAME_RE = re.compile(r"SHMTS_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})")


def _time_from_filename(mat_path: str) -> str:
    """Estimate the recording start from the time stamp in the file name."""
    fname = os.path.basename(mat_path)
    m = _FILENAME_RE.search(fname)
    if not m:
        raise ValueError(f"filename does not match expected SHMTS_YYYYMMDDHHMM pattern: {fname}")
    year, month, day, hour, minute = (int(g) for g in m.groups())
    file_ts = datetime(year, month, day, hour, minute)
    estimated_start = file_ts - timedelta(minutes=10)
    return estimated_start.strftime("%Y-%m-%d %H:%M:%S.000")
 
 
def _dominant_frequency(x, fs):
    """Find the strongest non-zero frequency in a signal via FFT."""
    x = x - np.mean(x)  # remove DC offset before FFT
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mags = np.abs(np.fft.rfft(x))
    mags[0] = 0.0  # ignore the DC bin itself
    peak_idx = int(np.argmax(mags))
    return float(freqs[peak_idx])
 
 
def extract_vibration_features(x, fs, prefix):
    """RMS, peak absolute value and dominant frequency of one channel's raw time series."""
    x = np.asarray(x, dtype=np.float64)
    return {
        f"{prefix}_rms": float(np.sqrt(np.mean(x ** 2))),
        f"{prefix}_peak": float(np.max(np.abs(x))),
        f"{prefix}_freq_hz": _dominant_frequency(x, fs),
    }
 
 
def parse_damage_label(mat_path: str) -> str:
    """Return the damage label from the folder name of a .mat file, unwrapping an 'additional' subfolder."""
    parent = os.path.basename(os.path.dirname(mat_path))
    if parent.lower() == "additional":
        parent = os.path.basename(os.path.dirname(os.path.dirname(mat_path)))
    m = re.match(r"^\d+_(.+)$", parent)
    return m.group(1) if m else parent

def _decode_string_cell(f, ref):
    """Decode an HDF5 reference to a MATLAB string into a Python string."""
    arr = np.array(f[ref]).flatten()
    return "".join(chr(c) for c in arr)
 
 
def load_session(mat_path: str) -> dict:
    """Load a .mat recording (original or v7.3/HDF5 format) into a flat dict."""
    try:
        m = sio.loadmat(mat_path, simplify_cells=True)
        d = m["Dat"]
        names = list(d["ChannelNames"])
        data = d["Data"]
        fs = float(d["Fs"])
        recording_start = d["Time"]
    except NotImplementedError:
        import h5py
        with h5py.File(mat_path, "r") as f:
            grp = f["Dat"]
            fs = float(np.array(grp["Fs"]).flatten()[0])

            # Time is stored as a direct array of uint16 character codes
            time_codes = np.array(grp["Time"]).flatten()
            try:
                candidate = "".join(chr(c) for c in time_codes)
                datetime.strptime(candidate[:19], "%Y-%m-%d %H:%M:%S")  # sanity check
                recording_start = candidate
                time_source = "dat_time"
            except (ValueError, OverflowError):
                recording_start = _time_from_filename(mat_path)
                time_source = "filename_estimate"

            name_refs = np.array(grp["ChannelNames"]).flatten()
            names = [_decode_string_cell(f, ref) for ref in name_refs]

            # v7.3 stores Data as (channels, samples); the original format is
            # (samples, channels), so transpose to match load_dataset's
            # column-indexing convention used below.
            data = np.array(grp["Data"]).T

    col_of = {name: i for i, name in enumerate(names)}

    row = {
        "source_file": os.path.basename(mat_path),
        "damage_label": parse_damage_label(mat_path),
        "recording_start": recording_start,
        "fs_hz": fs,
        "n_samples": data.shape[0],
    }

    if TEMP_CHANNEL in col_of:
        row["temperature_degC"] = float(np.mean(data[:, col_of[TEMP_CHANNEL]]))

    for ch in ACCEL_CHANNELS:
        if ch in col_of:
            row.update(extract_vibration_features(data[:, col_of[ch]], fs, ch))

    for ch in STRAIN_CHANNELS:
        if ch in col_of:
            row.update(extract_vibration_features(data[:, col_of[ch]], fs, ch))

    return row
 
 
def load_dataset(root_dir: str) -> pd.DataFrame:
    """Load every .mat file under root_dir into one DataFrame with one row per session."""
    rows = []
    for dirpath, _dirs, files in os.walk(root_dir):
        for fn in sorted(files):
            if fn.lower().endswith(".mat"):
                full = os.path.join(dirpath, fn)
                try:
                    rows.append(load_session(full))
                except Exception as e:
                    print(f"  skipped {full}: {e}")
    if not rows:
        raise ValueError(f"No .mat files found under {root_dir}")
    return pd.DataFrame(rows)
 
DEFAULT_PATH = os.environ.get("LUMO_DATA_DIR", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")))
OUT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "01_data_and_features"))
DEFAULT_OUT = f"{OUT_DIR}/lumo_features.csv"

DAMAGE_TIMELINE = [
    ("2020-08-01", "2020-10-13", "Healthy"),
    ("2020-10-13", "2020-10-27", "DAM6_111"),
    ("2020-10-27", "2020-11-09", "Healthy"),
    ("2020-11-09", "2020-11-24", "DAM4_111"),
    ("2020-11-24", "2021-03-18", "Healthy"),
    ("2021-03-18", "2021-04-20", "DAM3_111"),
    ("2021-04-20", "2021-05-04", "Healthy"),
    ("2021-05-04", "2021-05-19", "DAM6_010"),
    ("2021-05-19", "2021-05-28", "Healthy"),
    ("2021-05-28", "2021-06-14", "DAM4_010"),
    ("2021-06-14", "2021-06-25", "Healthy"),
    ("2021-06-25", "2021-07-12", "DAM3_010"),
    ("2021-07-12", "2021-07-31", "Healthy"),
]
_DAMAGE_TIMELINE_PARSED = [
    (datetime.strptime(s, "%Y-%m-%d"), datetime.strptime(e, "%Y-%m-%d"), label)
    for s, e, label in DAMAGE_TIMELINE
]


def damage_label_from_date(recording_start_str):
    """Return the damage label for a recording start time from Table 3 of Wernitz et al. 2022, or None outside the documented window."""
    ts = datetime.strptime(recording_start_str[:19], "%Y-%m-%d %H:%M:%S")
    for start, end, label in _DAMAGE_TIMELINE_PARSED:
        if start <= ts < end:
            return label
    return None  # outside documented range -- don't guess
 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=DEFAULT_PATH, required=(DEFAULT_PATH is None),
                        help="folder containing Healthy/ and DAMx_yyy/ subfolders")
    parser.add_argument("--out", default=DEFAULT_OUT, help="save the session table to this .csv path")
    args = parser.parse_args()

    if not args.path:
        parser.error("--path is required (or set DEFAULT_PATH at the top of this file)")

    df = load_dataset(args.path)
    print(df.shape)
    print(df.head())
    print("\nColumns:", list(df.columns))
    print(f"\nSessions loaded: {len(df)}" )
    print("\nDamage label counts:")
    print(df["damage_label"].value_counts())

    bins = [-float("inf"), 0, 10, 20, 30, float("inf")]
    labels = ["<0 °C", "0-10 °C", "10-20 °C", "20-30 °C", ">30 °C"]
    # right=False: 0 <= T < 10 goes in "0-10", T < 0 in "<0", T >= 30 in ">30"
    df["temp_bin"] = pd.cut(df["temperature_degC"], bins=bins, labels=labels, right=False)
    coverage = pd.crosstab(df["damage_label"], df["temp_bin"], margins=True, margins_name="Total")
    print("\nSessions per damage state and temperature range:")
    print(coverage.to_string())

    # temperature range and recording window per class
    t = pd.to_datetime(df["recording_start"].str[:19], errors="coerce")
    summary = (df.assign(recording_start_dt=t)
                 .groupby("damage_label")
                 .agg(sessions=("source_file", "count"),
                      first_recording=("recording_start_dt", "min"),
                      last_recording=("recording_start_dt", "max"),
                      temp_min=("temperature_degC", "min"),
                      temp_max=("temperature_degC", "max"))
                 .round(1))
    print("\nPer-class summary:")
    print(summary.to_string())

    if args.out:
        base = os.path.splitext(args.out)[0]
        df.drop(columns=["temp_bin"]).to_csv(args.out, index=False)
        coverage.to_csv(f"{base}_temperature_coverage.csv")
        summary.to_csv(f"{base}_class_summary.csv")
        print(f"\nSaved:\n  {args.out}\n  {base}_temperature_coverage.csv\n  {base}_class_summary.csv")
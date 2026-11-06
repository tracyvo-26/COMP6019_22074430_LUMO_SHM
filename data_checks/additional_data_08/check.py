"""
Scans a folder of raw LUMO files, gives each session its damage state, and flags sessions in temperature ranges that are still under-covered.

Needs: the raw LUMO data folder (LUMO_DATA_DIR).
"""

import os
import re
import glob
import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
from datetime import datetime, timedelta

FOLDER = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data_checks", "additional_data_08"))
OUT_CSV = os.path.join(FOLDER, "2020_08_temperature_scan.csv")

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

# sessions per damage state and temperature bin in the current data
# order: [<0, 0-10, 10-20, 20-30, >30]
COVERAGE_TABLE = {
    "Healthy":  [10, 14, 13, 2, 1],
    "DAM3_010": [0, 1, 3, 5, 4],
    "DAM3_111": [4, 4, 5, 4, 1],
    "DAM4_010": [0, 3, 3, 7, 6],
    "DAM4_111": [1, 4, 3, 1, 0],
    "DAM6_010": [2, 4, 2, 0, 1],
    "DAM6_111": [0, 6, 2, 1, 0],
}
BIN_LABELS = ["<0", "0-10", "10-20", "20-30", ">30"]

_FILENAME_RE = re.compile(r"SHMTS_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})")


def _time_from_filename(mat_path):
    """Estimate the recording start from the time stamp in the file name."""
    fname = os.path.basename(mat_path)
    m = _FILENAME_RE.search(fname)
    if not m:
        raise ValueError(f"filename does not match expected SHMTS_YYYYMMDDHHMM pattern: {fname}")
    year, month, day, hour, minute = (int(g) for g in m.groups())
    file_ts = datetime(year, month, day, hour, minute)
    return (file_ts - timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S.000")


def _decode_string_cell(f, ref):
    """Decode a cell array of HDF5 string references into a list of Python strings."""
    arr = np.array(f[ref]).flatten()
    return "".join(chr(c) for c in arr)


def damage_label_from_date(recording_start_str):
    """Return the damage state for a recording start time from the damage timeline."""
    ts = datetime.strptime(recording_start_str[:19], "%Y-%m-%d %H:%M:%S")
    for start, end, label in _DAMAGE_TIMELINE_PARSED:
        if start <= ts < end:
            return label
    return None


def bin_index(temp):
    """Return the index of the temperature bin of a temperature."""
    if temp < 0:
        return 0
    if temp < 10:
        return 1
    if temp < 20:
        return 2
    if temp < 30:
        return 3
    return 4


def gap_flag_for(label, temp):
    """Flag a session whose damage state and temperature bin have no or few sessions in the coverage table."""
    if label not in COVERAGE_TABLE:
        return ""
    idx = bin_index(temp)
    count = COVERAGE_TABLE[label][idx]
    if count == 0:
        return f"EMPTY GAP for {label} ({BIN_LABELS[idx]}°C, 0 currently)"
    if count <= 2:
        return f"thin for {label} ({BIN_LABELS[idx]}°C, {count} currently)"
    return ""


def load_session(mat_path):
    """Load one session from either .mat format and return its sampling rate, start time and signals."""
    try:
        m = sio.loadmat(mat_path, simplify_cells=True)
        d = m["Dat"]
        fs = float(d["Fs"])
        recording_start = d["Time"]
        names = list(d["ChannelNames"])
        data = d["Data"]
    except NotImplementedError:
        with h5py.File(mat_path, "r") as f:
            grp = f["Dat"]
            fs = float(np.array(grp["Fs"]).flatten()[0])

            time_codes = np.array(grp["Time"]).flatten()
            try:
                candidate = "".join(chr(c) for c in time_codes)
                datetime.strptime(candidate[:19], "%Y-%m-%d %H:%M:%S")
                recording_start = candidate
            except (ValueError, OverflowError):
                recording_start = _time_from_filename(mat_path)

            name_refs = np.array(grp["ChannelNames"]).flatten()
            names = [_decode_string_cell(f, ref) for ref in name_refs]
            data = np.array(grp["Data"]).T

    col_of = {name: i for i, name in enumerate(names)}
    temp = float(np.mean(data[:, col_of["temp01"]])) if "temp01" in col_of else None
    return recording_start, temp


def main():
    """Scan the folder, label each session and write the table with the coverage gap flags."""
    paths = []
    for dirpath, _dirs, files in os.walk(FOLDER):
        for fn in files:
            if fn.lower().endswith(".mat"):
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    rows = []
    n_skipped = 0
    for path in paths:
        try:
            recording_start, temp = load_session(path)
        except Exception as e:
            print(f"  skipped {os.path.basename(path)}: {e}")
            n_skipped += 1
            continue
        if temp is None:
            print(f"  skipped {os.path.basename(path)}: no temp01 channel")
            n_skipped += 1
            continue
        label = damage_label_from_date(recording_start)
        rows.append({
            "source_file": os.path.basename(path),
            "recording_start": recording_start,
            "temperature_degC": temp,
            "damage_label": label,
            "gap_flag": gap_flag_for(label, temp) if label else "",
        })

    print(f"\nFound {len(paths)} .mat files, successfully read {len(rows)}.")
    df = pd.DataFrame(rows).sort_values("temperature_degC")

    print("\n=== Damage label(s) present in this folder ===")
    print(df["damage_label"].value_counts())

    print("\n=== All sessions, sorted coldest to warmest ===")
    print(df.to_string(index=False))

    gapped = df[df["gap_flag"] != ""]
    print(f"\n=== {len(gapped)} session(s) filling a gap or thin band for their own damage state ===")
    print(gapped.to_string(index=False))

    print(f"\nTemperature range in this folder: {df['temperature_degC'].min():.2f}°C to {df['temperature_degC'].max():.2f}°C")

    out_csv = os.path.join(FOLDER, "2020_08_temperature_scan.csv")
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved full scan to {OUT_CSV}")


if __name__ == "__main__":
    main()
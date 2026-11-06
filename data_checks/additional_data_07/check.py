"""
Scans a folder of raw LUMO files, gives each session its damage state, and flags sessions that fall in a temperature range with too little coverage for that state.

Needs: the raw LUMO data folder (LUMO_DATA_DIR).
"""
import os
from datetime import datetime
import numpy as np
import pandas as pd
import scipy.io as sio

FOLDER_PATH = os.path.join(os.environ.get("LUMO_DATA_DIR", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))), "additional_data_07")
OUT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data_checks", "additional_data_07"))
OUT_CSV = f"{OUT_DIR}/2021_07_temperature_scan.csv"

# Table 3, Wernitz et al. 2022, "Currently accessible data according to
# structural state and damage position (effective November 2021)"
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

# Table 8, the coverage table. Bin order: <0, 0-10, 10-20,
# 20-30, >30. These are per-damage-state gaps, not a single universal one.
TEMP_BINS = [-float("inf"), 0, 10, 20, 30, float("inf")]
BIN_LABELS = ["<0\u00b0C", "0-10\u00b0C", "10-20\u00b0C", "20-30\u00b0C", ">30\u00b0C"]
COVERAGE_TABLE = {
    "Healthy":  [0, 14, 13, 2, 1],
    "DAM3_010": [0, 1, 3, 1, 4],
    "DAM3_111": [4, 4, 0, 0, 1],
    "DAM4_010": [0, 3, 3, 2, 1],
    "DAM4_111": [1, 4, 3, 1, 0],
    "DAM6_010": [2, 4, 2, 0, 1],
    "DAM6_111": [0, 6, 2, 1, 0],
}


def damage_label_from_date(ts: datetime):
    """Return the damage label for a recording start time from Table 3 of Wernitz et al. 2022, or None outside the documented window."""
    for start, end, label in _DAMAGE_TIMELINE_PARSED:
        if start <= ts < end:
            return label
    return None


def bin_index(temp):
    """Return the index of the temperature bin of a temperature."""
    if temp is None:
        return None
    for i in range(len(TEMP_BINS) - 1):
        if TEMP_BINS[i] <= temp < TEMP_BINS[i + 1]:
            return i
    return None


def gap_flag_for(label, temp):
    """Flag a session whose damage state and temperature bin have no or few sessions in the coverage table."""
    idx = bin_index(temp)
    if idx is None:
        return ""
    if label is None:
        return "unknown damage label (outside documented timeline)"
    counts = COVERAGE_TABLE.get(label)
    if counts is None:
        return f"no coverage data for {label}"
    count = counts[idx]
    if count == 0:
        return f"EMPTY GAP for {label} ({BIN_LABELS[idx]}, 0 currently)"
    if count <= 2:
        return f"thin for {label} ({BIN_LABELS[idx]}, {count} currently)"
    return ""


def _decode_string_cell(f, ref):
    """Decode a cell array of HDF5 string references into a list of Python strings."""
    arr = np.array(f[ref]).flatten()
    return "".join(chr(c) for c in arr)


def load_session_temp(mat_path):
    """Return the recording start and temperature of one .mat file."""
    try:
        d = sio.loadmat(mat_path, simplify_cells=True)["Dat"]
        names = list(d["ChannelNames"])
        data = d["Data"]
        recording_start = d["Time"]
    except NotImplementedError:
        import h5py
        with h5py.File(mat_path, "r") as f:
            grp = f["Dat"]
            time_codes = np.array(grp["Time"]).flatten()
            recording_start = "".join(chr(c) for c in time_codes)
            name_refs = np.array(grp["ChannelNames"]).flatten()
            names = [_decode_string_cell(f, ref) for ref in name_refs]
            data = np.array(grp["Data"]).T

    col_of = {name: i for i, name in enumerate(names)}
    if "temp01" not in col_of:
        return recording_start, None
    temp = float(np.mean(data[:, col_of["temp01"]]))
    return recording_start, temp


def main():
    """Scan the folder, label each session and write the table with the coverage gap flags."""
    rows = []
    n_found = 0
    for dirpath, _, filenames in os.walk(FOLDER_PATH):
        for fname in sorted(filenames):
            if not fname.lower().endswith(".mat"):
                continue
            n_found += 1
            mat_path = os.path.join(dirpath, fname)
            try:
                recording_start, temp = load_session_temp(mat_path)
            except Exception as e:
                print(f"  skipped {fname}: {e}")
                continue

            ts = datetime.strptime(recording_start[:19], "%Y-%m-%d %H:%M:%S")
            label = damage_label_from_date(ts)

            rows.append({
                "source_file": fname,
                "recording_start": recording_start,
                "temperature_degC": temp,
                "damage_label": label if label else "UNKNOWN",
                "gap_flag": gap_flag_for(label, temp),
            })

    if not rows:
        print(f"No .mat files found under {FOLDER_PATH}")
        return

    df = pd.DataFrame(rows).sort_values("temperature_degC")
    print(f"Found {n_found} .mat files, successfully read {len(df)}.\n")

    print("=== Damage label(s) present in this folder ===")
    print(df["damage_label"].value_counts().to_string())

    print("\n=== All sessions, sorted coldest to warmest ===")
    print(df.to_string(index=False))

    useful = df[df["gap_flag"].str.contains("GAP|thin", na=False)]
    print(f"\n=== {len(useful)} session(s) filling a gap or thin band for their own damage state ===")
    if len(useful):
        print(useful.to_string(index=False))
    else:
        print("None of these files land in a gap/thin band for their assigned damage state.")

    print(f"\nTemperature range in this folder: "
          f"{df['temperature_degC'].min():.2f}\u00b0C to {df['temperature_degC'].max():.2f}\u00b0C")

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved full scan to {OUT_CSV}")


if __name__ == "__main__":
    main()
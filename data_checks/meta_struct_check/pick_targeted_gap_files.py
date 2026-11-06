"""
Picks extra archive sessions for chosen damage state and temperature gaps.

Needs: the raw LUMO data folder (LUMO_DATA_DIR).
"""

import glob
import os
import h5py
import numpy as np
from collections import defaultdict

META_STRUCT_DIR = os.path.join(os.environ.get("LUMO_DATA_DIR", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))), "meta_struct")
EXISTING_DATA_ROOTS = [
    os.environ.get("LUMO_DATA_DIR", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))),
]
TARGET_COUNT = 3  # aim for this many sessions (existing + new) per listed bin

# Only the (label, bin) pairs to be topped up.
NEEDS = {
    "Healthy": ["<0"],
    "DAM3_010": ["<0", "0-10", "10-20"],
    "DAM3_111": ["10-20", "20-30", ">30"],
    "DAM4_010": ["<0", ">30"],
    "DAM4_111": ["<0", "20-30", ">30"],
    "DAM6_010": ["20-30", ">30"],
    "DAM6_111": ["<0", "10-20", "20-30", ">30"],
}

CONDITION_LABELS = {
    (7, 8, 9):    {(1, 1, 1): "DAM3_111", (0, 1, 0): "DAM3_010"},
    (10, 11, 12): {(1, 1, 1): "DAM4_111", (0, 1, 0): "DAM4_010"},
    (16, 17, 18): {(1, 1, 1): "DAM6_111", (0, 1, 0): "DAM6_010"},
}

TEMP_BINS = [-float("inf"), 0, 10, 20, 30, float("inf")]
TEMP_BIN_LABELS = ["<0", "0-10", "10-20", "20-30", ">30"]


def label_condition(cond):
    """Map a condition code to a damage state."""
    if cond[0] == 2.0 and all(v == 0.0 for v in cond[1:]):
        return "Healthy"
    if cond[0] != 3.0:
        return None
    for positions, patterns in CONDITION_LABELS.items():
        sub = tuple(int(round(cond[p])) for p in positions)
        if sub in patterns:
            return patterns[sub]
    return None


def decode_string_cell(f, dataset):
    """Decode a cell array of HDF5 string references into a list of Python strings."""
    strings = []
    for ref in dataset[:, 0] if dataset.ndim == 2 else dataset[:]:
        obj = f[ref]
        chars = np.asarray(obj[()]).flatten()
        strings.append("".join(chr(c) for c in chars if c != 0))
    return strings


def deref_value(f, ref):
    """Read the value behind an HDF5 reference and decode it to a string or an array."""
    obj = f[ref]
    arr = np.asarray(obj[()])
    if arr.dtype.kind in ("u", "i") and arr.size > 0 and arr.max() < 0x110000:
        try:
            flat = arr.flatten()
            s = "".join(chr(c) for c in flat if c != 0)
            if s.isprintable() and s.strip():
                return s
        except Exception:
            pass
    if arr.size == 1:
        return arr.item()
    return tuple(arr.flatten().tolist())


def bin_of(temp):
    """Return the label of the temperature bin of a temperature."""
    for i in range(len(TEMP_BINS) - 1):
        if TEMP_BINS[i] <= temp < TEMP_BINS[i + 1]:
            return TEMP_BIN_LABELS[i]
    return TEMP_BIN_LABELS[-1]


def get_already_owned_filenames():
    """Return the names of the session files that are already available locally."""
    owned = set()
    for root in EXISTING_DATA_ROOTS:
        for path in glob.glob(os.path.join(root, "**", "SHMTS_*.mat"), recursive=True):
            fname = os.path.basename(path)
            if "meta_struct" not in fname:
                owned.add(os.path.splitext(fname)[0])
    return owned


def main():
    """Pick extra sessions for the listed damage state and temperature bin gaps."""
    owned = get_already_owned_filenames()
    print(f"Found {len(owned)} session files already owned.\n")

    paths = sorted(glob.glob(os.path.join(META_STRUCT_DIR, "SHMTS_*_meta_struct.mat")))
    by_label_bin = defaultdict(list)  # (label, bin) -> list of (temp, fname)

    for path in paths:
        with h5py.File(path, "r") as f:
            channel_names = decode_string_cell(f, f["Dat/ChannelNames"])
            if "mean-temp01" not in channel_names:
                continue
            temp_idx = channel_names.index("mean-temp01")
            info = f["Dat/Info"]
            data = f["Dat/Data"]
            n = info.shape[1]
            for col in range(n):
                cond = deref_value(f, info[1, col])
                if not isinstance(cond, tuple):
                    continue
                label = label_condition(cond)
                if label is None:
                    continue
                fname = deref_value(f, info[0, col])
                temp = float(data[temp_idx, col])
                by_label_bin[(label, bin_of(temp))].append((temp, fname))

    grand_total = []
    for label, bins in NEEDS.items():
        print(f"{label}:")
        for bl in bins:
            entries = by_label_bin.get((label, bl), [])
            current_owned = [(t, fn) for t, fn in entries if fn in owned]
            available = [(t, fn) for t, fn in entries if fn not in owned]
            current = len(current_owned)

            if not entries:
                print(f"  {bl}: 0 entries anywhere in the archive -- genuine gap, cannot be requested")
                continue

            need = max(0, TARGET_COUNT - current)
            if need == 0:
                print(f"  {bl}: already have {current} (target {TARGET_COUNT}) -- no request needed")
                continue
            if not available:
                print(f"  {bl}: have {current}, need {need} more, but no unrequested files left "
                      f"(only {len(entries)} exist in archive total)")
                continue

            available.sort(key=lambda x: x[0])
            n_pick = min(need, len(available))
            if n_pick == 1:
                idxs = [len(available) // 2]
            else:
                idxs = [round(i * (len(available) - 1) / (n_pick - 1)) for i in range(n_pick)]
            picks = [available[i] for i in idxs]

            note = "" if n_pick == need else f"  (only {len(available)} available, capped)"
            print(f"  {bl}: have {current}, requesting {n_pick} more{note}:")
            for temp, fname in picks:
                print(f"    {fname}.mat  ({temp:.2f} degC)")
                grand_total.append((label, bl, fname, temp))
        print()

    print(f"=== Total new files to request: {len(grand_total)} ===")
    for label, bl, fname, temp in grand_total:
        print(f"  {label} [{bl}]  {fname}.mat  ({temp:.2f} degC)")


if __name__ == "__main__":
    main()
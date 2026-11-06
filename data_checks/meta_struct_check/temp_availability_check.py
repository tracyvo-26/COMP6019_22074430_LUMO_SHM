"""
Checks whether remaining temperature gaps are real limits of the monitoring campaign or just not yet picked.

Needs: the raw LUMO data folder (LUMO_DATA_DIR).
"""
 
import glob
import os
import h5py
import numpy as np
from collections import defaultdict
 
META_STRUCT_DIR = os.path.join(os.environ.get("LUMO_DATA_DIR", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))), "meta_struct")
 
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
 
 
def main():
    """Report the temperatures available in the archive for each damage state."""
    paths = sorted(glob.glob(os.path.join(META_STRUCT_DIR, "SHMTS_*_meta_struct.mat")))
    print(f"Scanning {len(paths)} meta_struct files for FULL temperature availability per damage label.\n")
 
    by_label = defaultdict(list)  # label -> list of (temp, fname)
 
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
                by_label[label].append((temp, fname))
 
    for label in sorted(by_label.keys()):
        entries = by_label[label]
        print(f"=== {label}: {len(entries)} entries found across the whole archive ===")
        bin_counts = defaultdict(int)
        for temp, fname in entries:
            bin_counts[bin_of(temp)] += 1
        for bl in TEMP_BIN_LABELS:
            print(f"  {bl}: {bin_counts[bl]} entries available in archive")
        # Show a few example filenames for any bin that's currently empty
        # in the requested set.
        for bl in TEMP_BIN_LABELS:
            if bin_counts[bl] == 0:
                continue
            examples = [fn for t, fn in entries if bin_of(t) == bl][:3]
            print(f"    e.g. in {bl}: {examples}")
        print()
 
 
if __name__ == "__main__":
    main()
"""
Picks archive sessions that add new temperature coverage for each damage state.

Needs: the raw LUMO data folder (LUMO_DATA_DIR).
"""
 
import glob
import os
import h5py
import numpy as np
from collections import defaultdict
 
META_STRUCT_DIR = os.path.join(os.environ.get("LUMO_DATA_DIR", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))), "meta_struct")
N_TARGETS_PER_CATEGORY = 5  # coldest, hottest, + a few spread across the gap
 
# Position -> label mapping, checked against the known session files.
# Each damage location owns a 3-position block; the label's numeric
# suffix IS that block's flag pattern (e.g. "111" = all 3 on, "010" =
# only the middle one on).
CONDITION_LABELS = {
    (7, 8, 9):    {(1, 1, 1): "DAM3_111", (0, 1, 0): "DAM3_010"},
    (10, 11, 12): {(1, 1, 1): "DAM4_111", (0, 1, 0): "DAM4_010"},
    (16, 17, 18): {(1, 1, 1): "DAM6_111", (0, 1, 0): "DAM6_010"},
}
 
# Temperature range covered by the original sessions of each category.
# Archive entries outside this range add new coverage.
ORIGINAL_RANGE = {
    "Healthy":  (2.6, 34.8),
    "DAM3_010": (15.5, 40.4),
    "DAM3_111": (-3.7, 4.5),
    "DAM4_010": (7.7, 20.8),
    "DAM4_111": (4.1, 15.5),
    "DAM6_010": (0.7, 14.1),
    "DAM6_111": (6.4, 14.2),
}
 
 
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
 
 
def label_condition(cond):
    """Map a condition code to a damage label, or None if it matches no known block."""
    if cond[0] == 2.0 and all(v == 0.0 for v in cond[1:]):
        return "Healthy"
    if cond[0] != 3.0:
        return None
    for positions, patterns in CONDITION_LABELS.items():
        sub = tuple(int(cond[p]) for p in positions)
        if sub in patterns:
            return patterns[sub]
    return None
 
 
def main():
    """Pick archive sessions for each damage state whose temperature lies outside the original range."""
    paths = sorted(glob.glob(os.path.join(META_STRUCT_DIR, "SHMTS_*_meta_struct.mat")))
    print(f"Found {len(paths)} meta_struct files\n")
 
    # label -> list of (temperature, filename)
    by_label = defaultdict(list)
 
    for path in paths:
        with h5py.File(path, "r") as f:
            channel_names = decode_string_cell(f, f["Dat/ChannelNames"])
            temp_idx = channel_names.index("mean-temp01")
            data = f["Dat/Data"]
            info = f["Dat/Info"]
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
 
    for label, entries in sorted(by_label.items()):
        orig_min, orig_max = ORIGINAL_RANGE.get(label, (None, None))
        print(f"=== {label} ({len(entries)} long-term entries found) ===")
        if orig_min is None:
            print("  (no original-sample range on file for this label)")
            continue
        print(f"  Original sample range: {orig_min} to {orig_max} degC")
 
        # entries genuinely outside the original range
        new_cold = sorted([e for e in entries if e[0] < orig_min], key=lambda e: e[0])
        new_hot = sorted([e for e in entries if e[0] > orig_max], key=lambda e: -e[0])
 
        if not new_cold and not new_hot:
            print("  No entries fall outside the original range -- long-term data "
                  "doesn't add new coverage for this category.\n")
            continue
 
        print(f"  New coverage available: "
              f"{len(new_cold)} entries colder, {len(new_hot)} entries hotter than original range")
 
        targets = []
        if new_cold:
            targets.append(("coldest available", new_cold[0]))
            if len(new_cold) > 2:
                targets.append(("mid-cold (fills gap smoothly)", new_cold[len(new_cold) // 2]))
        if new_hot:
            targets.append(("hottest available", new_hot[0]))
            if len(new_hot) > 2:
                targets.append(("mid-hot (fills gap smoothly)", new_hot[len(new_hot) // 2]))
 
        for desc, (temp, fname) in targets[:N_TARGETS_PER_CATEGORY]:
            print(f"    [{desc}] {fname}  ({temp:.2f} degC)")
        print()
 
 
if __name__ == "__main__":
    main()
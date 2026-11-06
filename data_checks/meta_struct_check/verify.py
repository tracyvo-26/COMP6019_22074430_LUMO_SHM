"""
Checks the condition-code to damage-state mapping against the folder name of every original session.

Needs: the raw LUMO data folder (LUMO_DATA_DIR).
"""
 
import glob
import os
import re
import h5py
import numpy as np
import scipy.io as sio
from collections import defaultdict
 
ROOT_DATA_DIR = os.environ.get("LUMO_DATA_DIR", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")))
META_STRUCT_DIR = os.path.join(os.environ.get("LUMO_DATA_DIR", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))), "meta_struct")
 
CONDITION_LABELS = {
    (7, 8, 9):    {(1, 1, 1): "DAM3_111", (0, 1, 0): "DAM3_010"},
    (10, 11, 12): {(1, 1, 1): "DAM4_111", (0, 1, 0): "DAM4_010"},
    (16, 17, 18): {(1, 1, 1): "DAM6_111", (0, 1, 0): "DAM6_010"},
}
 
 
def label_condition(cond):
    """Map a condition code to a damage state."""
    if cond[0] == 2.0 and all(v == 0.0 for v in cond[1:]):
        return "Healthy"
    if cond[0] != 3.0:
        return f"UNRECOGNISED(state={cond[0]})"
    for positions, patterns in CONDITION_LABELS.items():
        sub = tuple(int(round(cond[p])) for p in positions)
        if sub in patterns:
            return patterns[sub]
    return "UNRECOGNISED(damage active, no known block matched)"
 
 
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
 
 
def true_label_from_path(path):
    """Read the damage state from the folder name of a session file."""
    parent = os.path.basename(os.path.dirname(path))
    m = re.search(r"(DAM\d+_\d+)", parent, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    if "healthy" in parent.lower():
        return "Healthy"
    return None
 
 
def load_meta_struct_index(month_str, cache):
    """Return {filename: conditionCode} for one month, cached."""
    if month_str in cache:
        return cache[month_str]
    path = os.path.join(META_STRUCT_DIR, f"SHMTS_{month_str}_meta_struct.mat")
    if not os.path.exists(path):
        cache[month_str] = None
        return None
    index = {}
    with h5py.File(path, "r") as f:
        info = f["Dat/Info"]
        n = info.shape[1]
        for col in range(n):
            fname = deref_value(f, info[0, col])
            cond = deref_value(f, info[1, col])
            if isinstance(fname, str) and isinstance(cond, tuple):
                index[fname] = cond
    cache[month_str] = index
    return index
 
 
def main():
    """Check the condition-code mapping against the folder-name label of every original session file."""
    session_paths = sorted(glob.glob(os.path.join(ROOT_DATA_DIR, "**", "SHMTS_*.mat"), recursive=True))
    # exclude the meta_struct summary files themselves
    session_paths = [p for p in session_paths if "meta_struct" not in os.path.basename(p)]
    print(f"Found {len(session_paths)} session files under {ROOT_DATA_DIR}\n")
 
    cache = {}
    results = defaultdict(lambda: {"match": 0, "mismatch": 0, "no_meta_struct": 0, "not_found_in_index": 0})
    mismatches = []
 
    for path in session_paths:
        true_label = true_label_from_path(path)
        if true_label is None:
            continue
        fname_base = os.path.splitext(os.path.basename(path))[0]  # e.g. SHMTS_202103190009
        m = re.search(r"SHMTS_(\d{6})\d{6}", fname_base)
        if not m:
            continue
        month_str = m.group(1)
 
        index = load_meta_struct_index(month_str, cache)
        if index is None:
            results[true_label]["no_meta_struct"] += 1
            continue
        if fname_base not in index:
            results[true_label]["not_found_in_index"] += 1
            continue
 
        cond = index[fname_base]
        decoded_label = label_condition(cond)
 
        if decoded_label == true_label:
            results[true_label]["match"] += 1
        else:
            results[true_label]["mismatch"] += 1
            mismatches.append((path, true_label, decoded_label, cond))
 
    print("=== Per-category verification summary ===")
    total_match = total_mismatch = 0
    for label, r in sorted(results.items()):
        print(f"  {label:12s} match={r['match']:3d}  mismatch={r['mismatch']:3d}  "
              f"no_meta_struct_for_month={r['no_meta_struct']:3d}  "
              f"file_not_in_that_months_index={r['not_found_in_index']:3d}")
        total_match += r["match"]
        total_mismatch += r["mismatch"]
 
    print(f"\nTOTAL: {total_match} matched, {total_mismatch} mismatched "
          f"(out of files that had a downloaded meta_struct for their month)")
 
    if mismatches:
        print("\n=== MISMATCHES (these need looking at) ===")
        for path, true_label, decoded_label, cond in mismatches:
            print(f"  {os.path.basename(path)}: true={true_label}  decoded={decoded_label}  cond={cond}")
    else:
        print("\nNo mismatches found among files that could be checked.")
 
 
if __name__ == "__main__":
    main()
"""
Decodes the condition code stored in the LUMO meta_struct files into a damage state.

Needs: the raw LUMO data folder (LUMO_DATA_DIR).
"""
 
import glob
import os
import re
import h5py
import numpy as np
from collections import defaultdict
 
META_STRUCT_DIR = os.path.join(os.environ.get("LUMO_DATA_DIR", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))), "meta_struct")
SAMPLES_PER_CONDITION = 3
 
 
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
 
 
def main():
    """Group the meta_struct entries by condition code and match them to damage states."""
    paths = sorted(glob.glob(os.path.join(META_STRUCT_DIR, "SHMTS_*_meta_struct.mat")))
    print(f"Found {len(paths)} meta_struct files\n")
 
    # condition_pattern -> {"months": set, "count": int, "temps": [...], "samples": [...]}
    by_condition = defaultdict(lambda: {"months": set(), "count": 0, "temps": [], "samples": []})
 
    for path in paths:
        fname_month = os.path.basename(path).replace("SHMTS_", "").replace("_meta_struct.mat", "")
        with h5py.File(path, "r") as f:
            channel_names = decode_string_cell(f, f["Dat/ChannelNames"])
            temp_idx = channel_names.index("mean-temp01")
            data = f["Dat/Data"]
            info = f["Dat/Info"]
            n = info.shape[1]
 
            for col in range(n):
                cond = deref_value(f, info[1, col])
                fname = deref_value(f, info[0, col])
                temp = data[temp_idx, col]
 
                entry = by_condition[cond]
                entry["months"].add(fname_month)
                entry["count"] += 1
                entry["temps"].append(temp)
                if len(entry["samples"]) < SAMPLES_PER_CONDITION:
                    entry["samples"].append(fname)
 
    print(f"=== {len(by_condition)} distinct conditionCode patterns found across all months ===\n")
    for cond, entry in sorted(by_condition.items(), key=lambda kv: -kv[1]["count"]):
        temps = np.array(entry["temps"])
        # try to pull a damage label out of the sample filenames, same
        # convention as the original session data (e.g. ..._DAM3_111_...)
        labels_seen = set()
        for s in entry["samples"]:
            m = re.search(r"(DAM\d+_\d+|Healthy|healthy)", str(s))
            if m:
                labels_seen.add(m.group(1))
        print(f"condition={cond}")
        print(f"  months: {sorted(entry['months'])}")
        print(f"  total entries: {entry['count']}")
        print(f"  temp range: {temps.min():.2f} to {temps.max():.2f} degC")
        print(f"  label(s) found in filenames: {labels_seen or 'none matched -- see raw samples below'}")
        print(f"  sample filenames: {entry['samples']}")
        print()
 
 
if __name__ == "__main__":
    main()
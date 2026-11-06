"""
Checks whether the long-term archive contains any damage periods or only Healthy data.

Needs: the raw LUMO data folder (LUMO_DATA_DIR).
"""
 
import glob
import os
import h5py
import numpy as np
from collections import Counter
 
META_STRUCT_DIR = os.path.join(os.environ.get("LUMO_DATA_DIR", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))), "meta_struct")
 

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
    """Report the condition codes found in the meta_struct files and the temperatures during each condition."""
    paths = sorted(glob.glob(os.path.join(META_STRUCT_DIR, "SHMTS_*_meta_struct.mat")))
    print(f"Found {len(paths)} meta_struct files\n")
 
    for path in paths:
        fname = os.path.basename(path)
        with h5py.File(path, "r") as f:
            channel_names = decode_string_cell(f, f["Dat/ChannelNames"])
            temp_idx = channel_names.index("mean-temp01")
            data = f["Dat/Data"]
            info = f["Dat/Info"]
            n = info.shape[1]
 
            conditions = []
            for col in range(n):
                conditions.append(deref_value(f, info[1, col]))
 
            counts = Counter(conditions)
            print(f"{fname}  ({n} entries)")
            for code, count in counts.most_common():
                # temperature range specifically during this condition
                idxs = [i for i, c in enumerate(conditions) if c == code]
                temps = data[temp_idx, idxs]
                print(f"  condition={code!r}: {count} entries, "
                      f"temp range {np.nanmin(temps):.2f} to {np.nanmax(temps):.2f} degC")
            print()
 
 
if __name__ == "__main__":
    main()
"""
Lists Healthy sessions above 30 degC in the archive, mildest first.

Needs: the raw LUMO data folder (LUMO_DATA_DIR).
"""
 
import glob
import os
import h5py
import numpy as np
 
META_STRUCT_DIR = os.path.join(os.environ.get("LUMO_DATA_DIR", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))), "meta_struct")
EXISTING_DATA_ROOTS = [
    os.environ.get("LUMO_DATA_DIR", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))),
]
# Files already picked, so they are not listed again.
ALREADY_REQUESTED = {
    "SHMTS_202107251149", "SHMTS_202106181336",
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
    """List the Healthy sessions above 30 degC that are not yet owned, mildest first."""
    owned = get_already_owned_filenames() | ALREADY_REQUESTED
    paths = sorted(glob.glob(os.path.join(META_STRUCT_DIR, "SHMTS_*_meta_struct.mat")))
 
    entries = []
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
                if not (isinstance(cond, tuple) and cond[0] == 2.0 and all(v == 0.0 for v in cond[1:])):
                    continue  # only Healthy
                fname = deref_value(f, info[0, col])
                temp = float(data[temp_idx, col])
                if temp > 30 and fname not in owned:
                    entries.append((temp, fname))
 
    entries.sort(key=lambda x: x[0])  # mildest (closest to 30) first
    print(f"Found {len(entries)} unrequested Healthy sessions above 30 degC.\n")
 
    N_SHOW = 15
    idxs = [round(i * (len(entries) - 1) / (N_SHOW - 1)) for i in range(N_SHOW)]
    print(f"{N_SHOW} examples spread evenly from mildest to hottest:")
    for i in idxs:
        temp, fname = entries[i]
        print(f"  {fname}.mat  ({temp:.2f} degC)")
 
 
if __name__ == "__main__":
    main()
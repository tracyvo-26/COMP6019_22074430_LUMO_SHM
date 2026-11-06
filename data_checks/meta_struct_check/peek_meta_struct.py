"""
Prints the channel names stored in a LUMO meta_struct file.

Needs: the raw LUMO data folder (LUMO_DATA_DIR).
"""
 
import h5py
import numpy as np
import os
 
MAT_PATH = os.path.join(os.environ.get("LUMO_DATA_DIR", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))), "SHMTS_202101_meta_struct.mat")
 
 
def decode_string_cell(f, dataset):
    """Decode HDF5 references to MATLAB strings into a list of Python strings."""
    strings = []
    for ref in dataset[:, 0] if dataset.ndim == 2 else dataset[:]:
        obj = f[ref]
        arr = obj[()]
        # MATLAB v7.3 chars are stored as uint16 codes, shape (n_chars, 1) or (1, n_chars)
        chars = np.asarray(arr).flatten()
        s = "".join(chr(c) for c in chars if c != 0)
        strings.append(s)
    return strings
 
 
with h5py.File(MAT_PATH, "r") as f:
    print("=== Dat/ChannelNames (66 entries) ===")
    channel_names = decode_string_cell(f, f["Dat/ChannelNames"])
    for i, name in enumerate(channel_names):
        print(f"  [{i}] {name}")
 
    print("\n=== Dat/InfoNames (2 entries) ===")
    info_names = decode_string_cell(f, f["Dat/InfoNames"])
    for i, name in enumerate(info_names):
        print(f"  [{i}] {name}")
 
    print("\n=== Dat/Fs (sampling rate) ===")
    print(f["Dat/Fs"][()])
 
    print("\n=== Dat/Time (likely a MATLAB datevec: [year month day hour min sec]) ===")
    print(f["Dat/Time"][()])
 
    print("\n=== Dat/Timestamps ===")
    print(f["Dat/Timestamps"][()])
 
    print("\n=== Dat/Data shape and a peek at first 5 columns, first 5 rows ===")
    data = f["Dat/Data"]
    print("shape:", data.shape)
    print(data[:5, :5])
 
    print("\n=== Dat/Info shape and first 10 columns ===")
    info = f["Dat/Info"]
    print("shape:", info.shape)
    print(info[:, :10])
 
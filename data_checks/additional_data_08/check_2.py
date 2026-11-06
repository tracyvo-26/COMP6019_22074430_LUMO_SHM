"""
Quick look at the contents of one raw LUMO file.

Needs: the raw LUMO data folder (LUMO_DATA_DIR).
"""
import h5py
import numpy as np
import os

path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data_checks", "additional_data_08", "08", "SHMTS_202008010002.mat"))

with h5py.File(path, "r") as f:
    grp = f["Dat"]

    for key in ["Time", "Timestamps"]:
        ds = grp[key]
        raw = np.array(ds)
        print(f"\n{key}:")
        print("  shape:", raw.shape, " dtype:", raw.dtype)
        print("  flattened values:", raw.flatten())

        # try reinterpreting the raw bytes as float64, in case this is a
        # timestamp stored as a double but read back as the wrong dtype
        as_bytes = raw.tobytes()
        if len(as_bytes) % 8 == 0:
            as_f64 = np.frombuffer(as_bytes, dtype="<f8")
            print("  reinterpreted as float64:", as_f64)

    # sanity check: what does Fs and ChannelNames look like here?
    print("\nFs:", np.array(grp["Fs"]))
    print("ChannelNames shape/dtype:", grp["ChannelNames"].shape, grp["ChannelNames"].dtype)
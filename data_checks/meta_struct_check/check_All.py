
"""
Reports the format and temperature range of every meta_struct file in a folder.

Needs: the raw LUMO data folder (LUMO_DATA_DIR).
"""
 
import glob
import os
import h5py
import numpy as np
 
META_STRUCT_DIR = os.path.join(os.environ.get("LUMO_DATA_DIR", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))), "meta_struct")
 
 
def decode_string_cell(f, dataset):
    """Decode a cell array of HDF5 string references into a list of Python strings."""
    strings = []
    for ref in dataset[:, 0] if dataset.ndim == 2 else dataset[:]:
        obj = f[ref]
        chars = np.asarray(obj[()]).flatten()
        strings.append("".join(chr(c) for c in chars if c != 0))
    return strings
 
 
def main():
    """Print the format, channel count, sampling interval and temperature range of every meta_struct file."""
    paths = sorted(glob.glob(os.path.join(META_STRUCT_DIR, "SHMTS_*_meta_struct.mat")))
    if not paths:
        print(f"No SHMTS_*_meta_struct.mat files found under {META_STRUCT_DIR}")
        return
 
    print(f"Found {len(paths)} meta_struct files\n")
 
    for path in paths:
        fname = os.path.basename(path)
        try:
            with h5py.File(path, "r") as f:
                data = f["Dat/Data"]
                fs = f["Dat/Fs"][()].flatten()[0]
                channel_names = decode_string_cell(f, f["Dat/ChannelNames"])
 
                temp_idx = channel_names.index("mean-temp01") if "mean-temp01" in channel_names else None
                temp_summary = "n/a"
                if temp_idx is not None:
                    temp_vals = data[temp_idx, :]
                    temp_summary = (f"min={np.nanmin(temp_vals):.2f} "
                                     f"max={np.nanmax(temp_vals):.2f} "
                                     f"mean={np.nanmean(temp_vals):.2f}")
 
                print(f"{fname}")
                print(f"  Dat/Data shape: {data.shape}  (channels x time entries)")
                print(f"  n_channels: {len(channel_names)}")
                print(f"  Fs: {fs}  (interval = {1/fs/60:.1f} min between entries)")
                print(f"  mean-temp01 (deg C): {temp_summary}")
                print()
        except Exception as e:
            print(f"{fname}: FAILED to read -- {e}\n")
 
 
if __name__ == "__main__":
    main()
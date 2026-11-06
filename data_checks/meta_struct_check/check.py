"""
Quick look at the structure of one LUMO meta_struct file.

Needs: the raw LUMO data folder (LUMO_DATA_DIR).
"""
import h5py
import os

with h5py.File(os.path.join(os.environ.get("LUMO_DATA_DIR", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))), "SHMTS_202101_meta_struct.mat"), "r") as f:
    def show(name, obj):
        """Print the name and shape of every item in the file."""
        print(name, "-", "group" if isinstance(obj, h5py.Group) else obj.shape)
    f.visititems(show)
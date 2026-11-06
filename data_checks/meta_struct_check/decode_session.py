"""
Checks that a session file carries the same condition code as the meta_struct summaries.

Needs: the raw LUMO data folder (LUMO_DATA_DIR).
"""
 
import scipy.io as sio
import os
 
MAT_PATH = os.path.join(os.environ.get("LUMO_DATA_DIR", os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))), "exemplary_datasets_dam6_111", "02_DAM6_111", "SHMTS_202010140009.mat")
TRUE_LABEL = "DAM6_111"   # whatever this file's folder name says
 
d = sio.loadmat(MAT_PATH, simplify_cells=True)["Dat"]
 
print(f"File: {MAT_PATH}")
print(f"Known true label (from folder name): {TRUE_LABEL}\n")
 
print("=== All fields present in this file's 'Dat' struct ===")
for key in d.keys():
    val = d[key]
    shape = getattr(val, "shape", None)
    print(f"  {key}: {'shape ' + str(shape) if shape is not None else type(val).__name__}")
 
print("\n=== Looking for Info / conditionCode specifically ===")
if "Info" in d:
    print("Info:", d["Info"])
if "InfoNames" in d:
    print("InfoNames:", d["InfoNames"])
if "conditionCode" in d:
    print("conditionCode:", d["conditionCode"])
if not any(k in d for k in ("Info", "InfoNames", "conditionCode")):
    print("None of Info/InfoNames/conditionCode found directly under 'Dat' -- "
          "printing the full key list above so we can look for a differently "
          "named equivalent field.")
 

"""
Exploratory run without the two classes DAM3_010 and DAM4_010.

Needs: compare_vanilla_cal.py.
"""
from __future__ import annotations
import importlib.util
import os
import sys
from types import SimpleNamespace
 
import numpy as np
import torch
 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "core"))                      # shared modules (compare_vanilla_cal*.py)
HERE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "results"))   # all CSV results of gin_cal live here
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)
spec = importlib.util.spec_from_file_location("cvc", os.path.join(CORE_DIR, "compare_vanilla_cal.py"))
m = importlib.util.module_from_spec(spec)
sys.modules["cvc"] = m
spec.loader.exec_module(m)
 
EXCLUDED = ["DAM3_010", "DAM4_010"]
 
 
def exclude_classes(graphs, enc, excluded):
    """Remove the graphs of the excluded classes and relabel the remaining classes."""
    names = list(enc.classes_)
    missing = [c for c in excluded if c not in names]
    assert not missing, f"classes not found: {missing}; available: {names}"
    keep = [c for c in names if c not in excluded]
    idx = {c: i for i, c in enumerate(keep)}
    out = []
    for g in graphs:
        name = names[g.y.item()]
        if name in excluded:
            continue
        g2 = g.clone()
        g2.y = torch.tensor([idx[name]], dtype=g.y.dtype)
        out.append(g2)
    return out, SimpleNamespace(classes_=np.array(keep))
 
 
_orig_build = m.build_dataset
 
 
def build_without_excluded(df, **kw):
    """Build the graphs without the excluded classes."""
    graphs, enc = _orig_build(df, **kw)
    graphs, enc = exclude_classes(graphs, enc, EXCLUDED)
    print(f"Excluded {EXCLUDED}: {len(graphs)} graphs, classes {list(enc.classes_)}")
    return graphs, enc
 
 
m.build_dataset = build_without_excluded
m.DATASET_LABEL = f"without {', '.join(EXCLUDED)}"
m.CSV_PATH = os.path.join(HERE, "compare_vanilla_cal_excl_by_seed_3loss" + ("_quick" if m.QUICK else "") + ".csv")
 
if __name__ == "__main__":
    m.main()
 
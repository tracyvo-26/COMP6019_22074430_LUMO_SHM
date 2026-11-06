"""
Exploratory run that adds the mode-shape (spatial) features to the node features of the models.

Needs: compare_vanilla_cal.py, graph_builder.py and spatial_features.csv.
"""
from __future__ import annotations
import importlib.util
import os
import sys
 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "core"))                      # shared modules (compare_vanilla_cal*.py)
HERE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "results"))   # all CSV results of gin_cal live here
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)
spec = importlib.util.spec_from_file_location("cvc", os.path.join(CORE_DIR, "compare_vanilla_cal.py"))
m = importlib.util.module_from_spec(spec)
sys.modules["cvc"] = m
spec.loader.exec_module(m)
import graph_builder as gb          # compare_vanilla_cal.py already put the parent folder on sys.path
 
# name -> (USE_AMP, USE_CURV, USE_DET)
CONFIGS = {
    "det_only": (False, False, True),
    "amp":      (True,  False, True),
    "curv":     (False, True,  True),
    "amp_curv": (True,  True,  True),
}
CONFIGS_TO_RUN = ["amp_curv"]       # add "det_only", "amp", "curv" to run the full ablation
 
 
def make_builder(cfg_name):
    """Return a graph builder that adds the spatial node features of one configuration."""
    amp, curv, det = CONFIGS[cfg_name]
 
    def build_with_spatial(df, **kw):
        """Build the graphs with the spatial node features of the chosen configuration."""
        gb.USE_AMP, gb.USE_CURV, gb.USE_DET = amp, curv, det
        gb.USE_B2Y = False
        graphs, enc = gb.build_dataset(gb.merge_spatial(df), use_spatial=True, **kw)
        print(f"Spatial config '{cfg_name}': node feature dim = {graphs[0].x.shape[1]} "
              f"(6 base + {graphs[0].x.shape[1] - 6} spatial)")
        return graphs, enc
    return build_with_spatial
 
 
if __name__ == "__main__":
    names = sys.argv[1:] or CONFIGS_TO_RUN
    for name in names:
        assert name in CONFIGS, f"unknown config {name}; choose from {list(CONFIGS)}"
        print("\n" + "#" * 100 + f"\n# SPATIAL CONFIG: {name}\n" + "#" * 100)
        m.build_dataset = make_builder(name)
        m.DATASET_LABEL = f"all sessions, spatial features: {name}"
        m.CSV_PATH = os.path.join(HERE, f"compare_vanilla_cal_spatial_{name}_by_seed_3loss"
                                  + ("_quick" if m.QUICK else "") + ".csv")
        m.main()
 
"""
Setting 2a: the project GIN with settings closer to the original CAL code.

Needs: compare_vanilla_cal_backward.py.
"""
from __future__ import annotations
import os
import sys
 
import os as _os, sys as _sys
 
_sys.path.insert(0, _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "core")))
 
import compare_vanilla_cal_backward as bw
 
m = bw.m
INTERV = os.environ.get("PAPER_INTERV", "random")
WD = float(os.environ.get("PAPER_WD", "0"))
assert INTERV in ("random", "stratum"), "PAPER_INTERV must be random or stratum"
m.INTERVENTION = INTERV
m.WEIGHT_DECAY = WD
bw.TAG = f"n186_paper_{INTERV}_wd{WD:g}"
 
if __name__ == "__main__":
    exp, steps, variants = bw.parse(sys.argv[1:])
    assert hasattr(m, "EDGE_ATTR_DIM"), "compare_vanilla_cal.py must take the edge width from a module-level EDGE_ATTR_DIM"
    print(f"PAPER-LIKE settings: intervention {m.INTERVENTION}, weight decay {m.WEIGHT_DECAY}; "
          f"experiment {exp}; steps {steps}; variants {variants}; seeds {m.SEEDS}; lambdas ({m.LAMBDA1}, {m.LAMBDA2})")
    m.report = lambda res: None
    for s in steps:
        removed, node_keep, edge_keep = bw.kept_features(s)
        print("\n" + "#" * 100 + f"\n# STEP {s} (paper-like): removed so far {removed or 'nothing'}\n" + "#" * 100)
        m.build_dataset = bw.make_builder(exp, s)
        m.EDGE_ATTR_DIM = len(edge_keep)
        m.DATASET_LABEL = f"{bw.ef.LABEL[exp]}, backward step {s}, paper-like {INTERV} wd {WD:g}"
        m.CSV_PATH = bw.csv_path(exp, s)
        m.VARIANTS_TO_RUN = variants
        m.main()
    bw.curve(exp, list(range(bw.MAX_STEP + 1)), variants)
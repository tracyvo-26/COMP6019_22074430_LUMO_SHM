"""
Turns the feature tables into one graph per session (21 sensor nodes, 31 edges), ready for the GIN models.

Needs: lumo_features.csv and between_channel_features.csv (01_data_and_features); spatial_features.csv for the spatial runs.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import os

# node spec: name -> (one-hot type vector, [feature-column-prefix])
# type vector = [is_accel_x, is_accel_y, is_strain]
ACCEL_X_TYPE = (1.0, 0.0, 0.0)
ACCEL_Y_TYPE = (0.0, 1.0, 0.0)
STRAIN_TYPE = (0.0, 0.0, 1.0)

NODES = (
    [(f"accel{i:02d}x", ACCEL_X_TYPE, f"accel{i:02d}x") for i in range(1, 10)]
    + [(f"accel{i:02d}y", ACCEL_Y_TYPE, f"accel{i:02d}y") for i in range(1, 10)]
    + [(f"strain{i:02d}", STRAIN_TYPE, f"strain{i:02d}") for i in range(1, 4)]
)
NODE_NAMES = [n for n, _, _ in NODES]
NODE_INDEX = {name: i for i, name in enumerate(NODE_NAMES)}

# fixed edge list: x-chain, y-chain, same-level x-y pairs, fan-out to base
_EDGES = (
    [(f"accel{i:02d}x", f"accel{i+1:02d}x") for i in range(1, 9)]      # x-chain
    + [(f"accel{i:02d}y", f"accel{i+1:02d}y") for i in range(1, 9)]    # y-chain
    + [(f"accel{i:02d}x", f"accel{i:02d}y") for i in range(1, 10)]     # same-level x-y
    + [("accel09x", "strain01"), ("accel09x", "strain02"), ("accel09x", "strain03")]
    + [("accel09y", "strain01"), ("accel09y", "strain02"), ("accel09y", "strain03")]
)


def _edge_index():
    """Build the symmetric edge_index tensor from the fixed edge list."""
    src, dst = [], []
    for a, b in _EDGES:
        i, j = NODE_INDEX[a], NODE_INDEX[b]
        src += [i, j]
        dst += [j, i]
    return torch.tensor([src, dst], dtype=torch.long)


EDGE_INDEX = _edge_index()  # same for every graph -- fixed topology


def row_to_node_features(row) -> np.ndarray:
    """Return the (21, 6) node feature array of one session."""
    feats = np.zeros((len(NODES), 6), dtype=np.float32)
    for idx, (name, type_vec, prefix) in enumerate(NODES):
        rms = row[f"{prefix}_rms"]
        peak = row[f"{prefix}_peak"]
        freq = row[f"{prefix}_freq_hz"]
        feats[idx] = [rms, peak, freq, *type_vec]
    return feats


class LabelEncoder:
    """Minimal label encoder mapping damage labels to integer class ids."""
    def __init__(self, labels):
        self.classes_ = sorted(set(labels))
        self._to_id = {lbl: i for i, lbl in enumerate(self.classes_)}

    def transform(self, label: str) -> int:
        """Return the integer class id of a label."""
        return self._to_id[label]

    def __repr__(self):
        return f"LabelEncoder({self.classes_})"


def build_dataset(df, label_encoder=None, num_eov_bins=5, use_spatial=False):
    """Build the list of graphs and the label encoder from the session table."""
    if label_encoder is None:
        label_encoder = LabelEncoder(df["damage_label"].tolist())

    # EOV bins from the temperature spread in this data
    temps = df["temperature_degC"].to_numpy()
    bin_edges = np.quantile(temps, np.linspace(0, 1, num_eov_bins + 1))
    bin_edges[0] -= 1e-6

    graphs = []
    for _, row in df.iterrows():
        x_np = row_to_node_features(row)
        if use_spatial:
            x_np = np.concatenate([x_np, row_to_spatial_features(row, NODE_NAMES)], axis=1)
        x = torch.tensor(x_np, dtype=torch.float)
        edge_attr = torch.tensor(edge_features_from_row(row), dtype=torch.float)
        y = torch.tensor([label_encoder.transform(row["damage_label"])], dtype=torch.long)
        temp = torch.tensor([row["temperature_degC"]], dtype=torch.float)
        eov_bin = int(np.searchsorted(bin_edges, row["temperature_degC"], side="right") - 1)
        eov_bin = min(max(eov_bin, 0), num_eov_bins - 1)

        data = Data(
            x=x,
            edge_index=EDGE_INDEX,
            edge_attr=edge_attr,
            y=y,
            temp=temp,
            eov_bin=torch.tensor([eov_bin], dtype=torch.long),
            source_file=row.get("source_file", None),
        )
        graphs.append(data)

    return graphs, label_encoder

# edge feature spec, matches between_channel_features.py's column naming
FEAT_TYPES = ["xcov"] + [f"{feat}_{label}" for feat in ["coh", "trans"] for label in ("b1", "t1", "b2")]
# 7 values per edge: xcov, coh_b1, coh_t1, coh_b2, trans_b1, trans_t1, trans_b2

EDGE_ATTR_DIM = len(FEAT_TYPES)

DEFAULT_SPATIAL_CSV = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "01_data_and_features", "spatial_features", "spatial_features.csv"))
 
SPATIAL_MODES = ("B1", "B2")   # bending modes; each node uses its OWN direction (x node -> B1x/B2x, y node -> B1y/B2y)
USE_B2Y = False                # B2y may mix B2-x and B2-y; False -> y nodes get zeros in the B2 slot
USE_AMP, USE_CURV, USE_DET = True, True, True
 
 
def spatial_feature_dim():
    """Number of spatial node features per node for the current configuration."""
    return len(SPATIAL_MODES) * (int(USE_AMP) + int(USE_CURV) + int(USE_DET))
 
 
def merge_spatial(df, spatial_csv=DEFAULT_SPATIAL_CSV):
    """Left-merge the spatial features onto the session table."""
    sp = pd.read_csv(spatial_csv).drop(columns=["damage_label", "temperature_degC"], errors="ignore")
    out = df.merge(sp, left_on="source_file", right_on="session", how="left", suffixes=("", "_sp"))
    n_missing = out["det_B1x"].isna().sum()
    print(f"Spatial merge: {len(out) - n_missing}/{len(out)} sessions have spatial features")
    return out
 
 
def _get(row, col):
    """Read a value from a row, returning 0 if it is missing."""
    v = row.get(col, np.nan)
    return 0.0 if pd.isna(v) else float(v)
 
 
def row_to_spatial_features(row, node_names):
    """Return the (n_nodes, D) spatial feature array of one session."""
    D = spatial_feature_dim()
    feats = np.zeros((len(node_names), D), dtype=np.float32)
    for i, name in enumerate(node_names):
        if not name.startswith("accel"):
            continue
        level, d = int(name[5:7]), name[7]
        vals = []
        for M in SPATIAL_MODES:
            tag = f"{M}{d}"
            skip = (M == "B2" and d == "y" and not USE_B2Y)
            det = 0.0 if skip else _get(row, f"det_{tag}")
            ok = det == 1.0
            if USE_AMP:
                vals.append(_get(row, f"amp_{tag}_L{level}") if ok else 0.0)
            if USE_CURV:
                vals.append(_get(row, f"curv_{tag}_L{level}") if ok and 2 <= level <= 8 else 0.0)
            if USE_DET:
                vals.append(det)
        feats[i] = vals
    return feats
 


def edge_features_from_row(row) -> np.ndarray:
    """Return the (62, 7) edge feature array of one session, one row per directed edge."""
    per_edge = np.zeros((len(_EDGES), EDGE_ATTR_DIM), dtype=np.float32)
    for idx, (a, b) in enumerate(_EDGES):
        pair_name = f"{a}_{b}"
        for f_idx, feat in enumerate(FEAT_TYPES):
            col = f"{pair_name}_{feat}"
            val = row.get(col, np.nan)
            per_edge[idx, f_idx] = 0.0 if pd.isna(val) else val
    # duplicate each undirected edge's vector for both directions, matching
    # _edge_index()'s src += [i, j]; dst += [j, i] construction
    return np.repeat(per_edge, 2, axis=0)

DEFAULT_CSV = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "01_data_and_features", "lumo_features.csv"))
DEFAULT_BETWEEN_CHANNEL_CSV = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "01_data_and_features", "between_channel and temp explore", "between_channel_features", "between_channel_features.csv"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV, help="path to lumo_dataset.py's output CSV")
    parser.add_argument("--selftest", action="store_true", help="run the fake-row sanity check instead of loading real data")
    parser.add_argument("--between-channel-csv", default=DEFAULT_BETWEEN_CHANNEL_CSV)
    parser.add_argument("--use-spatial", action="store_true", help="append spatial (modal) node features")
    args = parser.parse_args()

    if args.selftest:
        fake_row = {"damage_label": "Healthy", "temperature_degC": 5.0}
        for name, _, prefix in NODES:
            fake_row[f"{prefix}_rms"] = 0.001
            fake_row[f"{prefix}_peak"] = 0.01
            fake_row[f"{prefix}_freq_hz"] = 2.8
        df = pd.DataFrame([fake_row, {**fake_row, "damage_label": "DAM3_111", "temperature_degC": 0.0}])
    else:
        df = pd.read_csv(args.csv)
        bc_df = pd.read_csv(args.between_channel_csv)
        df = df.merge(bc_df.drop(columns=["damage_label", "temperature_degC"]),
                      left_on="source_file", right_on="session", how="left")
        n_missing = df["session"].isna().sum()
        if n_missing:
            print(f"Warning: {n_missing} sessions had no matching between-channel feature row")
        print(f"Loaded {len(df)} sessions, merged with between-channel features")
        if args.use_spatial:
            df = merge_spatial(df)

    graphs, enc = build_dataset(df, use_spatial=args.use_spatial)

    print("Label encoder:", enc)
    print(f"Built {len(graphs)} graphs")
    print("Graph 0:", graphs[0])
    print("Number of nodes:", graphs[0].num_nodes, "(expect 21)")
    print("x shape:", tuple(graphs[0].x.shape), "(expect (21, 6), or (21, 12) with --use-spatial)")
    print("x (nodes x features):\n", graphs[0].x)
    print("edge_index:\n", graphs[0].edge_index)
    print("Number of edges (directed, i.e. 2x undirected):", graphs[0].edge_index.shape[1], "(expect 62)")
    print("y:", graphs[0].y.item(), "temp:", graphs[0].temp.item(), "eov_bin:", graphs[0].eov_bin.item())
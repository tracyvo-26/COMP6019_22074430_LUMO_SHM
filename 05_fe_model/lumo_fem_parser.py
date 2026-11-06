"""
Reads the finite element model of the tower (Abaqus .inp file) and returns its nodes and connections.

Needs: lumo_fem_healthy.inp (same folder).
"""
from __future__ import annotations
import re
import numpy as np
import os
 
 
class FemGraph:
    def __init__(self, node_ids, coords, edges, level_nodes=None):
        self.node_ids = node_ids            # list[str], global node ids, e.g. "Segment-1.30"
        self.coords = np.asarray(coords, dtype=float)   # (N, 3) float array, real assembled coords
        self.edges = np.asarray(edges, dtype=int)        # (E, 2) int array, indices into node_ids
        self.id_to_index = {nid: i for i, nid in enumerate(node_ids)}
        self.level_nodes = level_nodes or {}   # e.g. {"ML1": "Segment-3.30", ...}
 
    @property
    def num_nodes(self):
        """Number of nodes in the graph."""
        return len(self.node_ids)
 
    @property
    def num_edges(self):
        """Number of edges in the graph."""
        return len(self.edges)
 
    def summary(self):
        """Short description of the graph and its bounding box."""
        return (f"FemGraph: {self.num_nodes} nodes, {self.num_edges} edges, "
                f"bbox x[{self.coords[:,0].min():.3f},{self.coords[:,0].max():.3f}] "
                f"y[{self.coords[:,1].min():.3f},{self.coords[:,1].max():.3f}] "
                f"z[{self.coords[:,2].min():.3f},{self.coords[:,2].max():.3f}]")
 
 
def _rotation_matrix(axis_dir, angle_deg):
    """Rotation matrix for an angle about an axis (Rodrigues' formula)."""
    axis = axis_dir / np.linalg.norm(axis_dir)
    theta = np.radians(angle_deg)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    I = np.eye(3)
    return I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
 
 
def _split_csv_line(line):
    """Split a comma-separated line of the .inp file and strip the entries."""
    return [p.strip() for p in line.split(",")]
 
 
def parse_inp(path: str) -> FemGraph:
    """Parse the Abaqus .inp file into a graph of nodes, beam elements and measurement levels."""
    part_node_ids, part_coords = [], []
    part_edges_local = []  # (local_id_a, local_id_b)
 
    section = None
    element_type = None
    with open(path, "r", errors="ignore") as f:
        part_lines = f.readlines()
 
    i = 0
    in_part = False
    while i < len(part_lines):
        raw = part_lines[i]
        line = raw.strip()
        i += 1
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            upper = line.upper()
            if upper.startswith("*PART"):
                in_part = True
                section = None
                continue
            if upper.startswith("*END PART"):
                in_part = False
                section = None
                continue
            if not in_part:
                section = None
                continue
            if upper.startswith("*NODE"):
                section = "NODE"
            elif upper.startswith("*ELEMENT"):
                m = re.search(r"type\s*=\s*(\w+)", line, re.IGNORECASE)
                element_type = m.group(1).upper() if m else None
                section = "ELEMENT"
            else:
                section = None
            continue
 
        parts = _split_csv_line(line)
        if section == "NODE":
            try:
                nid = int(parts[0])
                x, y, z = (float(v) for v in parts[1:4])
                part_node_ids.append(nid)
                part_coords.append((x, y, z))
            except (ValueError, IndexError):
                continue
        elif section == "ELEMENT" and element_type == "B31":
            try:
                conn = [int(p) for p in parts[1:] if p != ""]
                for a in range(len(conn)):
                    for b in range(a + 1, len(conn)):
                        part_edges_local.append((conn[a], conn[b]))
            except (ValueError, IndexError):
                continue
        # other element types (MASS, etc.) intentionally skipped for edges --
        # they only ever list one node per element anyway, so the original
        # parser's pairing loop was already a no-op for them.
 
    if not part_node_ids:
        raise ValueError(
            f"No nodes parsed from the *Part block of {path}"
        )
 
    part_id_to_coord = {nid: np.array(c) for nid, c in zip(part_node_ids, part_coords)}
 
    instances = {}   # name -> {"translation": (3,), "rot": (pa, pb, angle) or None}
    level_nodes = {}  # "ML1" -> "Segment-3.30"
    nset_members = {}  # nset_name -> list of "Instance.localid" (only assembly-level, instance-qualified ones)
    surface_to_nset = {}  # surface name -> nset name
    tie_pairs = []  # (surface_a, surface_b)
 
    in_assembly = False
    cur_instance = None
    cur_nset_name = None
    cur_nset_instance = None
    expect_translation = False
    expect_rotation = False
    j = 0
    while j < len(part_lines):
        raw = part_lines[j]
        line = raw.strip()
        j += 1
        if not line or line.startswith("**"):
            continue
 
        if line.startswith("*"):
            upper = line.upper()
            if upper.startswith("*ASSEMBLY"):
                in_assembly = True
                continue
            if upper.startswith("*END ASSEMBLY"):
                in_assembly = False
                continue
            if not in_assembly:
                continue
            if upper.startswith("*INSTANCE"):
                m = re.search(r"name\s*=\s*([\w\-]+)", line, re.IGNORECASE)
                cur_instance = m.group(1)
                instances[cur_instance] = {"translation": None, "rot": None}
                expect_translation = True
                expect_rotation = False
                cur_nset_name = None
                continue
            if upper.startswith("*END INSTANCE"):
                cur_instance = None
                expect_translation = False
                expect_rotation = False
                continue
            if upper.startswith("*NSET"):
                m_name = re.search(r"nset\s*=\s*([\w\-]+)", line, re.IGNORECASE)
                m_inst = re.search(r"instance\s*=\s*([\w\-]+)", line, re.IGNORECASE)
                cur_nset_name = m_name.group(1) if m_name else None
                cur_nset_instance = m_inst.group(1) if m_inst else None
                if cur_nset_name and cur_nset_instance:
                    nset_members.setdefault(cur_nset_name, [])
                continue
            if upper.startswith("*SURFACE"):
                m_name = re.search(r"name\s*=\s*([\w\-]+)", line, re.IGNORECASE)
                cur_surface_name = m_name.group(1) if m_name else None
                if j < len(part_lines):
                    body = part_lines[j].strip()
                    ref = _split_csv_line(body)[0]
                    surface_to_nset[cur_surface_name] = ref
                cur_nset_name = None
                continue
            if upper.startswith("*TIE"):
                if j < len(part_lines):
                    body = part_lines[j].strip()
                    a, b = _split_csv_line(body)[:2]
                    tie_pairs.append((a, b))
                cur_nset_name = None
                continue
            # any other keyword line (e.g. *Surface, type=... continuation
            # markers) ends whichever nset we were reading
            cur_nset_name = None
            expect_translation = False
            expect_rotation = False
            continue
 
        if cur_instance is not None and expect_translation:
            vals = [float(v) for v in _split_csv_line(line) if v != ""]
            instances[cur_instance]["translation"] = np.array(vals[:3])
            expect_translation = False
            expect_rotation = True
            continue
        if cur_instance is not None and expect_rotation:
            vals = [float(v) for v in _split_csv_line(line) if v != ""]
            if len(vals) >= 7:
                pa, pb, angle = np.array(vals[0:3]), np.array(vals[3:6]), vals[6]
                instances[cur_instance]["rot"] = (pa, pb, angle)
            expect_rotation = False
            continue
        if cur_nset_name and cur_nset_instance:
            for tok in _split_csv_line(line):
                if tok == "":
                    continue
                try:
                    local_id = int(tok)
                except ValueError:
                    continue
                gid = f"{cur_nset_instance}.{local_id}"
                nset_members[cur_nset_name].append(gid)
                if cur_nset_name.startswith("ML"):
                    level_nodes[cur_nset_name] = gid
 
    global_ids, global_coords = [], []
    for inst_name, xform in instances.items():
        translation = xform["translation"] if xform["translation"] is not None else np.zeros(3)
        rot = xform["rot"]
        for local_id, local_coord in part_id_to_coord.items():
            p = local_coord + translation
            if rot is not None:
                pa, pb, angle = rot
                axis_dir = pb - pa
                R = _rotation_matrix(axis_dir, angle)
                p = R @ (p - pa) + pa
            global_ids.append(f"{inst_name}.{local_id}")
            global_coords.append(p)
 
    id_to_index = {gid: idx for idx, gid in enumerate(global_ids)}
 
    edge_set = set()
    for inst_name in instances:
        for a, b in part_edges_local:
            ga, gb = f"{inst_name}.{a}", f"{inst_name}.{b}"
            if ga in id_to_index and gb in id_to_index:
                ia, ib = id_to_index[ga], id_to_index[gb]
                edge_set.add(tuple(sorted((ia, ib))))
 
    parent = list(range(len(global_ids)))
 
    def find(x):
        """Find the representative of a node in the union-find structure."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
 
    def union(x, y):
        """Merge the groups of two tied nodes."""
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx
 
    for surf_a, surf_b in tie_pairs:
        nset_a = surface_to_nset.get(surf_a, surf_a)
        nset_b = surface_to_nset.get(surf_b, surf_b)
        members_a = nset_members.get(nset_a, [])
        members_b = nset_members.get(nset_b, [])
        for ga, gb in zip(members_a, members_b):
            if ga in id_to_index and gb in id_to_index:
                union(id_to_index[ga], id_to_index[gb])
 
    # collapse to canonical representative per merged group
    group_to_canon = {}
    canon_index = {}
    final_ids, final_coords = [], []
    for idx, gid in enumerate(global_ids):
        root = find(idx)
        if root not in group_to_canon:
            group_to_canon[root] = len(final_ids)
            final_ids.append(gid)  # keep first-seen id as the label
            final_coords.append(global_coords[idx])
        canon_index[idx] = group_to_canon[root]
 
    final_edges = set()
    for ia, ib in edge_set:
        fa, fb = canon_index[ia], canon_index[ib]
        if fa != fb:
            final_edges.add(tuple(sorted((fa, fb))))
 
    level_nodes_final = {
        lvl: final_ids[canon_index[id_to_index[gid]]]
        for lvl, gid in level_nodes.items() if gid in id_to_index
    }
 
    return FemGraph(final_ids, final_coords, sorted(final_edges), level_nodes_final)
 
 
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lumo_fem_healthy.inp"))
    g = parse_inp(path)
    print(g.summary())
    print("Measurement-level node mapping:")
    for lvl in sorted(g.level_nodes):
        idx = g.id_to_index[g.level_nodes[lvl]]
        print(f"  {lvl}: {g.level_nodes[lvl]}  coords={g.coords[idx]}")
 
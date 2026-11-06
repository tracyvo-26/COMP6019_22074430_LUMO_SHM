
"""
Ranks the 18 accelerometer positions with three classical methods (Effective Independence, Modal Kinetic Energy and information entropy).

Needs: ssi_mode_shapes.csv (from 02_modal_identification/SSI-COV); node masses taken from the finite element model (05_fe_model).
"""
 
import datetime
import os
import sys
 
import numpy as np
import pandas as pd
 
SHAPE_CSV_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "02_modal_identification", "SSI-COV", "output_batch", "blockrows20_final", "ssi_mode_shapes.csv"))   # or fdd_mode_shapes.csv
METHOD_LABEL = "SSI-COV"
REFERENCE_GROUP = "Healthy"
OUT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__))))
if len(sys.argv) > 1:
    SHAPE_CSV_PATH = sys.argv[1]
if len(sys.argv) > 2:
    OUT_DIR = sys.argv[2]
OUT_CSV = f"{OUT_DIR}/classical_osp_rankings.csv"
OUT_SUMMARY = f"{OUT_DIR}/classical_osp_dataset_summary.csv"
 
 
def session_shape_matrix(df, mode_label):
    """Pivot the long-format shape data of one mode into a session by channel matrix."""
    sub = df[df["mode_label"] == mode_label]
    pivot = sub.pivot_table(index="session", columns="channel", values="amplitude")
    meta = sub.groupby("session")[["damage_label"]].first()
    return pivot, meta
 
 
def build_reference_shape(pivot, meta, reference_group=REFERENCE_GROUP):
    """Build the reference shape of one mode by sign-aligning and averaging the reference sessions."""
    ref_sessions = meta[meta["damage_label"].str.lower() == reference_group.lower()].index
    ref_sessions = [s for s in ref_sessions if s in pivot.index]
    if not ref_sessions:
        return None
    anchor = pivot.loc[ref_sessions[0]].values
    aligned = [anchor]
    for s in ref_sessions[1:]:
        vec = pivot.loc[s].values
        if np.dot(vec, anchor) < 0:
            vec = -vec
        aligned.append(vec)
    return np.nanmean(aligned, axis=0)
 
 
def describe_input(df):
    """Print and return the path, date, session count and sessions per class of the input file."""
    sessions = df.drop_duplicates("session")
    n = len(sessions)
    try:
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(SHAPE_CSV_PATH)).strftime("%Y-%m-%d %H:%M")
    except OSError:
        mtime = "unknown"
    print(f"Input: {SHAPE_CSV_PATH}")
    print(f"       last modified {mtime}; {len(df)} rows")
    print(f"Sessions in the file: {n}")
    by_class = sessions["damage_label"].value_counts().sort_index()
    for label, c in by_class.items():
        print(f"  {label:<10} {c}")
    print(f"Channels in the file: {df['channel'].nunique()} ({', '.join(sorted(df['channel'].unique())[:3])}, ...)")
    print(f"Modes in the file: {', '.join(sorted(df['mode_label'].unique()))}\n")
    return n, by_class
 
 
def build_reference_matrix(df):
    """Build the mode shape matrix Phi (channels by modes) from the Healthy reference shapes."""
    modes = df["mode_label"].unique()
    channels = None
    columns = {}
    stats = []
    for mode_label in modes:
        pivot, meta = session_shape_matrix(df, mode_label)
        n_all = len(pivot)
        n_ref = int((meta["damage_label"].str.lower() == REFERENCE_GROUP.lower()).sum())
        ref = build_reference_shape(pivot, meta)
        used = ref is not None and not np.isnan(ref).any()
        stats.append(dict(mode=mode_label, sessions_with_mode=n_all, reference_sessions=n_ref, used_in_Phi=used))
        if not used:
            print(f"skip {mode_label}: no clean Healthy reference shape available "
                  f"({n_all} sessions with this mode, {n_ref} {REFERENCE_GROUP} sessions)")
            continue
        if channels is None:
            channels = list(pivot.columns)
        else:
            # align column order in case pivot tables differ in column order
            ref = pd.Series(ref, index=pivot.columns).reindex(channels).values
        columns[mode_label] = ref
 
    if not columns:
        raise ValueError("No modes had a usable reference shape.")
 
    mode_order = list(columns.keys())
    Phi = np.column_stack([columns[m] for m in mode_order])
    return Phi, channels, mode_order, stats
 
 
# 1. Effective Independence (EFI): backward elimination.
# ED_ROUND_DECIMALS rounds the values before the minimum is picked, so ties break the same way on every machine.
ED_ROUND_DECIMALS = 8
 
 
def run_efi(Phi, channels, k):
    """Effective Independence ranking by backward elimination of the least informative location."""
    Phi = Phi.copy()
    remaining = list(range(len(channels)))
    removal_order = []
 
    while len(remaining) > k:
        sub_Phi = Phi[remaining, :]
        # Fisher information matrix and its inverse (guard against singularity
        # as rows get removed and modes start to become harder to distinguish)
        Q = sub_Phi.T @ sub_Phi
        Q_inv = np.linalg.pinv(Q)
        # effective independence distribution: diagonal of Phi (Phi^T Phi)^-1 Phi^T
        G = sub_Phi @ Q_inv @ sub_Phi.T
        ed = np.diag(G)
        ed_rounded = np.round(ed, ED_ROUND_DECIMALS)
        worst_local_idx = int(np.argmin(ed_rounded))
        worst_global_idx = remaining[worst_local_idx]
        removal_order.append((channels[worst_global_idx], float(ed[worst_local_idx])))
        remaining.pop(worst_local_idx)
 
    kept = [channels[i] for i in remaining]
    return kept, removal_order
 
 
# 2. Modal Kinetic Energy (MKE): MKE_i = sum over modes of (mass_i * shape_i^2).
# Node masses come from the FE model; LEVEL_MASS_KG repeats its three node masses across the nine levels.
LEVEL_MASS_KG = {1: 0.5052, 4: 0.5052, 7: 0.5052,
                 2: 0.5199, 5: 0.5199, 8: 0.5199,
                 3: 0.5199, 6: 0.5199, 9: 0.5199}
 
 
def fe_mass_for_channels(channels):
    """Return the lumped mass from the FE model for each channel."""
    mass = np.ones(len(channels))
    for i, ch in enumerate(channels):
        m = pd.Series([ch]).str.extract(r"accel(\d+)")[0]
        try:
            level = int(m.iloc[0])
            mass[i] = LEVEL_MASS_KG.get(level, 1.0)
        except (TypeError, ValueError):
            pass  # non-accelerometer channel (e.g. strain) -- stays at 1.0
    return mass
 
 
def run_mke(Phi, channels, k, mass=None):
    """Modal kinetic energy ranking of the locations."""
    n = len(channels)
    if mass is None:
        mass = fe_mass_for_channels(channels)
    mke = (mass[:, None] * Phi ** 2).sum(axis=1)
    order = np.argsort(-mke)  # descending
    ranked = [(channels[i], float(mke[i])) for i in order]
    kept = [channels[i] for i in order[:k]]
    return kept, ranked
 
 
# 3. Information-entropy OSP: greedy forward selection maximising the log-determinant of the Fisher information matrix.
def _pseudo_logdet(Q, tol=1e-9):
    """Log of the product of the nonzero eigenvalues of Q."""
    eigvals = np.linalg.eigvalsh(Q)
    eigvals = eigvals[eigvals > tol * max(eigvals.max(), 1e-12)]
    if len(eigvals) == 0:
        return -np.inf
    return float(np.sum(np.log(eigvals)))
 
 
def run_entropy_osp(Phi, channels, k):
    """Greedy entropy-based ranking that records each channel's marginal gain in the log-determinant."""
    n, m = Phi.shape
    selected = []
    remaining = list(range(n))
    selection_order = []
 
    # seed with the location of largest norm
    first = int(np.argmax(np.linalg.norm(Phi, axis=1)))
    selected.append(first)
    remaining.remove(first)
    Q_selected = np.outer(Phi[first, :], Phi[first, :])
    prev_logdet = _pseudo_logdet(Q_selected)
    selection_order.append((channels[first], float(prev_logdet)))
 
    while len(selected) < k and remaining:
        best_gain, best_idx, best_logdet, best_Q = -np.inf, None, None, None
        for idx in remaining:
            phi_i = Phi[idx, :]
            Q_candidate = Q_selected + np.outer(phi_i, phi_i)
            logdet = _pseudo_logdet(Q_candidate)
            marginal = logdet - prev_logdet
            if marginal > best_gain:
                best_gain, best_idx, best_logdet, best_Q = marginal, idx, logdet, Q_candidate
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
        selection_order.append((channels[best_idx], float(best_gain)))
        prev_logdet = best_logdet
        Q_selected = best_Q
 
    kept = [channels[i] for i in selected]
    return kept, selection_order
 
 
def main():
    """Compute the three rankings and save the ranking table and the dataset summary."""
    df = pd.read_csv(SHAPE_CSV_PATH)
    n_sessions, by_class = describe_input(df)
    Phi, channels, mode_order, mode_stats = build_reference_matrix(df)
 
    print("Sessions per mode (reference shape = average of the Healthy sessions that have the mode):")
    for s in mode_stats:
        flag = "" if s["used_in_Phi"] else "   <-- NOT used in Phi"
        print(f"  {s['mode']:<6} sessions with the mode {s['sessions_with_mode']:>4} | "
              f"{REFERENCE_GROUP} sessions in the reference {s['reference_sessions']:>3}{flag}")
    print(f"\nReference mode shape matrix built: {Phi.shape[0]} candidate locations "
          f"x {Phi.shape[1]} modes ({', '.join(mode_order)}), from {METHOD_LABEL}.\n")
 
    # Each method's full ranking is computed once and saved to OUT_CSV below.
    _, mke_full_ranked = run_mke(Phi, channels, len(channels))
    mke_full_rank = [c for c, _ in mke_full_ranked]  # already sorted strongest-first
 
    efi_survivor, efi_removal_order = run_efi(Phi, channels, 1)
    # removal_order is weakest-removed-first, so it is reversed. The survivor is placed at rank 1.
    efi_full_rank = efi_survivor + [c for c, _ in reversed(efi_removal_order)]
 
    _, entropy_full_selection = run_entropy_osp(Phi, channels, len(channels))
    entropy_full_rank = [c for c, _ in entropy_full_selection]  # already strongest-first, since
    # this method selects greedily (first picked = most informative)
 
    # EFI score: the effective independence value of each channel when it was removed
    efi_score_map = dict(efi_removal_order)
    # The survivor has no ED value of its own, so it gets the value of the strongest removed channel.
    efi_survivor_score = max(efi_score_map.values()) if efi_score_map else 1.0
    efi_score_map[efi_survivor[0]] = efi_survivor_score
 
    # Entropy score: the pseudo-log-determinant gain when each channel was picked
    entropy_score_map = dict(entropy_full_selection)
 
    # save the full ranking detail (all 3 methods) to CSV for the report
    out = pd.DataFrame({
        "channel": channels,
        "mke_score": [dict(mke_full_ranked)[c] for c in channels],
        "efi_score": [efi_score_map.get(c) for c in channels],
        "entropy_score": [entropy_score_map.get(c) for c in channels],
    })
    out["mke_rank"] = out["mke_score"].rank(ascending=False).astype(int)
 
    # EFI rank: dense ranking on the rounded score (ties share a rank)
    efi_score_rounded = out["efi_score"].round(ED_ROUND_DECIMALS)
    out["efi_rank"] = efi_score_rounded.rank(method="dense", ascending=False).astype(int)
 
    out["entropy_rank"] = [entropy_full_rank.index(c) + 1 for c in channels]
    out = out.sort_values("mke_rank")
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved full ranking detail -> {OUT_CSV}")
 
    # dataset summary next to the rankings, so the file records which data it came from
    summary = pd.DataFrame(mode_stats)
    summary.insert(0, "input_file", SHAPE_CSV_PATH)
    summary.insert(1, "sessions_in_file", n_sessions)
    for label, c in by_class.items():
        summary[f"sessions_{label}"] = c
    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"Saved dataset summary    -> {OUT_SUMMARY}")
 
    tied = out[out.duplicated("efi_rank", keep=False)].sort_values("efi_rank")
    if not tied.empty:
        print("\nEFI ties detected -- channels sharing a rank number below are "
              "mathematically indistinguishable at this precision, NOT a real "
              "strict order:")
        for rank, group in tied.groupby("efi_rank"):
            print(f"  rank {rank}: {', '.join(group['channel'])}")
 
 
if __name__ == "__main__":
    main()
 
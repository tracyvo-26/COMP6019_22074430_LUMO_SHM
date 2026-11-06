"""
Turns classical_osp_rankings.csv into a summary table and plots of the three rankings.

Needs: classical_osp_rankings.csv (from classical_osp_baselines.py).
"""
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from scipy.stats import spearmanr
import os
 
RANKINGS_CSV = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "classical_osp_rankings.csv"))
OUT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__))))
 

N_LOCATIONS = 18
METHODS = [("efi_rank", "EFI"), ("mke_rank", "MKE"), ("entropy_rank", "Entropy")]
CMAP = plt.get_cmap("RdYlGn")  # red = rank 1 (most important) -> green = rank 18
NORM = Normalize(vmin=1, vmax=N_LOCATIONS)
 
# heights (m) of each measurement level, from the FE model
LEVEL_HEIGHTS_M = {1: 8.95, 2: 8.00, 3: 7.00, 4: 5.95, 5: 5.00,
                   6: 4.00, 7: 2.95, 8: 2.00, 9: 1.00}
 
 
def level_of(channel):
    """Level number (1 to 9) of a channel name."""
    return int(channel[5:7])
 
 
def direction_of(channel):
    """Direction (x or y) of a channel name."""
    return channel[-1]  # 'x' or 'y'
 
 
def build_summary_table(df):
    # EFI can report ties (several channels with the same rank), so every
    # channel that shares a rank goes into one cell, separated by "/".
    """Build the markdown table of the three rankings and their rank correlations."""
    lines = [f"# Classical OSP baseline -- full ranking, all {N_LOCATIONS} locations\n",
             "| Rank | EFI | MKE | Entropy |", "|---|---|---|---|"]
    all_ranks = sorted(set(df["efi_rank"]) | set(df["mke_rank"]) | set(df["entropy_rank"]))
    for rank in all_ranks:
        row = [str(rank)]
        for col, _ in METHODS:
            chs = df.loc[df[col] == rank, "channel"].tolist()
            row.append(" / ".join(chs) if chs else "-")
        lines.append("| " + " | ".join(row) + " |")
 
    # agreement between methods over the full ranking (Spearman rank correlation)
    lines.append("\n## Agreement between methods (Spearman rank correlation, all 18 locations)\n")
    lines.append("| | EFI | MKE | Entropy |")
    lines.append("|---|---|---|---|")
    ranks = {name: df.set_index("channel")[col].reindex(df["channel"]) for col, name in METHODS}
    for _, name_a in METHODS:
        row = [name_a]
        for _, name_b in METHODS:
            if name_a == name_b:
                row.append("--")
            else:
                rho, _ = spearmanr(ranks[name_a], ranks[name_b])
                row.append(f"{rho:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
 
 
def plot_tower_diagram(df, out_path):
    # Dots are labelled with the sensor level number (1-9). The colour shows the
    # rank. The y position is the real height, so level 1 is at the top.
    """Plot the rank of every sensor position on the tower for each method."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 7), sharey=True)
    rank_lookup = {col: df.set_index("channel")[col] for col, _ in METHODS}
    for ax, (col, name) in zip(axes, METHODS):
        for level, height in LEVEL_HEIGHTS_M.items():
            # same-level edge connecting the x and y sensor of a level
            ax.plot([0, 1], [height, height], color="lightgray", zorder=1, linewidth=1)
            for direction, xpos in (("x", 0), ("y", 1)):
                ch = f"accel{level:02d}{direction}"
                rank = int(rank_lookup[col][ch])
                color = CMAP(NORM(rank))
                ax.scatter(xpos, height, s=260, color=color,
                           edgecolor="black", zorder=3, linewidth=1)
                ax.annotate(str(level), (xpos, height), ha="center", va="center",
                            fontsize=8, fontweight="bold", color="black", zorder=4)
        # x-chain and y-chain edges (consecutive levels, same direction)
        ax.plot([0, 0], [1, 8.95], color="gray", zorder=1, linewidth=1)
        ax.plot([1, 1], [1, 8.95], color="gray", zorder=1, linewidth=1)
        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["x", "y"])
        ax.set_yticks([])  # no height ticks, the dots carry the level number
        ax.set_title(name)
    axes[0].set_ylabel("Tower height (top to bottom) -- number in each dot is the sensor level (1-9)")
    sm = cm.ScalarMappable(cmap=CMAP, norm=NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.05, pad=0.08)
    cbar.set_label("Rank (1 = most important, 18 = least important)")
    fig.suptitle("Classical OSP baselines -- full rank of all 18 locations on the real tower shape")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
 
 
def plot_score_barchart(df, out_path):
    # uses each method's score rather than its rank, so the size of the differences is visible
    """Plot the score of every sensor position for each method."""
    score_cols = [("efi_score", "efi_rank", "EFI"),
                  ("mke_score", "mke_rank", "MKE"),
                  ("entropy_score", "entropy_rank", "Entropy")]
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=False)
    fig.subplots_adjust(right=0.85, hspace=0.9, top=0.93, bottom=0.06)
    for ax, (score_col, rank_col, name) in zip(axes, score_cols):
        ordered = df.sort_values(rank_col)
        colors = [CMAP(NORM(r)) for r in ordered[rank_col]]
        ax.bar(ordered["channel"], ordered[score_col], color=colors)
        ax.set_ylabel("Score (real magnitude)")
        ax.set_title(name)
        ax.tick_params(axis="x", rotation=90)
    sm = cm.ScalarMappable(cmap=CMAP, norm=NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.03, pad=0.02)
    cbar.set_label("Rank (1 = most important, 18 = least important)")
    fig.suptitle("Classical OSP baselines -- score by location per method, all 18 ranked")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
 
 
def main():
    """Create the summary table and the two figures from the rankings file."""
    df = pd.read_csv(RANKINGS_CSV)
 
    table_md = build_summary_table(df)
    table_path = f"{OUT_DIR}/classical_osp_summary_table.md"
    with open(table_path, "w") as f:
        f.write(table_md)
    print(table_md)
    print(f"\nSaved -> {table_path}")
 
    diagram_path = f"{OUT_DIR}/classical_osp_tower_diagram.png"
    plot_tower_diagram(df, diagram_path)
    print(f"Saved -> {diagram_path}")
 
    barchart_path = f"{OUT_DIR}/classical_osp_score_barchart.png"
    plot_score_barchart(df, barchart_path)
    print(f"Saved -> {barchart_path}")
 
 
if __name__ == "__main__":
    main()
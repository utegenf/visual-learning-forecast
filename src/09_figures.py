"""Redraw Figures 1 and 2 with stage annotations (presentation only: no new statistics).

Shaded backgrounds mark task phases: gratings training (TRAINING_0-2), image training (TRAINING_3-5 and OPHYS_1,
familiar images A), novel images (OPHYS_4) and reward omission (OPHYS_6). d' is not strictly comparable across phases.
Usage: python src/09_figures.py
"""

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

PHASE = {"TRAINING_0": "gratings", "TRAINING_1": "gratings", "TRAINING_2": "gratings", "TRAINING_3": "images A",
         "TRAINING_4": "images A", "TRAINING_5": "images A", "OPHYS_1": "images A", "OPHYS_4": "novel images B",
         "OPHYS_6": "reward omission"}
COLOR = {"gratings": "#eeeeee", "images A": "#dfeaf5", "novel images B": "#d4ecd4", "reward omission": "#f7d9d9"}


def shade(ax, g):
    g = g.assign(phase=g.stage.map(PHASE))
    for _, r in g.dropna(subset=["phase"]).iterrows():
        ax.axvspan(r.session_number - 0.5, r.session_number + 0.5, color=COLOR[r.phase], lw=0, zorder=0)


def main():
    beh = pd.read_csv("results/tables/behavior_sessions.csv", dtype={"mouse": str})
    nf = pd.read_csv("results/tables/neural_features_v2.csv", dtype={"mouse": str})
    s = nf.groupby(["mouse", "session", "session_number", "stage"], as_index=False)[["decode_acc", "change_index"]].mean()
    legend = [Patch(color=c, label=k) for k, c in COLOR.items()]

    fig, axes = plt.subplots(2, 3, figsize=(11, 5.8), sharey=True)
    for ax, (m, g) in zip(axes.flat, beh.groupby("mouse")):
        shade(ax, g)
        ax.plot(g.session_number, g.dprime, "-o", color="0.2", ms=3, lw=1)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_title(f"mouse {m}", fontsize=9)
        ax.set_xlabel("session")
    axes[0, 0].set_ylabel("d'")
    axes[1, 0].set_ylabel("d'")
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Fig. 1. Task performance across sessions. Shading marks task phases, and d' is not strictly comparable across phases.",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig("results/figures/fig1_learning_curves.png", dpi=200)

    fig, axes = plt.subplots(2, 6, figsize=(15, 4.8), sharey="row")
    for k, (m, g) in enumerate(s.groupby("mouse")):
        g = g.dropna(subset=["decode_acc"])
        for row, (col, lab) in enumerate((("decode_acc", "change vs non-change\ndecoding accuracy"),
                                          ("change_index", "change-evoked\nresponse index (ΔF/F)"))):
            ax = axes[row, k]
            shade(ax, g)
            ax.plot(g.session_number, g[col], "-o", color="0.2", ms=3, lw=1)
            if row == 0:
                ax.axhline(0.5, color="0.6", lw=0.8, ls="--")
                ax.set_title(f"mouse {m}", fontsize=9)
            if k == 0:
                ax.set_ylabel(lab, fontsize=8)
            ax.set_xlabel("session" if row == 1 else "")
    fig.legend(handles=legend[1:], loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Fig. 2. Session-level population features (whole recorded population of each session, not tracked neurons).",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig("results/figures/fig2_neural_across_sessions.png", dpi=200)


if __name__ == "__main__":
    main()

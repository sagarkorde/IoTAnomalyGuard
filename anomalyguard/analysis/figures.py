"""Paper figures (300 dpi PNG, IEEE single-column width).

Palette: the validated categorical order of the dataviz reference palette, assigned to
entities in a fixed order; every multi-series figure also uses distinct markers/line styles
so identity never relies on colour alone (print / colour-vision deficiency).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .. import config as C

R = C.RESULTS_DIR
OUT = R / "figures"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL_W = 3.5  # inches (IEEE column)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.5,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.16, 2.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")

    def box(x, y, w, h, title, sub, accent=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8",
                                    fc="#eef4fc" if accent else "#f6f6f4",
                                    ec=SLOTS[0] if accent else "#9a9893", lw=0.8))
        ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=6.2, weight="bold",
                color=INK, linespacing=1.1)
        ax.text(x + w / 2, y + h / 2 - (1.6 if "\n" in title else 0.6), sub, ha="center", va="center",
                fontsize=5.2, color=INK2, linespacing=1.3)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                     lw=0.7, color=INK2, shrinkA=0, shrinkB=0))

    box(0.5, 9, 12.5, 16, "Capture", "X-IoMT v1.1\n1.58 M packets:\nheaders, device\nand vital-sign\ntelemetry")
    box(16, 9, 13.5, 16, "Shortcut-free\nviews", "absent values\ncanonicalised;\nno IP, MAC, client\nID or timestamp;\nchronological split", True)
    box(33, 19, 15, 13, "Network view", "header behaviour\n+ causal per-source\ncontext\n+ port novelty", True)
    box(33, 2, 15, 13, "Telemetry view", "device + vital-sign\nfields, scored only\nwhen present", True)
    box(51.5, 19, 13, 13, "Detector", "Isolation Forest,\nψ from benign\nservice mix\n(label-free)", True)
    box(51.5, 2, 13, 13, "Detector", "same family,\nown benign\ncalibration", True)
    box(68, 9, 13.5, 16, "Fusion", "conformal p-values\nBonferroni min-p\nthreshold at 1 %\nof benign\nvalidation", True)
    box(85, 19, 14.5, 13, "Context", "session association:\nk alerts in 10 s,\nk set on benign\ntraffic", True)
    box(85, 2, 14.5, 13, "Triage", "route by view to\nnetwork-security or\ndevice/clinical queue;\nincidents")
    arrow(13.2, 17, 15.8, 17)
    arrow(29.7, 19, 32.8, 25); arrow(29.7, 15, 32.8, 9)
    arrow(48.2, 25.5, 51.3, 25.5); arrow(48.2, 8.5, 51.3, 8.5)
    arrow(64.7, 25, 67.8, 20); arrow(64.7, 9, 67.8, 14)
    arrow(81.7, 20, 84.8, 25); arrow(81.7, 14, 84.8, 9)
    fig.savefig(OUT / "F1_architecture.png")
    plt.close(fig)


def fig_architecture_compact():
    """Single-column version of the pipeline figure (used in the 6-page paper)."""
    fig, ax = plt.subplots(figsize=(3.45, 1.9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 55); ax.axis("off")

    def box(x, y, w, h, title, sub, accent=True):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc="#eef4fc" if accent else "#f6f6f4",
                                    ec=SLOTS[0] if accent else "#9a9893", lw=0.7))
        ax.text(x + w / 2, y + h - 1.8, title, ha="center", va="top", fontsize=6.0, weight="bold", color=INK,
                linespacing=1.1)
        ax.text(x + w / 2, y + (h - (9.5 if "\n" in title else 5.5)) / 2 + 0.4, sub, ha="center", va="center",
                fontsize=5.3, color=INK2, linespacing=1.2)

    def arrow(x0, y0, x1, y1, rad=0.0):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=5.5, lw=0.6,
                                     color=INK2, shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))

    box(0.8, 27, 18, 26, "Packet\nstream", "one stream\nper sensor", False)
    box(24, 41, 38, 12, "Network view", "headers + causal context\n→ IF with label-free ψ*")
    box(24, 27, 38, 12, "Telemetry view", "device and vital signs\n→ same detector family")
    box(67, 27, 32, 26, "Calibration\n+ fusion", "conformal p-values\n(benign data),\nBonferroni min-p,\n1 % threshold")
    box(24, 2, 38, 19, "Session association", "flag packets to a host\nwith ≥ k alerts in 10 s")
    box(67, 2, 32, 19, "Triage", "queues and\n60-s incidents", False)
    arrow(19.4, 44, 23.6, 47); arrow(19.4, 36, 23.6, 33)
    arrow(62.4, 47, 66.6, 44); arrow(62.4, 33, 66.6, 36)
    arrow(83, 26.6, 83, 21.4)
    arrow(66.6, 29, 62.4, 17, rad=-0.2)
    arrow(62.4, 9, 66.6, 9)
    fig.savefig(OUT / "F1_architecture_col.png")
    plt.close(fig)


def fig_audit():
    p = pd.concat([pd.read_csv(R / "e1_probes.csv"), pd.read_csv(R / "e1_protocols.csv")])
    b = pd.read_csv(R / "e2_benchmark.csv")
    ag = b[(b["detector"] == "AnomalyGuard-IF") & (b["variant"] == "fused+propagation")]["roc_auc"]
    items = [
        ("Probe: absence spelling only", p[p.setting == "probe:absence_spelling"]["roc_auc"], 1),
        ("Probe: host identity only", p[p.setting == "probe:host_identity"]["roc_auc"], 1),
        ("HGB, all columns, random split", p[p.setting == "naive:HGB:random-split"]["roc_auc"], 1),
        ("HGB, all columns, chronological", p[p.setting == "naive:HGB:chronological"]["roc_auc"], 1),
        ("HGB, behaviour-only, chronological", p[p.setting == "leakage-aware:HGB:chronological"]["roc_auc"], 0),
        ("IF ψ=256, all columns, random split", p[p.setting == "naive:IF-default:random-split"]["roc_auc"], 1),
        ("AnomalyGuard-IF, behaviour-only", ag, 0),
    ]
    fig, ax = plt.subplots(figsize=(COL_W, 2.05))
    y = np.arange(len(items))[::-1]
    for yi, (label, s, leaky) in zip(y, items):
        m, sd = float(s.mean()), float(s.std(ddof=1)) if len(s) > 1 else 0.0
        ax.barh(yi, m - 0.5, left=0.5, height=0.56, color=SLOTS[1] if leaky else SLOTS[0],
                hatch="////" if leaky else None, edgecolor="white", linewidth=0.0)
        ax.errorbar(m, yi, xerr=sd, color=INK, lw=0.7, capsize=1.5)
        ax.text(m + 0.006 + sd, yi, f"{m:.3f}", va="center", fontsize=6.3, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([i[0] for i in items])
    ax.set_xlim(0.5, 1.045); ax.set_xlabel("Test AUROC (mean ± s.d. over seeds)")
    ax.grid(axis="x"); ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=SLOTS[1], hatch="////", ec="white", label="uses leaky / non-behavioural fields"),
                       Patch(fc=SLOTS[0], label="behaviour-only (this work)")],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False, handlelength=1.4)
    fig.savefig(OUT / "F2_shortcut_audit.png")
    plt.close(fig)


def fig_psi():
    d = pd.read_csv(R / "e4_psi_sweep.csv")
    d = d[d["sweep"] == "psi"]
    g = d.groupby("psi")
    fig, ax = plt.subplots(figsize=(COL_W, 1.95))
    series = [("macro_recall", "Macro recall", SLOTS[0], "o", "-"),
              ("recall_network", "Network recall", SLOTS[1], "s", "--"),
              ("macro_auc", "Macro AUROC", SLOTS[2], "^", ":")]
    x = np.log2(g.size().index.to_numpy())
    for col, label, c, mk, ls in series:
        m, s = g[col].mean().to_numpy(), g[col].std(ddof=1).to_numpy()
        ax.plot(x, m, color=c, marker=mk, ms=3.6, lw=1.2, ls=ls, label=label)
        ax.fill_between(x, m - s, m + s, color=c, alpha=0.15, lw=0)
    rule = int(d["psi_rule"].mode().iloc[0])
    ax.axvline(np.log2(rule), color=INK2, lw=0.7, ls="-.")
    ax.text(np.log2(rule) - 0.1, 0.05, f"label-free\nrule ψ* = {rule}", fontsize=6.2, color=INK2, ha="right")
    ax.axvline(8, color=INK2, lw=0.5, ls=":")
    ax.text(8 + 0.1, 0.55, "default\nψ = 256", fontsize=6.2, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([f"{int(2 ** v)}" for v in x], rotation=0)
    ax.set_xlabel("Isolation Forest subsample size ψ (log scale)")
    ax.set_ylabel("Score at 1 % FPR budget"); ax.set_ylim(0, 1.02)
    ax.grid(axis="y"); ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False, handlelength=2.2,
              columnspacing=1.0)
    fig.savefig(OUT / "F3_psi_sweep.png")
    plt.close(fig)


def fig_contamination():
    d = pd.read_csv(R / "e5_contamination.csv")
    fig, ax = plt.subplots(figsize=(COL_W, 1.85))
    styles = {"IF-default": (SLOTS[1], "s", "--"), "AnomalyGuard-IF": (SLOTS[0], "o", "-"),
              "OCSVM": (SLOTS[2], "^", ":")}
    for det, sub in d.groupby("detector", sort=False):
        g = sub.groupby("rho")["macro_recall"]
        x = 100 * g.mean().index.to_numpy()
        m, s = g.mean().to_numpy(), g.std(ddof=1).to_numpy()
        c, mk, ls = styles[det]
        ax.plot(x, m, color=c, marker=mk, ms=3.6, lw=1.2, ls=ls, label=det)
        ax.fill_between(x, m - s, m + s, color=c, alpha=0.15, lw=0)
    ax.set_xscale("symlog", linthresh=0.1); ax.set_xlim(0, 10.5)
    ax.set_xticks([0, 0.1, 0.5, 1, 2, 5, 10]); ax.set_xticklabels(["0", "0.1", "0.5", "1", "2", "5", "10"])
    ax.set_xlabel("Unlabelled anomalies in the training pool ρ (%)")
    ax.set_ylabel("Macro recall at 1 % FPR"); ax.set_ylim(0, 1.0)
    ax.grid(axis="y"); ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, handlelength=2.4)
    fig.savefig(OUT / "F4_contamination.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_architecture, fig_architecture_compact, fig_audit, fig_psi, fig_contamination):
        try:
            f(); print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, e)


if __name__ == "__main__":
    main()

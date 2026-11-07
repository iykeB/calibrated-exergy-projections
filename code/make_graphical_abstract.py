"""
make_graphical_abstract.py

Graphical abstract for Energy (Elsevier): 5:2 ratio, at least 1328 x 531 pixels at 300 dpi,
Arial, reads left to right, no heading. Three steps on the real results:
(1) a published point projection (Utlu and Hepbasli 2006, Turkish transport 2020, data to 2000);
(2) driver projection errors of the other 32 countries, pooled by horizon (smoothed 90% band);
(3) the calibrated 90% interval with the observed 2020 value.
Writes ../figures/Graphical_abstract.{png,pdf,tif}.
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import framework as F
import projection_calibration as PC

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
RES = os.path.join(HERE, "..", "results")
BLUE, ORANGE, RED, INK, INK2, GRID = "#2a78d6", "#eb6834", "#e34948", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "Arial", "font.size": 15, "axes.edgecolor": INK2, "axes.linewidth": 1.0,
                     "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 13, "ytick.labelsize": 13,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic"})
UH_EX = 1385.73


def main():
    A = F.accounts()
    _, gdp, pop = PC.load()
    rec = [r for r in F.collect(A, gdp, pop) if "ex_driver" in r]
    E = {}
    for r in rec:
        if r["g"] != "TR":
            E.setdefault(r["h"], []).append(np.log(r["ex"] / r["ex_driver"]))
    Q = PC.smooth_quantiles(E, 0.9)
    hs = np.linspace(1, 24, 120)
    lo = np.array([Q(h)[0] for h in hs])
    hi = np.array([Q(h)[2] for h in hs])
    l20, _, h20 = Q(20)
    band = (UH_EX * np.exp(l20), UH_EX * np.exp(h20))
    i20 = list(PC.YEARS).index(2020)
    obs = float(A["TR"]["ex"][i20])
    R = json.load(open(os.path.join(RES, "framework.json")))["driver|persist"]
    cov = [100 * R[k]["all"]["cover"] for k in ("ex", "phi", "imp")]
    print("band", band, "observed", obs, "coverage", cov)

    fig = plt.figure(figsize=(13.28, 5.312))
    ax1 = fig.add_axes([0.065, 0.20, 0.20, 0.64])
    ax2 = fig.add_axes([0.395, 0.20, 0.24, 0.64])
    ax3 = fig.add_axes([0.765, 0.20, 0.20, 0.64])
    ylim = (0, 3000)
    for ax, title in ((ax1, "1  Published point projection"), (ax2, "2  Errors in 32 other countries"),
                      (ax3, "3  Calibrated 90% interval")):
        ax.text(-0.12, 1.08, title, transform=ax.transAxes, fontsize=16, fontweight="bold", color=INK, ha="left")
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)

    # 1: point value only
    ax1.plot([2020], [UH_EX], "*", color=ORANGE, ms=22, mec="white", mew=0.8)
    ax1.text(2020, UH_EX + 260, "1386 PJ", ha="center", fontsize=15, color=INK)
    ax1.set_xlim(2010, 2025)
    ax1.set_ylim(*ylim)
    ax1.set_xticks([2010, 2020])
    ax1.set_ylabel("Exergy input (PJ)", fontsize=14)
    ax1.text(0.04, 0.95, "Turkish transport, 2020,\nprojected from data to 2000", transform=ax1.transAxes,
             fontsize=13, color=INK2, va="top")

    # 2: pooled error band by horizon
    ax2.fill_between(hs, 100 * (np.exp(lo) - 1), 100 * (np.exp(hi) - 1), color=BLUE, alpha=0.25, lw=0)
    ax2.plot(hs, 100 * (np.exp(lo) - 1), color=BLUE, lw=2)
    ax2.plot(hs, 100 * (np.exp(hi) - 1), color=BLUE, lw=2)
    ax2.axhline(0, color=INK2, lw=1.0, ls="--")
    ax2.axvline(20, color=ORANGE, lw=1.6, ls=":")
    ax2.set_xlim(0, 25)
    ax2.set_xlabel("Forecast horizon (years)", fontsize=14)
    ax2.set_ylabel("Error (%)", fontsize=14)
    ax2.text(0.04, 0.95, "90% band of driver errors", transform=ax2.transAxes, fontsize=13,
             color=INK2, va="top")

    # 3: calibrated interval and observed value
    ax3.plot([2020, 2020], band, color=ORANGE, lw=5, solid_capstyle="butt")
    ax3.plot([2020], [UH_EX], "*", color=ORANGE, ms=22, mec="white", mew=0.8)
    ax3.plot([2021.2], [obs], "D", color=RED, ms=11, mfc="white", mew=2)
    ax3.text(2018.8, band[1], f"{band[1]:.0f}", ha="right", va="center", fontsize=13, color=INK2)
    ax3.text(2018.8, band[0], f"{band[0]:.0f}", ha="right", va="center", fontsize=13, color=INK2)
    ax3.text(2022.0, obs, f"observed\n{obs:.0f} PJ", ha="left", va="center", fontsize=13, color=INK)
    ax3.set_xlim(2010, 2029)
    ax3.set_ylim(*ylim)
    ax3.set_xticks([2010, 2020])
    ax3.set_ylabel("Exergy input (PJ)", fontsize=14)

    for x0, x1 in ((0.285, 0.330), (0.655, 0.700)):
        fig.patches.append(FancyArrowPatch((x0, 0.52), (x1, 0.52), transform=fig.transFigure,
                                           arrowstyle="-|>", mutation_scale=28, lw=2.5, color=INK2))
    fig.text(0.5, 0.03, f"Across 33 European transport sectors, 1 to 24 years ahead, nominal 90% intervals covered "
             f"{min(cov):.1f} to {max(cov):.1f}% of observed values",
             ha="center", fontsize=14, color=INK)
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf", "tif"):
        p = os.path.join(FIG, f"Graphical_abstract.{ext}")
        fig.savefig(p, dpi=300, facecolor="white")
        print("wrote", p)


if __name__ == "__main__":
    main()

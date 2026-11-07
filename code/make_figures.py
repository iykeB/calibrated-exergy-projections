"""
make_figures.py

Figure A: Turkiye, projections made with data to 2000 and calibrated on the other 32
countries, against the observed accounts and the point projection of Utlu and Hepbasli (2006).
Figure B: validation over the 33 countries (coverage by horizon with country bootstrap
intervals) and the simulation benchmark (coverage by origin year).
Reads ../results/framework.json and ../results/simulation.json; writes ../figures/.
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import framework as F
import projection_calibration as PC

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
BLUE, ORANGE, AQUA, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e34948"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
plt.rcParams.update({
    "font.family": "Arial", "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 600, "lines.linewidth": 1.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "custom",
    "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
})
UH = {"ex": 1385.73, "phi": 28.85, "imp": 700.0}  # Utlu and Hepbasli (2006), 2020
ORIGIN = 2000


def save(fig, p):
    """Write the PNG and a vector PDF (embedded TrueType fonts) next to it."""
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    fig.savefig(p[:-4] + ".pdf", bbox_inches="tight", facecolor="white")


def panel(ax, letter):
    ax.text(-0.02, 1.04, f"({letter})", transform=ax.transAxes, fontsize=9.5, fontweight="bold",
            ha="right", va="bottom", color=INK)


def turkiye_bands():
    A = F.accounts()
    _, gdp, pop = PC.load()
    rec = [r for r in F.collect(A, gdp, pop) if "ex_driver" in r]
    other = [r for r in rec if r["g"] != "TR"]
    mine = sorted([r for r in rec if r["g"] == "TR" and r["o"] == ORIGIN], key=lambda r: r["h"])
    Eex, Ephi, pairs = {}, {}, {}
    for r in other:
        e1, e2 = np.log(r["ex"] / r["ex_driver"]), r["phi"] - r["phi_persist"]
        Eex.setdefault(r["h"], []).append(e1)
        Ephi.setdefault(r["h"], []).append(e2)
        pairs.setdefault(r["h"], []).append((e1, e2))
    out = {k: {q: [] for q in ("lo", "mid", "hi", "lo50", "hi50")} for k in ("ex", "phi", "imp")}
    yrs = []
    for r in mine:
        h = r["h"]
        yrs.append(ORIGIN + h)
        for a, (klo, khi) in ((0.9, ("lo", "hi")), (0.5, ("lo50", "hi50"))):
            lo, _, hi = PC.smooth_quantiles(Eex, a)(h)
            out["ex"][klo].append(r["ex_driver"] * np.exp(lo))
            out["ex"][khi].append(r["ex_driver"] * np.exp(hi))
            lo2, _, hi2 = PC.smooth_quantiles(Ephi, a)(h)
            out["phi"][klo].append(100 * (r["phi_persist"] + lo2))
            out["phi"][khi].append(100 * (r["phi_persist"] + hi2))
            P = np.array([p for hh in (h - 1, h, h + 1) for p in pairs.get(hh, [])])
            imp = (1 - np.clip(r["phi_persist"] + P[:, 1], 0, 1)) ** 2 * r["ex_driver"] * np.exp(P[:, 0])
            q = np.percentile(imp, [50 * (1 - a), 50 * (1 + a)])
            out["imp"][klo].append(q[0])
            out["imp"][khi].append(q[1])
        out["ex"]["mid"].append(r["ex_driver"])
        out["phi"]["mid"].append(100 * r["phi_persist"])
        out["imp"]["mid"].append((1 - r["phi_persist"]) ** 2 * r["ex_driver"])
    # calibrated band around the Utlu and Hepbasli 2020 point (h = 20 from data to 2000)
    uhb = {}
    lo, _, hi = PC.smooth_quantiles(Eex, 0.9)(20)
    uhb["ex"] = (UH["ex"] * np.exp(lo), UH["ex"] * np.exp(hi))
    lo2, _, hi2 = PC.smooth_quantiles(Ephi, 0.9)(20)
    uhb["phi"] = (UH["phi"] + 100 * lo2, UH["phi"] + 100 * hi2)
    P = np.array([p for hh in (19, 20, 21) for p in pairs.get(hh, [])])
    phi_s = np.clip(UH["phi"] / 100 + P[:, 1], 0, 1)
    imp_s = (1 - phi_s) ** 2 * UH["ex"] * np.exp(P[:, 0])
    # rescale so the band is centred on their stated IMP rather than on (1 - phi)^2 Ex
    scale = UH["imp"] / ((1 - UH["phi"] / 100) ** 2 * UH["ex"])
    uhb["imp"] = tuple(np.percentile(imp_s * scale, [5, 95]))
    obs = {k: np.array(A["TR"][k]) * (100 if k == "phi" else 1) for k in ("ex", "phi", "imp")}
    return np.array(yrs), out, uhb, obs


def figure_a():
    yrs, B, uhb, obs = turkiye_bands()
    t = PC.YEARS
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    lab = {"ex": "Exergy input (PJ)", "phi": "Exergy efficiency (%)", "imp": "Improvement potential (PJ)"}
    for ax, k, letter in zip(axes, ("ex", "phi", "imp"), "abc"):
        b = B[k]
        ax.fill_between(yrs, b["lo"], b["hi"], color=BLUE, alpha=0.15, lw=0, label="90% interval")
        ax.fill_between(yrs, b["lo50"], b["hi50"], color=BLUE, alpha=0.30, lw=0, label="50% interval")
        ax.plot(yrs, b["mid"], color=BLUE, lw=1.6, label="Driver projection")
        h = t <= ORIGIN
        ax.plot(t[h], obs[k][h], "o", color=INK, ms=2.4, label="Data used (1990 to 2000)")
        ax.plot(t[~h], obs[k][~h], "D", color=RED, ms=2.8, mfc="white", mew=0.9, label="Observed (2001 to 2024)")
        lo, hi = uhb[k]
        ax.errorbar([2020.4], [UH[k]], yerr=[[UH[k] - lo], [hi - UH[k]]], fmt="*", color=ORANGE, ms=8,
                    mec="white", mew=0.5, elinewidth=1.1, capsize=2.5, label="Utlu and Hepbasli (2006),\ncalibrated 90% interval")
        ax.axvline(ORIGIN + 0.5, color=MUTED, lw=0.7, ls=":")
        ax.set_xlim(1989, 2025.5)
        ax.set_xlabel("Year")
        ax.set_ylabel(lab[k])
        panel(ax, letter)
    fig.tight_layout(w_pad=1.4)
    hd, lb = axes[0].get_legend_handles_labels()
    fig.legend(hd, [l.replace(",\n", ", ") for l in lb], loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=3, fontsize=7)
    p = os.path.join(FIG, "FigA_turkiye_2000_origin.png")
    save(fig, p)
    plt.close(fig)
    print("wrote", p)


def figure_b():
    R = json.load(open(os.path.join(RES, "framework.json")))
    S = json.load(open(os.path.join(RES, "simulation.json")))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ax = axes[0]
    bands = ["1-5", "6-10", "11-24", "all"]
    x = np.arange(len(bands))
    for j, (k, col, mk, nm) in enumerate((("ex", BLUE, "o", "Exergy input"), ("phi", AQUA, "s", "Exergy efficiency"),
                                          ("imp", VIOLET, "D", "Improvement potential"))):
        ev = R["driver|persist"][k]
        c = np.array([ev[b]["cover"] for b in bands]) * 100
        ci = np.array([ev[b]["ci95"] for b in bands]) * 100
        ax.errorbar(x + (j - 1) * 0.18, c, yerr=[c - ci[:, 0], ci[:, 1] - c], fmt=mk, color=col, ms=4.5,
                    mec="white", mew=0.5, elinewidth=1.0, capsize=2, label=nm)
    ax.axhline(90, color=INK2, lw=0.8, ls="--")
    ax.set_xticks(x, ["1 to 5", "6 to 10", "11 to 24", "All"])
    ax.set_xlabel("Forecast horizon (years)")
    ax.set_ylabel("Coverage of 90% interval (%)")
    ax.set_ylim(70, 100)
    ax.legend(loc="lower left", fontsize=7)
    panel(ax, "a")
    ax = axes[1]
    origins = ["o2000", "o2005", "o2010", "o2015"]
    xo = np.arange(len(origins))
    cols = {"S0": INK2, "S1": BLUE, "S2": AQUA, "S3": VIOLET, "S4": RED}
    mks = {"S0": "o", "S1": "s", "S2": "^", "S3": "v", "S4": "P"}
    names = {"S0": "S0 exchangeable", "S1": "S1 growth breaks", "S2": "S2 common shocks",
             "S3": "S3 diverse elasticities", "S4": "S4 shared regional break"}
    for j, s in enumerate(cols):
        m = np.array([S[s][o]["mean"] for o in origins]) * 100
        lo = np.array([S[s][o]["p05"] for o in origins]) * 100
        hi = np.array([S[s][o]["p95"] for o in origins]) * 100
        ax.errorbar(xo + (j - 2) * 0.12, m, yerr=[m - lo, hi - m], fmt=mks[s], color=cols[s], ms=4.0,
                    mec="white", mew=0.5, elinewidth=0.9, capsize=1.5, label=names[s])
    real = R["by_origin_driver"]
    ax.plot(xo, [real[str(o)]["cover"] * 100 for o in (2000, 2005, 2010, 2015)], "D", color=INK, ms=4.5,
            mfc="white", mew=1.0, label="Observed panel")
    ax.axhline(90, color=INK2, lw=0.8, ls="--")
    ax.set_xticks(xo, ["2000", "2005", "2010", "2015"])
    ax.set_xlabel("Origin year")
    ax.set_ylabel("Coverage of 90% interval (%)")
    ax.set_ylim(50, 100)
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    panel(ax, "b")
    fig.tight_layout(w_pad=2.0)
    p = os.path.join(FIG, "FigB_validation.png")
    save(fig, p)
    plt.close(fig)
    print("wrote", p)




def figure_1():
    """Schematic of the framework: accounts, projection, error calibration, intervals, validation."""
    from matplotlib.patches import FancyArrowPatch, Rectangle
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.35, 4.85)
    ax.axis("off")
    LINE = INK2
    H = 1.35
    TOP, BOT = 2.45, 0.5

    def box(x, y, w, title, body, dash=None):
        ax.add_patch(Rectangle((x, y), w, H, fc="white", ec=LINE, lw=0.8, ls=dash or "-"))
        ax.text(x + w / 2, y + H - 0.12, title, ha="center", va="top", fontsize=7.6, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + H - 0.4, body, ha="center", va="top", fontsize=7, color=INK2, linespacing=1.35)

    def arrow(pts):
        xs, ys = zip(*pts)
        ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=LINE, lw=0.8, solid_capstyle="butt")
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=8, lw=0.8,
                                     color=LINE, shrinkA=0, shrinkB=0))

    cols = {"acc": (0.1, 2.3), "proj": (2.85, 2.2), "cal": (5.5, 2.1), "out": (8.05, 1.85)}
    mid = {k: x + w / 2 for k, (x, w) in cols.items()}
    for k, t in (("acc", "Accounts"), ("proj", "Projection"), ("cal", "Calibration"), ("out", "Output")):
        ax.text(mid[k], 4.7, t, ha="center", va="center", fontsize=8, color=INK, style="italic")
    box(cols["acc"][0], TOP, cols["acc"][1], "Panel of countries",
        "energy use by mode and fuel\nincome and population\n1990 to 2024")
    box(cols["acc"][0], BOT, cols["acc"][1], "Exergy accounts",
        "exergy input $Ex$, efficiency $\\varphi$\nimprovement potential $IMP$\nEqs. (1) to (3)")
    box(cols["proj"][0], TOP, cols["proj"][1], "Point projector",
        "drift, driver or oracle\nor a published value\nEqs. (4) and (5)")
    box(cols["proj"][0], BOT, cols["proj"][1], "Projection errors",
        "$e^{Ex}=\\ln(Ex/\\hat{Ex})$\n$e^{\\varphi}=\\varphi-\\hat{\\varphi}$\nby origin and horizon $h$\nEq. (6)")
    box(cols["cal"][0], TOP, cols["cal"][1], "Leave one country out",
        "pool the errors of the\nother countries only")
    box(cols["cal"][0], BOT, cols["cal"][1], "Horizon law",
        "$q_{\\pm}(h)=b\\,h\\pm a_{\\pm}h^{p_{\\pm}}$\nfitted to pooled quantiles\nEq. (7)")
    box(cols["out"][0], TOP, cols["out"][1], "Calibrated intervals",
        "$Ex$ and $\\varphi$, Eq. (8)\n$IMP$ from paired\nerrors, Eq. (9)")
    box(cols["out"][0], BOT, cols["out"][1], "Validation",
        "coverage, width and\ninterval score; country\nbootstrap; simulation", dash=(0, (4, 2)))
    yt, yb = TOP + H / 2, BOT + H / 2
    right = {k: x + w for k, (x, w) in cols.items()}
    left = {k: x for k, (x, w) in cols.items()}
    arrow([(mid["acc"], TOP), (mid["acc"], BOT + H)])
    arrow([(right["acc"], yt), (left["proj"], yt)])
    arrow([(right["acc"], yb), (left["proj"], yb)])
    arrow([(mid["proj"], TOP), (mid["proj"], BOT + H)])
    arrow([(right["proj"], yb), (left["cal"] - 0.22, yb), (left["cal"] - 0.22, yt), (left["cal"], yt)])
    arrow([(mid["cal"], TOP), (mid["cal"], BOT + H)])
    arrow([(right["cal"], yb), (left["out"] - 0.22, yb), (left["out"] - 0.22, yt), (left["out"], yt)])
    arrow([(mid["out"], TOP), (mid["out"], BOT + H)])
    ytop = TOP + H + 0.28
    arrow([(mid["proj"], TOP + H), (mid["proj"], ytop), (mid["out"], ytop), (mid["out"], TOP + H)])
    ax.text(mid["cal"], ytop + 0.1, "point value", ha="center", va="bottom", fontsize=7, color=INK2)
    p = os.path.join(FIG, "Fig1_framework.png")
    save(fig, p)
    plt.close(fig)
    print("wrote", p)


def figure_errors():
    """(a) Mean absolute log error by horizon for the three projectors; (b) interval width and score."""
    A = F.accounts()
    _, gdp, pop = PC.load()
    rec = [r for r in F.collect(A, gdp, pop) if "ex_driver" in r]
    R = json.load(open(os.path.join(RES, "framework.json")))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    hs = np.arange(1, F.HMAX + 1)
    styles = (("drift", ORANGE, "Drift"), ("driver", BLUE, "Driver"), ("oracle", AQUA, "Oracle"))
    for k, col, nm in styles:
        m = [np.mean([abs(np.log(r["ex"] / r["ex_" + k])) for r in rec if r["h"] == h]) for h in hs]
        ax.plot(hs, m, color=col, lw=1.6, label=nm)
        ax.text(hs[-1] + 0.3, m[-1], nm, color=INK2, fontsize=7, va="center")
    ax.set_xlabel("Forecast horizon (years)")
    ax.set_ylabel("Mean absolute log error of $Ex$")
    ax.set_xlim(0, 27.5)
    ax.set_ylim(0, None)
    panel(ax, "a")
    ax = axes[1]
    x = np.arange(3)
    w = [R[f"{k}|persist"]["ex"]["mean_relative_width"] for k, _, _ in styles]
    sc = [R[f"{k}|persist"]["ex"]["interval_score"] for k, _, _ in styles]
    ax.bar(x - 0.19, w, width=0.36, color=MUTED, label="Relative width")
    ax.bar(x + 0.19, sc, width=0.36, color=BLUE, label="Interval score")
    for xi, a_, b_ in zip(x, w, sc):
        ax.text(xi - 0.19, a_ + 0.03, f"{a_:.2f}", ha="center", fontsize=7, color=INK2)
        ax.text(xi + 0.19, b_ + 0.03, f"{b_:.2f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, [nm for _, _, nm in styles])
    ax.set_ylabel("Width or score (lower is better)")
    ax.set_ylim(0, 1.95)
    ax.legend(loc="upper right", fontsize=7)
    panel(ax, "b")
    fig.tight_layout(w_pad=2.0)
    p = os.path.join(FIG, "Fig2_errors_sharpness.png")
    save(fig, p)
    plt.close(fig)
    print("wrote", p)


def figure_outside():
    """Coverage for countries outside the panel, three quantities, with the 90% target."""
    N = json.load(open(os.path.join(RES, "noneu_check.json")))
    X = json.load(open(os.path.join(RES, "external_check.json")))["external"]
    rows = [("Australia", N["Australia"]), ("Canada", N["Canada"]), ("Chile", N["Chile"]), ("Japan", N["Japan"]),
            ("Korea", N["Korea"]), ("Mexico", N["Mexico"]), ("New Zealand", N["New Zealand"]),
            ("United States", N["USA"]),
            ("Eight, all panel errors", N["strong_pooled"]), ("Eight, 20 income peers", N["strong_peer20"]),
            ("United Kingdom", {k: X["UK"][k]["cover"] for k in ("ex", "phi", "imp")}),
            ("Ukraine", {k: X["UA"][k]["cover"] for k in ("ex", "phi", "imp")})]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    y = np.arange(len(rows))[::-1]
    for j, (k, col, mk, nm) in enumerate((("ex", BLUE, "o", "Exergy input"), ("phi", AQUA, "s", "Exergy efficiency"),
                                          ("imp", VIOLET, "D", "Improvement potential"))):
        ax.plot([100 * r[k] for _, r in rows], y + (1 - j) * 0.2, mk, color=col, ms=4.2, mec="white", mew=0.5,
                ls="", label=nm)
    ax.axvline(90, color=INK2, lw=0.8, ls="--")
    for yy in (y[7] - 0.5, y[9] - 0.5):
        ax.axhline(yy, color=GRID, lw=0.8)
    ax.set_yticks(y, [nm for nm, _ in rows])
    ax.set_xlabel("Coverage of 90% interval (%)")
    ax.set_xlim(25, 102)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=7, ncol=3)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    p = os.path.join(FIG, "Fig4_outside_panel.png")
    save(fig, p)
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    figure_1()
    figure_errors()
    figure_b()
    figure_outside()
    figure_a()

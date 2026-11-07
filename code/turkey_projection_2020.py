"""
turkey_projection_2020.py

Scores the deterministic 2020 projection of Utlu and Hepbasli (2006) against
a calibrated probabilistic forecast that uses the same information they had
(Turkiye transport fuel use 1990 to 2000) and against the observed data
(Eurostat nrg_bal_c, 2001 to 2024).

Design choices (all fixed before looking at post 2000 data):
  model        equal weight combination of GP hybrid and probabilistic drift
  calibration  kappa = 1.15, transferred from the Nigeria study (no Turkish data used)
  series       road diesel, road Otto (gasoline plus LPG, as in their Otto engine class),
               rail diesel, rail electricity, domestic aviation, sea diesel, sea fuel oil
  efficiency   their 1999 to 2000 mode efficiencies (Utlu and Hepbasli 2004, Table IV),
               each uncertain with a standard deviation of 2 points (case E1)
  grade        exergy to energy factors from their Table 1 (2006)
Writes ../results/turkey_projection_2020.json and ../figures/Fig_projection_2020.png.
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import revision as R
import turkey_first_test as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUTF = os.path.join(HERE, "..", "results", "turkey_projection_2020.json")
FIG = os.path.join(HERE, "..", "figures")
ORIGIN = 2000
KAPPA = 1.15  # Nigeria combined model, Thermo paper
N = 4000
TQ = np.arange(ORIGIN + 1, 2025, dtype=float)

# mode efficiency (%) from Utlu and Hepbasli (2004) Table IV and grade factor from (2006) Table 1
SERIES = {
    "Road_diesel": (35, 1.07),
    "Road_Otto": (20, 0.99),
    "Rail_diesel": (40, 1.07),
    "Rail_electricity": (40, 1.00),
    "Air_jet": (10, 0.99),
    "Sea_diesel": (43, 1.07),
    "Sea_fueloil": (43, 0.99),
}
EFF_SD = 2.0

# Utlu and Hepbasli (2006): exergy input 497.01 PJ (2000) and 1385.73 PJ (2020), Table 2;
# exergy efficiency 23.65 and 28.85% (Table 3); improvement potential about 290 PJ rising
# to about 700 PJ at 4.5% a year (Fig. 3).
UH = {"ex_in": (497.01, 1385.73), "phi": (23.65, 28.85), "imp": (290.0, 700.0)}

BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
plt.rcParams.update({
    "font.family": "Arial", "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 6.8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 600, "lines.linewidth": 1.6,
})


def series_data():
    t, Y = T.load()
    Y = dict(Y)
    Y["Road_Otto"] = np.nansum([Y.pop("Road_gasoline"), Y.pop("Road_LPG")], axis=0)
    return t, {s: Y[s] for s in SERIES}


def accounts(E, eff):
    """Exergy input, exergy efficiency (%) and improvement potential from energy use E[s]."""
    ex = {s: SERIES[s][1] * E[s] for s in SERIES}
    ex_in = sum(ex.values())
    ex_out = sum(eff[s] / 100.0 * ex[s] for s in SERIES)
    phi = ex_out / ex_in
    return ex_in, 100 * phi, (1 - phi) * (ex_in - ex_out)


def main():
    t, Y = series_data()
    tr = t <= ORIGIN
    rng = np.random.default_rng(20)
    E = {}
    for j, s in enumerate(SERIES):
        y = np.nan_to_num(Y[s][tr])
        D = R.model_draws(t[tr], y, TQ, N, 7000 + 31 * j, "combined")
        med = np.median(D, axis=0)
        E[s] = np.maximum(0.0, med + KAPPA * (D - med))
    eff = {s: rng.normal(SERIES[s][0], EFF_SD, size=(N, 1)) for s in SERIES}
    ex_in, phi, imp = accounts(E, eff)

    # observed accounts with the base efficiencies
    Eo = {s: np.nan_to_num(Y[s]) for s in SERIES}
    o_in, o_phi, o_imp = accounts(Eo, {s: SERIES[s][0] for s in SERIES})

    q = lambda x: {p: np.nanpercentile(x, p, axis=0).tolist() for p in (5, 25, 50, 75, 95)}
    obs_mask = t > ORIGIN
    res = {"origin": ORIGIN, "kappa": KAPPA, "years": TQ.tolist(),
           "fan": {"ex_in": q(ex_in), "phi": q(phi), "imp": q(imp)},
           "observed": {"years": t.tolist(), "ex_in": o_in.tolist(), "phi": o_phi.tolist(), "imp": o_imp.tolist()},
           "uh2006": UH}
    # where the observed values fall in the predictive distribution
    for key, F, O in (("ex_in", ex_in, o_in), ("phi", phi, o_phi), ("imp", imp, o_imp)):
        ob = O[obs_mask]
        pit = np.array([np.nanmean(F[:, k] <= ob[k]) for k in range(len(TQ))])
        lo, hi = np.nanpercentile(F, 5, axis=0), np.nanpercentile(F, 95, axis=0)
        k20 = int(np.where(TQ == 2020)[0][0])
        res[key + "_score"] = {"pit": pit.tolist(), "cover90_2001_2024": float(np.mean((ob >= lo) & (ob <= hi))),
                               "pit_2019": float(pit[k20 - 1]), "pit_2020": float(pit[k20]),
                               "uh_2020_percentile": float(np.mean(F[:, k20] <= UH[key][1]) * 100)}
    json.dump(res, open(OUTF, "w"), indent=1)
    for key in ("ex_in", "phi", "imp"):
        s = res[key + "_score"]
        print(f"{key}: observed inside 90% in {s['cover90_2001_2024']:.0%} of 2001-24; "
              f"PIT 2019 {s['pit_2019']:.2f}, 2020 {s['pit_2020']:.2f}; UH 2020 value at percentile {s['uh_2020_percentile']:.0f}")
    print("observed 2019/2020/2024 ex_in", [round(o_in[t == y][0], 1) for y in (2019, 2020, 2024)],
          "phi", [round(o_phi[t == y][0], 2) for y in (2019, 2020, 2024)],
          "imp", [round(o_imp[t == y][0], 1) for y in (2019, 2020, 2024)])
    figure(res, t, Y)


def figure(res, t, Y):
    os.makedirs(FIG, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    yrs = np.array(res["years"])
    ot = np.array(res["observed"]["years"])
    labels = {"ex_in": "Exergy input (PJ)", "phi": "Exergy efficiency (%)", "imp": "Improvement potential (PJ)"}
    for ax, key, letter in zip(axes, ("ex_in", "phi", "imp"), "abc"):
        f = res["fan"][key]
        ax.fill_between(yrs, f[5], f[95], color=BLUE, alpha=0.15, lw=0, label="90% interval")
        ax.fill_between(yrs, f[25], f[75], color=BLUE, alpha=0.30, lw=0, label="50% interval")
        ax.plot(yrs, f[50], color=BLUE, lw=1.6, label="Calibrated median")
        o = np.array(res["observed"][key])
        h = ot <= ORIGIN
        ax.plot(ot[h], o[h], "o", color=INK, ms=2.4, label="Data used (1990 to 2000)")
        ax.plot(ot[~h], o[~h], "D", color=RED, ms=2.8, mfc="white", mew=0.9, label="Observed (2001 to 2024)")
        a, b = UH[key]
        if key == "imp":
            uy = np.arange(2000, 2021)
            uv = a * (b / a) ** ((uy - 2000) / 20)
        else:
            uy = np.array([2000, 2020])
            uv = np.array([a, b])
        ax.plot(uy, uv, ls="--", color=ORANGE, lw=1.1)
        ax.plot([2020], [b], "*", color=ORANGE, ms=8, mec="white", mew=0.5, label="Utlu and Hepbasli (2006)")
        ax.axvline(ORIGIN + 0.5, color=MUTED, lw=0.7, ls=":")
        ax.set_xlim(1989, 2025)
        ax.set_xlabel("Year")
        ax.set_ylabel(labels[key])
        ax.text(-0.02, 1.04, f"({letter})", transform=ax.transAxes, fontsize=9.5, fontweight="bold",
                ha="right", va="bottom", color=INK)
    axes[0].legend(loc="upper left", fontsize=6.3)
    fig.tight_layout(w_pad=1.4)
    p = os.path.join(FIG, "Fig_projection_2020.png")
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    print("wrote", p)


if __name__ == "__main__":
    main()

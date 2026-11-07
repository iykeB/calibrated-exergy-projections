"""
external_check.py

Two robustness checks of the calibration.
  1. External countries. The United Kingdom (data to 2019) and Ukraine (data to 2020) are
     not in the 33 country panel. Their projections are calibrated with the errors of all
     33 panel countries and scored on their own observed accounts.
  2. Smoothing law. Coverage of the panel with the raw empirical quantiles at each horizon
     instead of the smoothed law of Equation (7).
Writes ../results/external_check.json.
"""

import json
import os

import numpy as np

import framework as F
import projection_calibration as PC

HERE = os.path.dirname(os.path.abspath(__file__))
OUTF = os.path.join(HERE, "..", "results", "external_check.json")
EXTRA = {"UK": "GBR", "UA": "UKR"}


def main():
    _, gdp, pop = PC.load()
    A = F.accounts()
    rec = [r for r in F.collect(A, gdp, pop) if "ex_driver" in r]
    panel = list(F.ISO3)
    # accounts and projections for the external countries
    PC.ISO3.update(EXTRA)
    try:
        Ax = F.accounts()
        _, gdpx, popx = PC.load()
    finally:
        for k in EXTRA:
            PC.ISO3.pop(k)
    res = {"external": {}}
    E, Ephi, pairs = {}, {}, {}
    for r in rec:
        e1, e2 = np.log(r["ex"] / r["ex_driver"]), r["phi"] - r["phi_persist"]
        E.setdefault(r["h"], []).append(e1)
        Ephi.setdefault(r["h"], []).append(e2)
        pairs.setdefault(r["h"], []).append((e1, e2))
    Q, Qp = PC.smooth_quantiles(E, 0.9), PC.smooth_quantiles(Ephi, 0.9)
    el = {(g, int(o)): PC.elasticity(A[g]["ex"], gdp[g], pop[g], o) for g in panel for o in F.ORIGINS}
    for g in EXTRA:
        ex, phi, imp = Ax[g]["ex"], Ax[g]["phi"], Ax[g]["imp"]
        last = int(F.YEARS[np.nonzero(ex)[0][-1]])
        fl = {"ex": [], "phi": [], "imp": []}
        for o in F.ORIGINS:
            h = last - int(o)
            if h < 1:
                continue
            h = min(h, F.HMAX)
            eps = float(np.median([el[(c, int(o))] for c in panel if np.isfinite(el[(c, int(o))])]))
            pr = PC.project(ex, gdpx[g], popx[g], o, h, eps)
            if "driver" not in pr:
                continue
            i1 = int(o - F.YEARS[0]) + 1
            for j in range(h):
                hh = j + 1
                lo, _, hi = Q(hh)
                y = ex[i1 + j]
                fl["ex"].append(pr["driver"][j] * np.exp(lo) <= y <= pr["driver"][j] * np.exp(hi))
                lo2, _, hi2 = Qp(hh)
                fl["phi"].append(phi[int(o - F.YEARS[0])] + lo2 <= phi[i1 + j] <= phi[int(o - F.YEARS[0])] + hi2)
                P = np.array([p for k in (hh - 1, hh, hh + 1) for p in pairs.get(k, [])])
                s = (1 - np.clip(phi[int(o - F.YEARS[0])] + P[:, 1], 0, 1)) ** 2 * pr["driver"][j] * np.exp(P[:, 0])
                ilo, ihi = np.percentile(s, [5, 95])
                fl["imp"].append(ilo <= imp[i1 + j] <= ihi)
        res["external"][g] = {k: {"cover": float(np.mean(v)), "n": len(v)} for k, v in fl.items()}
        res["external"][g]["last_year"] = last
        print(g, "to", last, {k: f"{100*v['cover']:.1f}% (n={v['n']})" for k, v in res["external"][g].items() if k != "last_year"})
    # smoothing law against raw empirical quantiles, leave one country out
    raw, smo = [], []
    for g in panel:
        Eo = {}
        for r in rec:
            if r["g"] != g:
                Eo.setdefault(r["h"], []).append(np.log(r["ex"] / r["ex_driver"]))
        Qs = PC.smooth_quantiles(Eo, 0.9)
        for r in rec:
            if r["g"] == g:
                e = np.log(r["ex"] / r["ex_driver"])
                lo, hi = np.percentile(Eo[r["h"]], [5, 95])
                raw.append(lo <= e <= hi)
                lo2, _, hi2 = Qs(r["h"])
                smo.append(lo2 <= e <= hi2)
    res["smoothing"] = {"raw_empirical": float(np.mean(raw)), "smoothed_law": float(np.mean(smo))}
    print("Ex coverage, raw quantiles", f"{100*np.mean(raw):.1f}%", "smoothed law", f"{100*np.mean(smo):.1f}%")
    json.dump(res, open(OUTF, "w"), indent=1)


if __name__ == "__main__":
    main()

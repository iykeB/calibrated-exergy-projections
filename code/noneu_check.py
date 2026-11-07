"""
noneu_check.py

External validation outside Europe. Transport energy use by mode and fuel comes from the
UN Energy Statistics Database (UNSD, dataflow DF_UNDATA_ENERGY), which follows the same
international recommendations as Eurostat. Accounts use the same grade factors and mode
efficiencies as framework.py. Each country is projected with the driver projector, with the
elasticity pooled over the 33 panel countries, and calibrated only with the panel errors.
Scored on its own observed accounts, 1990 to 2023.
Selection rule (applied before scoring): UNSD series continuous 1990 to 2023, no year to year
change above 20% outside the pandemic years 2020 to 2022, all four modes reported, and income
per person in 2000 inside the range of the panel. Argentina (no rail) is scored but reported
separately. A second variant calibrates with the 20 panel countries nearest in log income
per person at the origin.
Writes ../results/noneu_check.json.
"""

import json
import os

import numpy as np

import framework as F
import projection_calibration as PC

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUTF = os.path.join(HERE, "..", "results", "noneu_check.json")
COUNTRIES = {"USA": "USA", "Canada": "CAN", "Japan": "JPN", "Australia": "AUS", "Korea": "KOR",
             "New Zealand": "NZL", "Chile": "CHL", "Mexico": "MEX", "Argentina": "ARG"}
STRONG = ("USA", "Canada", "Japan", "Australia", "Korea", "New Zealand", "Chile", "Mexico")
MODE = {"1221": "FC_TRA_ROAD_E", "1222": "FC_TRA_RAIL_E", "1223": "FC_TRA_DAVI_E", "1224": "FC_TRA_DNAVI_E"}
FUEL = {"4670": "O4671XR5220B", "4652": "O4652XR5210B", "4661": "O4661XR5230B", "4680": "O4680",
        "4630": "O4630", "3000": "G3000", "7000": "E7000", "5210": "R5210B", "5220": "R5220B"}
LAST = 2023


def wb(name):
    t = {}
    for r in json.load(open(os.path.join(DATA, f"wb_nonEU_{name}.json")))[1]:
        if r["value"] is not None:
            t.setdefault(r["countryiso3code"], {})[int(r["date"])] = r["value"]
    return t


def accounts_unsd(rec):
    yrs = F.YEARS
    ex, out = np.zeros(len(yrs)), np.zeros(len(yrs))
    for key, tj in rec.items():
        y, tr, cm = key.split("|")
        y = int(y)
        if y > LAST or not np.isfinite(tj) or tr not in MODE:
            continue
        b, s = MODE[tr], FUEL[cm]
        eta = F.ETA[b].get(s, list(F.ETA[b].values())[0])
        i = y - int(yrs[0])
        ex[i] += F.GAMMA[s] * tj / 1000.0
        out[i] += eta / 100.0 * F.GAMMA[s] * tj / 1000.0
    with np.errstate(invalid="ignore", divide="ignore"):
        phi = out / ex
    return ex, phi, (1 - phi) * (ex - out)


def peers_by_income(lg_panel, lgc, i0, k):
    d = sorted((abs(lg_panel[c][i0] - lgc[i0]), c) for c in F.ISO3 if np.isfinite(lg_panel[c][i0]))
    return [c for _, c in d[:k]]


def main():
    U = json.load(open(os.path.join(DATA, "unsd", "unsd_transport_TJ.json")))
    G, P = wb("NY.GDP.MKTP.KD"), wb("SP.POP.TOTL")
    A = F.accounts()
    _, gdp, pop = PC.load()
    rec = [r for r in F.collect(A, gdp, pop) if "ex_driver" in r]
    E, Ephi, pairs = {}, {}, {}
    for r in rec:
        e1, e2 = np.log(r["ex"] / r["ex_driver"]), r["phi"] - r["phi_persist"]
        E.setdefault(r["h"], []).append(e1)
        Ephi.setdefault(r["h"], []).append(e2)
        pairs.setdefault(r["h"], []).append((e1, e2))
    Q, Qp = PC.smooth_quantiles(E, 0.9), PC.smooth_quantiles(Ephi, 0.9)
    el = {(g, int(o)): PC.elasticity(A[g]["ex"], gdp[g], pop[g], o) for g in F.ISO3 for o in F.ORIGINS}
    res = {}
    pooled = {"ex": [], "phi": [], "imp": []}
    pooled_width = []
    for nm, iso in COUNTRIES.items():
        ex, phi, imp = accounts_unsd(U[nm])
        gd = np.array([G[iso].get(int(y), np.nan) for y in F.YEARS])
        pp = np.array([P[iso].get(int(y), np.nan) for y in F.YEARS])
        fl = {"ex": [], "phi": [], "imp": []}
        for o in F.ORIGINS:
            h = min(LAST - int(o), F.HMAX)
            if h < 1:
                continue
            eps = float(np.median([el[(c, int(o))] for c in F.ISO3 if np.isfinite(el[(c, int(o))])]))
            pr = PC.project(ex, gd, pp, o, h, eps)
            if "driver" not in pr:
                continue
            i0 = int(o - F.YEARS[0])
            for j in range(h):
                hh, y = j + 1, i0 + 1 + j
                lo, _, hi = Q(hh)
                fl["ex"].append(pr["driver"][j] * np.exp(lo) <= ex[y] <= pr["driver"][j] * np.exp(hi))
                if nm in STRONG:
                    pooled_width.append(np.exp(hi) - np.exp(lo))
                lo2, _, hi2 = Qp(hh)
                fl["phi"].append(phi[i0] + lo2 <= phi[y] <= phi[i0] + hi2)
                Pm = np.array([p for k in (hh - 1, hh, hh + 1) for p in pairs.get(k, [])])
                s = (1 - np.clip(phi[i0] + Pm[:, 1], 0, 1)) ** 2 * pr["driver"][j] * np.exp(Pm[:, 0])
                ilo, ihi = np.percentile(s, [5, 95])
                fl["imp"].append(ilo <= imp[y] <= ihi)
        res[nm] = {k: float(np.mean(v)) for k, v in fl.items()}
        res[nm]["n"] = len(fl["ex"])
        res[nm]["phi_2023"] = float(phi[LAST - int(F.YEARS[0])])
        if nm in STRONG:
            for k in pooled:
                pooled[k] += fl[k]
        print(f"{nm:12s} n={res[nm]['n']:3d} cover90 Ex {100*res[nm]['ex']:5.1f}% phi {100*res[nm]['phi']:5.1f}% "
              f"IMP {100*res[nm]['imp']:5.1f}% | phi 2023 {100*res[nm]['phi_2023']:.1f}%")
    # peer variant: 20 panel countries nearest in income at the origin
    lg = {c: np.log(gdp[c] / pop[c]) for c in F.ISO3}
    byc = {}
    for r in rec:
        byc.setdefault(r["g"], []).append(r)
    peer = {"ex": [], "phi": [], "imp": [], "width": []}
    for nm in STRONG:
        iso = COUNTRIES[nm]
        ex, phi, imp = accounts_unsd(U[nm])
        gd = np.array([G[iso].get(int(y), np.nan) for y in F.YEARS])
        pp = np.array([P[iso].get(int(y), np.nan) for y in F.YEARS])
        lgc = np.log(gd / pp)
        for o in F.ORIGINS:
            h = min(LAST - int(o), F.HMAX)
            i0 = int(o - F.YEARS[0])
            Ek, Epk, prk = {}, {}, {}
            for c in peers_by_income(lg, lgc, i0, 20):
                for r in byc[c]:
                    e1, e2 = np.log(r["ex"] / r["ex_driver"]), r["phi"] - r["phi_persist"]
                    Ek.setdefault(r["h"], []).append(e1)
                    Epk.setdefault(r["h"], []).append(e2)
                    prk.setdefault(r["h"], []).append((e1, e2))
            Qk, Qpk = PC.smooth_quantiles(Ek, 0.9), PC.smooth_quantiles(Epk, 0.9)
            eps = float(np.median([el[(c, int(o))] for c in F.ISO3 if np.isfinite(el[(c, int(o))])]))
            pr = PC.project(ex, gd, pp, o, h, eps)["driver"]
            for j in range(h):
                hh, y = j + 1, i0 + 1 + j
                lo, _, hi = Qk(hh)
                peer["ex"].append(pr[j] * np.exp(lo) <= ex[y] <= pr[j] * np.exp(hi))
                peer["width"].append(np.exp(hi) - np.exp(lo))
                lo2, _, hi2 = Qpk(hh)
                peer["phi"].append(phi[i0] + lo2 <= phi[y] <= phi[i0] + hi2)
                Pm = np.array([p for k in (hh - 1, hh, hh + 1) for p in prk.get(k, [])])
                s_ = (1 - np.clip(phi[i0] + Pm[:, 1], 0, 1)) ** 2 * pr[j] * np.exp(Pm[:, 0])
                ilo, ihi = np.percentile(s_, [5, 95])
                peer["imp"].append(ilo <= imp[y] <= ihi)
    res["strong_peer20"] = {k: float(np.mean(v)) for k, v in peer.items()}
    print("strong group, 20 income peers:", {k: round(100 * v, 1) if k != "width" else round(v, 2) for k, v in res["strong_peer20"].items()})
    res["strong_pooled"] = {k: float(np.mean(v)) for k, v in pooled.items()}
    res["strong_pooled"]["n"] = len(pooled["ex"])
    res["strong_pooled"]["width"] = float(np.mean(pooled_width))
    print("strong group pooled:", {k: (round(100 * v, 1) if k != "n" else v) for k, v in res["strong_pooled"].items()})
    json.dump(res, open(OUTF, "w"), indent=1)


if __name__ == "__main__":
    main()

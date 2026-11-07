"""
framework.py

Calibrated uncertainty for point projections of national transport energy and exergy
accounts: exergy input Ex, exergy efficiency phi and improvement potential IMP.

Accounts. For each country and year, mode and fuel energy use (Eurostat nrg_bal_c) gives
exergy input Ex = sum gamma_f E_mf and exergy output sum eta_mf gamma_f E_mf, so that
phi = output / Ex and IMP = (1 - phi) (Ex - output) (Van Gool). eta_mf are fixed mode
efficiencies, so phi moves with the mix of modes and fuels (the mix effect).

Projections from each origin o and horizon h (point values only):
  Ex   drift    last 10 year log trend
       driver   per capita income elasticity pooled over the other countries; income and
                population at their last 10 year rates (planner information)
       oracle   driver with realised income and population (scenario bound)
  phi  persist  phi held at its origin value (the usual assumption in sectoral studies)
       drift    last 10 year linear trend
Calibration. The errors e_Ex = ln(obs / proj) and e_phi = obs - proj are collected by horizon.
For a held out country, intervals use only the other countries' errors (leave one country
out). Ex and phi bands use the smoothed horizon law of projection_calibration.py. IMP bands
use the paired errors (e_Ex, e_phi) of the same other country, origin and horizon, so the
dependence between the two errors is kept without a copula. Errors at h-1, h and h+1 are
pooled for IMP. Sharpness and calibration are scored together with the interval score
(Gneiting and Raftery 2007): on the log scale for Ex, in points for phi and relative to
the observed value for IMP. Coverage is reported with a 95% interval from a bootstrap over countries,
which treats countries, not overlapping forecasts, as the independent units.
Writes ../results/framework.json.
"""

import json
import os

import numpy as np

import projection_calibration as PC

HERE = os.path.dirname(os.path.abspath(__file__))
OUTF = os.path.join(HERE, "..", "results", "framework.json")
YEARS, ORIGINS, HMAX, ISO3 = PC.YEARS, PC.ORIGINS, PC.HMAX, PC.ISO3

# grade factors: Utlu and Hepbasli (2006), Table 1. Mode efficiencies: Utlu and Hepbasli
# (2004), Table IV (diesel engines 35%, Otto engines 20%, aeroplane 10%, ship 43%, train 40%)
# and Section 4.3 for electric drive (half large motors at 90%, half small at 80%: 85%).
GAMMA = PC.GAMMA
ETA = {
    "FC_TRA_ROAD_E": {"O4671XR5220B": 35, "R5220B": 35, "O4652XR5210B": 20, "R5210B": 20,
                      "O4630": 20, "G3000": 20, "E7000": 85},
    "FC_TRA_RAIL_E": {"O4671XR5220B": 40, "R5220B": 40, "E7000": 85},
    "FC_TRA_DAVI_E": {"O4661XR5230B": 10, "O4652XR5210B": 10},
    "FC_TRA_DNAVI_E": {"O4671XR5220B": 43, "R5220B": 43, "O4680": 43, "O4652XR5210B": 20,
                       "R5210B": 20},
}
ALPHA = 0.9
N_BOOT = 1000
TECH = 0.0  # technical efficiency gain, percentage points per year after 1990 (sensitivity only)


def accounts():
    d = json.load(open(os.path.join(PC.DATA, "eurostat_nrg_bal_c_all_transport_1990_2024.json")))
    ids, size = d["id"], d["size"]
    idx = {k: d["dimension"][k]["category"]["index"] for k in ids}

    def val(**kw):
        p = 0
        for k, s in zip(ids, size):
            p = p * s + (idx[k][kw[k]] if k in kw else 0)
        return d["value"].get(str(p)) or 0.0

    A = {}
    for g in ISO3:
        ex = np.zeros(len(YEARS))
        out = np.zeros(len(YEARS))
        for b in idx["nrg_bal"]:
            for s in idx["siec"]:
                e = np.array([val(geo=g, nrg_bal=b, siec=s, time=str(y)) for y in YEARS]) / 1000.0
                if not e.any():
                    continue
                eta = ETA[b].get(s)
                if eta is None:  # rare pairs (below 0.01% of panel use) take the mode's main fuel value
                    eta = list(ETA[b].values())[0]
                ex += GAMMA[s] * e
                out += (eta + TECH * (YEARS - YEARS[0])) / 100.0 * GAMMA[s] * e
        A[g] = {"ex": ex, "phi": out / ex}
        A[g]["imp"] = (1 - A[g]["phi"]) * (ex - out)
    return A


def phi_project(phi, origin, h):
    i0 = int(origin - YEARS[0])
    steps = np.arange(1, h + 1)
    return {"persist": np.full(h, phi[i0]),
            "drift": phi[i0] + (phi[i0] - phi[i0 - 10]) / 10 * steps}


def collect(A, gdp, pop):
    ex = {g: A[g]["ex"] for g in ISO3}
    el = {(g, int(o)): PC.elasticity(ex[g], gdp[g], pop[g], o) for g in ISO3 for o in ORIGINS}
    rec = []  # dicts per (g, o, h)
    for g in ISO3:
        for o in ORIGINS:
            others = [el[(c, int(o))] for c in ISO3 if c != g and np.isfinite(el[(c, int(o))])]
            eps = float(np.median(others))
            h = min(HMAX, int(YEARS[-1] - o))
            pe = PC.project(ex[g], gdp[g], pop[g], o, h, eps)
            pp = phi_project(A[g]["phi"], o, h)
            i1 = int(o - YEARS[0]) + 1
            for j in range(h):
                r = {"g": g, "o": int(o), "h": j + 1,
                     "ex": float(ex[g][i1 + j]), "phi": float(A[g]["phi"][i1 + j]),
                     "imp": float(A[g]["imp"][i1 + j])}
                for k, v in pe.items():
                    r["ex_" + k] = float(v[j])
                for k, v in pp.items():
                    r["phi_" + k] = float(v[j])
                rec.append(r)
    return rec


def cover_boot(flags_by_country, seed=1):
    """Coverage and 95% bootstrap interval, resampling countries."""
    gs = list(flags_by_country)
    allf = np.concatenate([flags_by_country[g] for g in gs])
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(N_BOOT):
        pick = rng.choice(len(gs), len(gs))
        bs.append(np.mean(np.concatenate([flags_by_country[gs[i]] for i in pick])))
    return float(np.mean(allf)), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def band_from(errs, a=ALPHA):
    return np.percentile(errs, [50 * (1 - a), 50, 50 * (1 + a)])


def evaluate(rec, exk="driver", phik="persist"):
    """Leave one country out scores for Ex, phi and IMP."""
    out = {}
    bands = ((1, 5), (6, 10), (11, 24))
    flags = {v: {b: {} for b in bands + ("all",)} for v in ("ex", "phi", "imp")}
    width = {v: [] for v in ("ex", "phi", "imp")}
    iscore = {v: [] for v in ("ex", "phi", "imp")}
    a_ = 1 - ALPHA

    def ints(lo_, hi_, y_):
        return (hi_ - lo_) + 2 / a_ * (lo_ - y_) * (y_ < lo_) + 2 / a_ * (y_ - hi_) * (y_ > hi_)
    for g in ISO3:
        other = [r for r in rec if r["g"] != g and ("ex_" + exk) in r]
        mine = [r for r in rec if r["g"] == g and ("ex_" + exk) in r]
        Eex, Ephi, pairs = {}, {}, {}
        for r in other:
            e1 = np.log(r["ex"] / r["ex_" + exk])
            e2 = r["phi"] - r["phi_" + phik]
            Eex.setdefault(r["h"], []).append(e1)
            Ephi.setdefault(r["h"], []).append(e2)
            pairs.setdefault(r["h"], []).append((e1, e2))
        Qex = PC.smooth_quantiles(Eex, ALPHA)
        Qphi = PC.smooth_quantiles(Ephi, ALPHA)
        for r in mine:
            h = r["h"]
            lo, _, hi = Qex(h)
            f_ex = r["ex_" + exk] * np.exp(lo) <= r["ex"] <= r["ex_" + exk] * np.exp(hi)
            lo2, _, hi2 = Qphi(h)
            f_phi = r["phi_" + phik] + lo2 <= r["phi"] <= r["phi_" + phik] + hi2
            P = np.array([p for hh in (h - 1, h, h + 1) for p in pairs.get(hh, [])])
            ex_s = r["ex_" + exk] * np.exp(P[:, 0])
            phi_s = np.clip(r["phi_" + phik] + P[:, 1], 0, 1)
            imp_s = (1 - phi_s) ** 2 * ex_s
            ilo, _, ihi = band_from(imp_s)
            f_imp = ilo <= r["imp"] <= ihi
            width["ex"].append(np.exp(hi) - np.exp(lo))
            width["phi"].append(hi2 - lo2)
            width["imp"].append((ihi - ilo) / max(r["imp"], 1e-9))
            iscore["ex"].append(ints(lo, hi, np.log(r["ex"] / r["ex_" + exk])))
            iscore["phi"].append(100 * ints(r["phi_" + phik] + lo2, r["phi_" + phik] + hi2, r["phi"]))
            iscore["imp"].append(ints(ilo, ihi, r["imp"]) / max(r["imp"], 1e-9))
            for v, f in (("ex", f_ex), ("phi", f_phi), ("imp", f_imp)):
                key = next(b for b in bands if b[0] <= h <= b[1])
                flags[v][key].setdefault(g, []).append(f)
                flags[v]["all"].setdefault(g, []).append(f)
    for v in flags:
        out[v] = {}
        for key, fb in flags[v].items():
            fb = {g: np.array(x) for g, x in fb.items()}
            c, ci = cover_boot(fb)
            out[v]["all" if key == "all" else f"{key[0]}-{key[1]}"] = {"cover": c, "ci95": ci,
                                                                       "n": int(sum(len(x) for x in fb.values()))}
        out[v]["mean_relative_width"] = float(np.mean(width[v]))
        out[v]["interval_score"] = float(np.mean(iscore[v]))
    return out


def by_origin(rec, exk="driver"):
    """Coverage of the Ex band per origin year (checks dependence across overlapping forecasts)."""
    res = {}
    for o in ORIGINS[::5]:
        f = []
        for g in ISO3:
            other = [r for r in rec if r["g"] != g and ("ex_" + exk) in r]
            E = {}
            for r in other:
                E.setdefault(r["h"], []).append(np.log(r["ex"] / r["ex_" + exk]))
            Q = PC.smooth_quantiles(E, ALPHA)
            for r in rec:
                if r["g"] == g and r["o"] == o and ("ex_" + exk) in r:
                    lo, _, hi = Q(r["h"])
                    f.append(r["ex_" + exk] * np.exp(lo) <= r["ex"] <= r["ex_" + exk] * np.exp(hi))
        res[int(o)] = {"cover": float(np.mean(f)), "n": len(f)}
    return res


def main():
    A = accounts()
    _, gdp, pop = PC.load()
    rec = collect(A, gdp, pop)
    res = {"n_countries": len(ISO3), "n_records": len(rec),
           "phi_range_2024": [float(min(A[g]["phi"][-1] for g in ISO3)), float(max(A[g]["phi"][-1] for g in ISO3))]}
    for exk in ("drift", "driver", "oracle"):
        for phik in ("persist", "drift"):
            ev = evaluate(rec, exk, phik)
            res[f"{exk}|{phik}"] = ev
            print(f"Ex {exk:6s} phi {phik:7s} | cover90 Ex {ev['ex']['all']['cover']:.3f} "
                  f"[{ev['ex']['all']['ci95'][0]:.3f},{ev['ex']['all']['ci95'][1]:.3f}] "
                  f"phi {ev['phi']['all']['cover']:.3f} IMP {ev['imp']['all']['cover']:.3f} "
                  f"[{ev['imp']['all']['ci95'][0]:.3f},{ev['imp']['all']['ci95'][1]:.3f}] | "
                  f"IMP by h: {ev['imp']['1-5']['cover']:.2f} {ev['imp']['6-10']['cover']:.2f} {ev['imp']['11-24']['cover']:.2f} | "
                  f"width phi {ev['phi']['mean_relative_width']*100:.2f} pts | IS Ex {ev['ex']['interval_score']:.3f} "
                  f"phi {ev['phi']['interval_score']:.2f} IMP {ev['imp']['interval_score']:.3f}")
    res["by_origin_driver"] = by_origin(rec)
    print("Ex driver coverage by origin:", {o: round(v["cover"], 3) for o, v in res["by_origin_driver"].items()})
    res["accounts"] = {g: {k: A[g][k].tolist() for k in ("ex", "phi", "imp")} for g in ISO3}
    json.dump(res, open(OUTF, "w"), indent=1)
    print("written", OUTF)


if __name__ == "__main__":
    main()

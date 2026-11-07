"""
projection_calibration.py

First version of the headline method: calibrated uncertainty for point projections
of national transport energy and exergy use.

For every country, origin year and horizon we make a point projection of total
transport exergy input with three projectors:
  P1 drift      linear trend of the last 10 years (what a trend extrapolation gives)
  P2 driver     per capita exergy use grows with per capita income at an elasticity pooled
                from the other countries; income and population grow at their last 10 year
                rates (what a planner knows at the origin)
  P3 oracle     P2 with the income and population that actually happened (scenario bound)
The log error e = ln(observed / projected) is pooled by horizon. A country's interval
uses only the errors of the other countries (leave one country out), so every score is
out of sample. The interval is point x exp(q_alpha(e_h)), with quantiles smoothed
across horizons by a power law in h fitted to their spread.
Writes ../results/projection_calibration.json.
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUTF = os.path.join(HERE, "..", "results", "projection_calibration.json")

ISO3 = {"BE": "BEL", "BG": "BGR", "CZ": "CZE", "DK": "DNK", "DE": "DEU", "EE": "EST", "IE": "IRL",
        "EL": "GRC", "ES": "ESP", "FR": "FRA", "HR": "HRV", "IT": "ITA", "CY": "CYP", "LV": "LVA",
        "LT": "LTU", "LU": "LUX", "HU": "HUN", "MT": "MLT", "NL": "NLD", "AT": "AUT", "PL": "POL",
        "PT": "PRT", "RO": "ROU", "SI": "SVN", "SK": "SVK", "FI": "FIN", "SE": "SWE", "IS": "ISL",
        "NO": "NOR", "MK": "MKD", "AL": "ALB", "RS": "SRB", "TR": "TUR"}
# exergy to energy grade factors (Utlu and Hepbasli 2006, Table 1; electricity 1.00).
# Biofuels take the factor of the fossil fuel they replace; LPG takes that of gasoline.
GAMMA = {"O4652XR5210B": 0.99, "O4671XR5220B": 1.07, "O4661XR5230B": 0.99, "O4680": 0.99,
         "O4630": 0.99, "E7000": 1.00, "G3000": 0.93, "R5210B": 0.99, "R5220B": 1.07}
YEARS = np.arange(1990, 2025)
ORIGINS = np.arange(2000, 2020)
HMAX = 24
ALPHAS = (0.5, 0.9)


def load():
    d = json.load(open(os.path.join(DATA, "eurostat_nrg_bal_c_all_transport_1990_2024.json")))
    ids, size = d["id"], d["size"]
    idx = {k: d["dimension"][k]["category"]["index"] for k in ids}

    def val(**kw):
        p = 0
        for k, s in zip(ids, size):
            p = p * s + (idx[k][kw[k]] if k in kw else 0)
        return d["value"].get(str(p)) or 0.0

    ex = {}
    for g in ISO3:
        ex[g] = np.array([sum(GAMMA[s] * val(geo=g, nrg_bal=b, siec=s, time=str(y))
                              for b in idx["nrg_bal"] for s in GAMMA) for y in YEARS]) / 1000.0  # PJ
    wb = {}
    for ind in ("NY.GDP.MKTP.KD", "SP.POP.TOTL"):
        rows = json.load(open(os.path.join(DATA, f"wb_{ind}.json")))[1]
        tab = {}
        for r in rows:
            if r["value"] is not None:
                tab.setdefault(r["countryiso3code"], {})[int(r["date"])] = r["value"]
        wb[ind] = {g: np.array([tab.get(ISO3[g], {}).get(int(y), np.nan) for y in YEARS]) for g in ISO3}
    return ex, wb["NY.GDP.MKTP.KD"], wb["SP.POP.TOTL"]


def growth_path(x, i0, h, win=10):
    """Project x from index i0 by its mean log growth over the last win years."""
    seg = np.log(x[i0 - win:i0 + 1])
    gr = (seg[-1] - seg[0]) / win
    return np.exp(seg[-1] + gr * np.arange(1, h + 1))


def elasticity(ex, gdp, pop, origin, win=15):
    """Per capita income elasticity of exergy use, fitted on up to win years before origin."""
    i0 = int(origin - YEARS[0])
    sl = slice(max(0, i0 - win + 1), i0 + 1)
    x = np.log(gdp[sl] / pop[sl])
    y = np.log(ex[sl] / pop[sl])
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8 or np.ptp(x[ok]) < 0.05:
        return np.nan
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def project(ex, gdp, pop, origin, h, eps):
    """Point projections from the origin; eps is the pooled elasticity (other countries)."""
    i0 = int(origin - YEARS[0])
    out = {}
    y = np.log(ex[:i0 + 1])
    steps = np.arange(1, h + 1)
    # P1 drift on log exergy, last 10 years
    out["drift"] = np.exp(y[-1] + (y[-1] - y[-11]) / 10 * steps)
    if np.isfinite(eps) and np.all(np.isfinite([gdp[i0], pop[i0], gdp[i0 - 10], pop[i0 - 10]])):
        gpc = np.log(gdp / pop)
        lp = np.log(pop)
        # P2 planner: per capita income and population grow at their last 10 year rates
        d_gpc = (gpc[i0] - gpc[i0 - 10]) / 10 * steps
        d_lp = (lp[i0] - lp[i0 - 10]) / 10 * steps
        out["driver"] = np.exp(y[-1] + eps * d_gpc + d_lp)
        # P3 oracle: the income and population that actually happened
        o_gpc = gpc[i0 + 1:i0 + 1 + h] - gpc[i0]
        o_lp = lp[i0 + 1:i0 + 1 + h] - lp[i0]
        if np.all(np.isfinite(o_gpc)) and np.all(np.isfinite(o_lp)):
            out["oracle"] = np.exp(y[-1] + eps * o_gpc + o_lp)
    return out


def errors(ex, gdp, pop):
    """Rows (country, origin, h, projector, log error). Elasticities pooled leave one country out."""
    el = {(g, int(o)): elasticity(ex[g], gdp[g], pop[g], o) for g in ISO3 for o in ORIGINS}
    rows = []
    for g in ISO3:
        for o in ORIGINS:
            others = [el[(c, int(o))] for c in ISO3 if c != g and np.isfinite(el[(c, int(o))])]
            eps = float(np.median(others)) if others else np.nan
            h = min(HMAX, int(YEARS[-1] - o))
            pr = project(ex[g], gdp[g], pop[g], o, h, eps)
            obs = ex[g][int(o - YEARS[0]) + 1:int(o - YEARS[0]) + 1 + h]
            for k in ("drift", "driver", "oracle"):
                if k in pr:
                    e = np.log(obs / pr[k])
                    for j in range(h):
                        if np.isfinite(e[j]):
                            rows.append((g, int(o), j + 1, k, float(e[j])))
    return rows, el


def band(errs_h, a):
    lo, hi = np.percentile(errs_h, [50 * (1 - a), 50 * (1 + a)])
    return lo, hi


def smooth_quantiles(E, a):
    """Quantiles of pooled log error by horizon, smoothed as m(h) + c h^p (separate c, p per side)."""
    hs = np.arange(1, HMAX + 1)
    lo = np.full(HMAX, np.nan)
    hi = np.full(HMAX, np.nan)
    med = np.full(HMAX, np.nan)
    for h in hs:
        e = E.get(h, [])
        if len(e) >= 20:
            lo[h - 1], hi[h - 1] = band(e, a)
            med[h - 1] = np.median(e)
    m = np.isfinite(lo)
    b_med = np.sum(med[m] * hs[m]) / np.sum(hs[m] ** 2)
    fits = []
    for side in (med[m] - lo[m], hi[m] - med[m]):
        p, logc = np.polyfit(np.log(hs[m]), np.log(np.maximum(side, 1e-6)), 1)
        fits.append((np.exp(logc), p))
    (cl, pl), (ch, ph) = fits
    return lambda h: (b_med * h - cl * h ** pl, b_med * h, b_med * h + ch * h ** ph)


def main():
    ex, gdp, pop = load()
    rows, el = errors(ex, gdp, pop)
    res = {"n_countries": len(ISO3), "n_errors": len(rows), "projectors": {}}
    for k in ("drift", "driver", "oracle"):
        R = [r for r in rows if r[3] == k]
        rep = {"raw_mae_log_by_h": {}}
        for h in (1, 5, 10, 15, 20):
            e = [r[4] for r in R if r[2] == h]
            rep["raw_mae_log_by_h"][h] = float(np.mean(np.abs(e))) if e else None
        # leave one country out coverage
        cov = {a: {"all": [], "h1_5": [], "h6_10": [], "h11_24": []} for a in ALPHAS}
        for g in ISO3:
            E = {}
            for r in R:
                if r[0] != g:
                    E.setdefault(r[2], []).append(r[4])
            Q = {a: smooth_quantiles(E, a) for a in ALPHAS}
            for r in R:
                if r[0] == g:
                    for a in ALPHAS:
                        lo, _, hi = Q[a](r[2])
                        inside = lo <= r[4] <= hi
                        cov[a]["all"].append(inside)
                        key = "h1_5" if r[2] <= 5 else ("h6_10" if r[2] <= 10 else "h11_24")
                        cov[a][key].append(inside)
        rep["loco_coverage"] = {str(a): {kk: float(np.mean(v)) for kk, v in c.items()} for a, c in cov.items()}
        E = {}
        for r in R:
            E.setdefault(r[2], []).append(r[4])
        Q = smooth_quantiles(E, 0.9)
        rep["pooled_90_band_h20"] = [float(np.exp(x)) for x in Q(20)]
        res["projectors"][k] = rep
        c = rep["loco_coverage"]
        print(f"{k:7s} MAE(log) h1 {rep['raw_mae_log_by_h'][1]:.3f} h10 {rep['raw_mae_log_by_h'][10]:.3f} "
              f"h20 {rep['raw_mae_log_by_h'][20]:.3f} | LOCO cover90 all {c['0.9']['all']:.3f} "
              f"(1-5 {c['0.9']['h1_5']:.3f}, 6-10 {c['0.9']['h6_10']:.3f}, 11-24 {c['0.9']['h11_24']:.3f}) "
              f"cover50 {c['0.5']['all']:.3f} | h20 90% band x{rep['pooled_90_band_h20'][0]:.2f} to x{rep['pooled_90_band_h20'][2]:.2f}")

    # Turkiye: calibrated band around a point projection made in 2000, other countries only
    g = "TR"
    E = {}
    for r in rows:
        if r[0] != g and r[3] == "driver":
            E.setdefault(r[2], []).append(r[4])
    Q = smooth_quantiles(E, 0.9)
    lo, md, hi = (float(np.exp(x)) for x in Q(20))
    uh = 1385.73  # Utlu and Hepbasli (2006) exergy input 2020, a point projection made from data to 2000
    obs20 = float(ex[g][YEARS == 2020][0])
    obs19 = float(ex[g][YEARS == 2019][0])
    res["turkiye_uh2006"] = {"point_2020": uh, "band90": [uh * lo, uh * hi], "median_adj": uh * md,
                             "observed_2019": obs19, "observed_2020": obs20}
    print(f"TR: UH 2020 point {uh:.0f} PJ -> calibrated 90% band {uh*lo:.0f} to {uh*hi:.0f} PJ "
          f"(observed 2019 {obs19:.0f}, 2020 {obs20:.0f})")
    json.dump(res, open(OUTF, "w"), indent=1)


if __name__ == "__main__":
    main()

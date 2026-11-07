"""
pipeline.py

Single reproducible pipeline for the Thermo manuscript
"Calibrated Uncertainty Quantification for Sectoral Exergy Analysis of
Transport Using Probabilistic Machine Learning".

Steps
 1. Load the reconstructed subsector fuel series (1988 to 2019 used).
 2. Rolling origin backtests: unweighted quadratic, drift, hybrid model.
 3. Interval calibration (kappa), pooled and leave one cutoff out.
 4. Final calibrated forecasts, 2020 to 2040.
 5. Monte Carlo propagation to emissions and improvement potential,
    with variance attribution and an efficiency sensitivity scenario.
 6. One parameter indicator family, reproduction checks.

All results are written to results.json; figures are produced by
make_figures.py from that file only.

Run: python pipeline.py
"""

import json
import os
import warnings

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))

SERIES = ["Road_PMS", "Road_AGO", "Water_FO", "Water_AGO", "Air_AVK", "Rail_FO", "Rail_AGO"]
SERIES_FUEL = {"Road_PMS": "gasoline", "Road_AGO": "diesel", "Water_FO": "fuel_oil",
               "Water_AGO": "diesel", "Air_AVK": "aviation", "Rail_FO": "fuel_oil",
               "Rail_AGO": "diesel"}
SUBSECTOR_SERIES = {"road": ["Road_PMS", "Road_AGO"], "water": ["Water_FO", "Water_AGO"],
                    "air": ["Air_AVK"], "rail": ["Rail_FO", "Rail_AGO"]}

# Fuel properties (heating value kJ/kg, chemical exergy kJ/kg, grade function),
# after Rosen and Dincer (1997).
GAMMA = {"gasoline": 0.99, "fuel_oil": 0.99, "diesel": 1.07, "aviation": 1.00}

# Adopted emission factors: CO2 in kg/GJ; SO2, NOx, CO in g/GJ (Motasemi et al. 2014).
EF = {
    "gasoline": {"CO2": 72.79, "SO2": 0.46, "NOx": 111.72, "CO": 2225.97},
    "fuel_oil": {"CO2": 73.99, "SO2": 0.47, "NOx": 257.64, "CO": 97.20},
    "diesel":   {"CO2": 73.10, "SO2": 22.99, "NOx": 859.00, "CO": 6972.00},
    "aviation": {"CO2": 73.99, "SO2": 0.47, "NOx": 257.64, "CO": 97.20},
}
# IPCC 2006, Vol. 2, Ch. 1, Table 1.4: default, lower, upper (95% CI), kg/GJ.
IPCC = {"gasoline": (69.3, 67.5, 73.0), "diesel": (74.1, 72.6, 74.8),
        "fuel_oil": (77.4, 75.5, 78.8), "aviation": (71.5, 69.7, 74.4)}

# Relative standard uncertainties used in the Monte Carlo (Section 2.8).
U_EF = {"CO2": 0.05, "NOx": 0.30, "CO": 0.40, "SO2": 0.25}
U_GAMMA = 0.02

# 1999 to 2000 exergy flows by subsector (input, product, destruction), PJ.
EXERGY_FLOWS = {"road": (565.38, 121.22, 444.18), "water": (5.165, 0.775, 4.39),
                "air": (10.09, 2.79, 7.29), "rail": (0.778, 0.117, 0.66)}
ENERGY_FLOWS = {"road": (565.39, 121.44, 442.95), "water": (5.165, 0.783, 4.43),
                "air": (10.09, 2.82, 7.26), "rail": (0.778, 0.118, 0.67)}
STATED_EFF = {"road": (21.48, 21.44), "water": (15.12, 15.00),
              "air": (27.95, 27.73), "rail": (15.17, 15.04)}

TAU = 12.0
T_TRAIN_END = 2019
SHORT_CUTOFFS = [2004, 2009, 2014]
SHORT_H = 5
LONG_SPLITS = [(2003, 2019), (2007, 2019)]
KAPPA_GRID = np.round(np.arange(0.5, 3.0001, 0.05), 2)
NOMINAL = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
N_BT = 1000
N_FC = 4000


# ----------------------------------------------------------------- data
def load_data():
    with open(os.path.join(HERE, "digitized_fuel_consumption.json")) as f:
        d = json.load(f)
    years = np.array([y for y in range(1988, T_TRAIN_END + 1)])
    Y = {s: np.array([d[str(y)][s] for y in years], dtype=float) for s in SERIES}
    # Rail FO points where the marker could not be separated from the
    # overlapping Rail AGO line were digitized as exact zeros: treat as missing.
    Y["Rail_FO"][Y["Rail_FO"] == 0.0] = np.nan
    return years.astype(float), Y


# ---------------------------------------------------------------- models
def wls_quadratic(t, y):
    t0, tmax = t.min(), t.max()
    w = 0.5 ** ((tmax - t) / TAU)
    x = t - t0
    X = np.column_stack([np.ones_like(x), x, x ** 2])
    XtW = X.T * w
    A = np.linalg.inv(XtW @ X)
    beta = A @ XtW @ y
    r = y - X @ beta
    sigma2 = (w * r ** 2).sum() / max(len(y) - 3, 1)
    return beta, sigma2 * A, t0


def design(tq, t0):
    x = tq - t0
    return np.column_stack([np.ones_like(x), x, x ** 2])


def hybrid_draws(t, y, tq, n, seed):
    m = ~np.isnan(y)
    t, y = t[m], y[m]
    beta, S, t0 = wls_quadratic(t, y)
    res = y - design(t, t0) @ beta
    kern = ConstantKernel(1.0, (1e-4, 1e6)) * RBF(5.0, (0.5, 100.0)) + WhiteKernel(1.0, (1e-4, 1e6))
    gp = GaussianProcessRegressor(kernel=kern, normalize_y=False, n_restarts_optimizer=5,
                                  random_state=seed)
    gp.fit(t.reshape(-1, 1), res)
    rng = np.random.default_rng(seed)
    B = rng.multivariate_normal(beta, S, size=n)
    trend = B @ design(tq, t0).T
    resid = gp.sample_y(tq.reshape(-1, 1), n_samples=n, random_state=seed + 1).T
    return np.maximum(0.0, trend + resid)


def quad_point(t, y, tq):
    m = ~np.isnan(y)
    return np.polyval(np.polyfit(t[m], y[m], 2), tq)


def drift_point(t, y, tq):
    m = ~np.isnan(y)
    t, y = t[m], y[m]
    return y[0] + (y[-1] - y[0]) / (t[-1] - t[0]) * (tq - t[0])


# ------------------------------------------------------------- backtests
def run_fold(t, Y, cutoff, end, seed):
    rows = []
    for k, s in enumerate(SERIES):
        y = Y[s]
        tr = t <= cutoff
        te = (t > cutoff) & (t <= end) & ~np.isnan(y)
        tt, yt = t[te], y[te]
        D = hybrid_draws(t[tr], y[tr], tt, N_BT, seed + 97 * k)
        med = np.median(D, axis=0)
        yt_tr = y[tr][~np.isnan(y[tr])]
        rows.append({
            "series": s, "cutoff": cutoff, "years": tt.tolist(), "y": yt.tolist(),
            "quad": quad_point(t[tr], y[tr], tt).tolist(),
            "drift": drift_point(t[tr], y[tr], tt).tolist(),
            "med": med.tolist(),
            "q": {str(a): [np.percentile(D, 50 * (1 - a), axis=0).tolist(),
                           np.percentile(D, 50 * (1 + a), axis=0).tolist()] for a in NOMINAL},
            "R_train": float(yt_tr.max() - yt_tr.min()),
        })
    return rows


def flatten(rows, key):
    return np.concatenate([np.asarray(r[key], float) for r in rows])


def interval(rows, a, kappa):
    med = flatten(rows, "med")
    lo = np.concatenate([np.asarray(r["q"][str(a)][0]) for r in rows])
    hi = np.concatenate([np.asarray(r["q"][str(a)][1]) for r in rows])
    return med - kappa * (med - lo), med + kappa * (hi - med)


def coverage(rows, a, kappa):
    y = flatten(rows, "y")
    L, U = interval(rows, a, kappa)
    return float(np.mean((y >= L) & (y <= U)))


def pinaw(rows, a, kappa):
    L, U = interval(rows, a, kappa)
    R = np.concatenate([np.full(len(r["y"]), r["R_train"]) for r in rows])
    return float(np.mean((U - L) / R))


def best_kappa(rows, a=0.9):
    covs = np.array([coverage(rows, a, k) for k in KAPPA_GRID])
    i = int(np.argmin(np.abs(covs - a)))
    return float(KAPPA_GRID[i]), covs


def mae(rows, key):
    return float(np.mean(np.abs(flatten(rows, "y") - flatten(rows, key))))


def mae_by_series_mean(rows, key):
    # mean over series of per series MAE (equal weight per series and fold)
    return float(np.mean([np.mean(np.abs(np.asarray(r["y"]) - np.asarray(r[key]))) for r in rows]))


# ------------------------------------------------------------------ main
def main():
    t, Y = load_data()
    out = {}

    # ---- reproduction of 1999 to 2000 efficiencies from the flows
    repro = {}
    for s in EXERGY_FLOWS:
        en = ENERGY_FLOWS[s][1] / ENERGY_FLOWS[s][0] * 100
        ex = EXERGY_FLOWS[s][1] / EXERGY_FLOWS[s][0] * 100
        cl_en = (ENERGY_FLOWS[s][1] + ENERGY_FLOWS[s][2] - ENERGY_FLOWS[s][0]) / ENERGY_FLOWS[s][0] * 100
        cl_ex = (EXERGY_FLOWS[s][1] + EXERGY_FLOWS[s][2] - EXERGY_FLOWS[s][0]) / EXERGY_FLOWS[s][0] * 100
        repro[s] = {"energy_eff": en, "exergy_eff": ex, "stated_energy": STATED_EFF[s][0],
                    "stated_exergy": STATED_EFF[s][1], "closure_energy_pct": cl_en,
                    "closure_exergy_pct": cl_ex}
    out["reproduction"] = repro
    phi = {s: EXERGY_FLOWS[s][1] / EXERGY_FLOWS[s][0] for s in EXERGY_FLOWS}
    dest = {s: EXERGY_FLOWS[s][2] for s in EXERGY_FLOWS}
    ri = {s: dest[s] / sum(dest.values()) for s in dest}
    ind = {}
    for s, p in phi.items():
        ind[s] = {"phi": p, "DN_WER_FIE": 1 - p, "ESI": 1 / (1 - p), "LOP_EEF": (1 - p) / p,
                  "CIE": 1 / (100 * p), "EPI": (1 - p) + 1 / (100 * p),
                  "IMP_ratio": (1 - p) ** 2, "RI": ri[s]}
    out["phi"] = phi
    out["indicators"] = ind

    # ---- emission factors versus IPCC
    out["ipcc"] = {f: {"adopted": EF[f]["CO2"], "default": IPCC[f][0], "lower": IPCC[f][1],
                       "upper": IPCC[f][2],
                       "dev_pct": (EF[f]["CO2"] - IPCC[f][0]) / IPCC[f][0] * 100,
                       "inside_ci": IPCC[f][1] <= EF[f]["CO2"] <= IPCC[f][2],
                       "ipcc_rel_sd_pct": (IPCC[f][2] - IPCC[f][1]) / 3.92 / IPCC[f][0] * 100}
                   for f in IPCC}

    # ---- historical data snapshot
    out["history"] = {s: {"years": t.tolist(), "y": [None if np.isnan(v) else float(v) for v in Y[s]]}
                      for s in SERIES}

    # ---- backtests
    print("short horizon backtests ...")
    short = []
    for j, c in enumerate(SHORT_CUTOFFS):
        short += run_fold(t, Y, c, c + SHORT_H, seed=1000 + 10 * j)
    print("long horizon backtests ...")
    long_ = {}
    for j, (c, e) in enumerate(LONG_SPLITS):
        long_[f"{c}"] = run_fold(t, Y, c, e, seed=2000 + 10 * j)

    bt = {"short": {k: mae_by_series_mean(short, k) for k in ["quad", "drift", "med"]}}
    for c, rows in long_.items():
        bt[f"long_{c}"] = {k: mae_by_series_mean(rows, k) for k in ["quad", "drift", "med"]}
    # per fold, per series detail for the short horizon
    bt["short_by_cutoff"] = {str(c): {k: mae_by_series_mean([r for r in short if r["cutoff"] == c], k)
                                      for k in ["quad", "drift", "med"]} for c in SHORT_CUTOFFS}
    # win counts (hybrid vs each baseline) across series x folds, short horizon
    def per(r, k):
        return np.mean(np.abs(np.asarray(r["y"]) - np.asarray(r[k])))
    bt["short_hybrid_beats_quad"] = int(sum(per(r, "med") < per(r, "quad") for r in short))
    bt["short_hybrid_beats_drift"] = int(sum(per(r, "med") < per(r, "drift") for r in short))
    bt["short_n_cases"] = len(short)
    out["backtest"] = bt

    # ---- calibration
    k_star, covs = best_kappa(short, 0.9)
    cal = {"kappa_grid": KAPPA_GRID.tolist(), "coverage_grid": covs.tolist(),
           "pinaw_grid": [pinaw(short, 0.9, k) for k in KAPPA_GRID],
           "kappa_star": k_star, "picp_raw": coverage(short, 0.9, 1.0),
           "picp_cal": coverage(short, 0.9, k_star), "pinaw_raw": pinaw(short, 0.9, 1.0),
           "pinaw_cal": pinaw(short, 0.9, k_star), "N": int(len(flatten(short, "y")))}
    # leave one cutoff out
    loco = []
    y_all, inside_all, width_all = [], [], []
    for c in SHORT_CUTOFFS:
        fit = [r for r in short if r["cutoff"] != c]
        test = [r for r in short if r["cutoff"] == c]
        kc, _ = best_kappa(fit, 0.9)
        cv = coverage(test, 0.9, kc)
        loco.append({"held_out": c, "kappa": kc, "picp_test": cv, "n": int(len(flatten(test, "y")))})
        y = flatten(test, "y")
        L, U = interval(test, 0.9, kc)
        inside_all.append((y >= L) & (y <= U))
    cal["loco"] = loco
    cal["picp_loco_pooled"] = float(np.mean(np.concatenate(inside_all)))
    # transfer to long horizon folds
    cal["long_transfer"] = {c: {"picp_raw": coverage(rows, 0.9, 1.0),
                                "picp_cal": coverage(rows, 0.9, k_star),
                                "pinaw_cal": pinaw(rows, 0.9, k_star),
                                "n": int(len(flatten(rows, "y")))} for c, rows in long_.items()}
    # reliability curve
    cal["reliability"] = {"nominal": NOMINAL,
                          "raw": [coverage(short, a, 1.0) for a in NOMINAL],
                          "cal": [coverage(short, a, k_star) for a in NOMINAL]}
    # per series diagnostics (short horizon)
    cal["by_series"] = {}
    for s in SERIES:
        rows = [r for r in short if r["series"] == s]
        cal["by_series"][s] = {"mae_hybrid": mae(rows, "med"), "mae_drift": mae(rows, "drift"),
                               "mae_quad": mae(rows, "quad"), "picp_raw": coverage(rows, 0.9, 1.0),
                               "picp_cal": coverage(rows, 0.9, k_star),
                               "pinaw_cal": pinaw(rows, 0.9, k_star)}
    out["calibration"] = cal
    print("kappa*", k_star, "PICP raw", cal["picp_raw"], "cal", cal["picp_cal"],
          "LOCO", cal["picp_loco_pooled"], "PINAW", cal["pinaw_cal"])

    # ---- final calibrated forecasts 2020 to 2040
    print("final forecasts ...")
    tq = np.arange(2020, 2041, dtype=float)
    FC = {}
    fc_sum = {}
    for k, s in enumerate(SERIES):
        D = hybrid_draws(t, Y[s], tq, N_FC, seed=5000 + 31 * k)
        med = np.median(D, axis=0)
        Dc = np.maximum(0.0, med + k_star * (D - med))
        FC[s] = Dc
        fc_sum[s] = {"p5": np.percentile(Dc, 5, axis=0).tolist(),
                     "p25": np.percentile(Dc, 25, axis=0).tolist(),
                     "p50": np.percentile(Dc, 50, axis=0).tolist(),
                     "p75": np.percentile(Dc, 75, axis=0).tolist(),
                     "p95": np.percentile(Dc, 95, axis=0).tolist(),
                     "quad_2040": float(quad_point(t, Y[s], np.array([2040.0]))[0]),
                     "y2019": float(Y[s][-1]) if not np.isnan(Y[s][-1]) else None}
    total = sum(FC[s] for s in SERIES)
    fc_sum["TOTAL"] = {p: np.percentile(total, q, axis=0).tolist()
                       for p, q in [("p5", 5), ("p10", 10), ("p25", 25), ("p50", 50),
                                    ("p75", 75), ("p90", 90), ("p95", 95)]}
    hist_total = np.nansum(np.vstack([Y[s] for s in SERIES]), axis=0)
    fc_sum["TOTAL"]["hist"] = hist_total.tolist()
    fc_sum["years"] = tq.tolist()
    out["forecast"] = fc_sum

    # ---- Monte Carlo propagation
    print("monte carlo ...")
    rng = np.random.default_rng(777)
    fuels = list(EF)
    pol = ["CO2", "SO2", "NOx", "CO"]
    Zef = {(f, p): rng.standard_normal(N_FC) for f in fuels for p in pol}
    Zg = {f: rng.standard_normal(N_FC) for f in fuels}

    def fuel_demand(fc):
        dmd = {f: np.zeros((N_FC, len(tq))) for f in fuels}
        for s in SERIES:
            dmd[SERIES_FUEL[s]] += fc[s]
        return dmd

    dmd = fuel_demand(FC)
    dmd_med = {f: np.broadcast_to(np.median(dmd[f], axis=0), dmd[f].shape) for f in fuels}

    def emis(d, p, ef_unc=True):
        tot = np.zeros((N_FC, len(tq)))
        for f in fuels:
            base = EF[f][p] / (1.0 if p == "CO2" else 1000.0)  # kg/GJ
            e = base * (1 + U_EF[p] * Zef[(f, p)]) if ef_unc else np.full(N_FC, base)
            e = np.clip(e, 0, None)[:, None]
            tot += d[f] * 1e6 * e / 1e9  # PJ -> GJ, kg -> Mt
        return tot

    em = {}
    attrib = {}
    for p in pol:
        full = emis(dmd, p, True)
        em[p] = {q: np.percentile(full, v, axis=0).tolist()
                 for q, v in [("p5", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95)]}
        v_full = np.var(full[:, -1])
        v_dem = np.var(emis(dmd, p, False)[:, -1])
        v_ef = np.var(emis(dmd_med, p, True)[:, -1])
        attrib[p] = {"demand": v_dem / (v_dem + v_ef), "factor": v_ef / (v_dem + v_ef),
                     "cv_2040": float(np.std(full[:, -1]) / np.mean(full[:, -1]))}
    # historical deterministic emissions (for context)
    hist_em = {}
    for p in pol:
        tot = np.zeros(len(t))
        for s in SERIES:
            f = SERIES_FUEL[s]
            base = EF[f][p] / (1.0 if p == "CO2" else 1000.0)
            tot += np.nan_to_num(Y[s]) * 1e6 * base / 1e9
        hist_em[p] = tot.tolist()
    out["emissions"] = {"forecast": em, "history": hist_em, "attribution_2040": attrib}

    # improvement potential
    imp = {}
    imp_draws = {}
    imp_draws_nog = {}
    for sub, ss in SUBSECTOR_SERIES.items():
        ex = np.zeros((N_FC, len(tq)))
        ex_ng = np.zeros((N_FC, len(tq)))
        for s in ss:
            g = GAMMA[SERIES_FUEL[s]]
            ex += FC[s] * (g * (1 + U_GAMMA * Zg[SERIES_FUEL[s]]))[:, None]
            ex_ng += FC[s] * g
        imp_draws[sub] = (1 - phi[sub]) ** 2 * ex
        imp_draws_nog[sub] = (1 - phi[sub]) ** 2 * ex_ng
        imp[sub] = {q: np.percentile(imp_draws[sub], v, axis=0).tolist()
                    for q, v in [("p5", 5), ("p50", 50), ("p95", 95)]}
        imp[sub]["exin_p50_2040"] = float(np.median(ex[:, -1]))
    tot_imp = sum(imp_draws.values())
    imp["TOTAL"] = {q: np.percentile(tot_imp, v, axis=0).tolist()
                    for q, v in [("p5", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95)]}
    share = {sub: float(np.median(imp_draws[sub][:, -1] / tot_imp[:, -1])) for sub in imp_draws}
    imp["share_2040"] = share
    v_full = np.var(tot_imp[:, -1])
    v_ng = np.var(sum(imp_draws_nog.values())[:, -1])
    imp["gamma_var_share_2040"] = float(max(v_full - v_ng, 0) / v_full)
    # efficiency improvement scenario: dphi = 0..0.10 absolute
    dphi = np.round(np.arange(0, 0.1001, 0.01), 2)
    scen = {}
    for sub in SUBSECTOR_SERIES:
        exm = imp[sub]["exin_p50_2040"]
        scen[sub] = [float((1 - phi[sub] - d) ** 2 * exm) for d in dphi]
    imp["dphi_grid"] = dphi.tolist()
    imp["dphi_scenario_2040"] = scen
    # local leverage dIMP/dphi per percentage point, 2040 median input
    imp["leverage_PJ_per_pp"] = {sub: float(2 * (1 - phi[sub]) * imp[sub]["exin_p50_2040"] / 100)
                                 for sub in SUBSECTOR_SERIES}
    out["imp"] = imp

    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved results.json")


if __name__ == "__main__":
    main()

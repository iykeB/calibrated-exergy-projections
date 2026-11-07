"""
revision.py

Round 1 revision analyses for thermo-4658922. Builds on pipeline.py and
writes revision_results.json. Every number quoted in the revised manuscript
and the response letter comes from this file.

  A  trend covariance: weighted LS (original) versus sandwich with GP covariance (R2.6)
  B  probabilistic drift baseline, calibrated the same way (R2.2)
  C  sharpness and interval score at matched coverage (R1)
  D  calibration by forecast horizon (R2.1)
  E  external validation against NBS PMS truck-out, 2020 to 2023 (R1, R2.4)
  F  training extended to 2023 with the NBS index (R2.4)
  G  structural change scenarios to 2040 (R1, R2.4)
  H  exergy efficiency sensitivity (R2.3)
  I  demand correlation and variance interaction (R2.5)
"""

import json
import os
import warnings

import numpy as np
from scipy.stats import norm, rankdata
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

import pipeline as P

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
NBS = json.load(open(os.path.join(HERE, "nbs_series.json")))
TO = {int(k): v for k, v in NBS["pms_truckout_L"].items()}
NOM = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
OUT = {}


# ------------------------------------------------------------------ models
def gp_fit(t, r, seed):
    kern = ConstantKernel(1.0, (1e-4, 1e6)) * RBF(5.0, (0.5, 100.0)) + WhiteKernel(1.0, (1e-4, 1e6))
    gp = GaussianProcessRegressor(kernel=kern, normalize_y=False, n_restarts_optimizer=5, random_state=seed)
    gp.fit(t.reshape(-1, 1), r)
    return gp


def hybrid_draws(t, y, tq, n, seed, cov="sandwich"):
    """Hybrid forecast draws. cov='wls' reproduces the submitted model;
    cov='sandwich' uses Sigma = A X'W K W X A with K the fitted GP covariance."""
    m = ~np.isnan(y)
    t, y = t[m], y[m]
    t0, tmax = t.min(), t.max()
    w = 0.5 ** ((tmax - t) / P.TAU)
    X = P.design(t, t0)
    XtW = X.T * w
    A = np.linalg.inv(XtW @ X)
    beta = A @ XtW @ y
    res = y - X @ beta
    gp = gp_fit(t, res, seed)
    if cov == "wls":
        S = (w * res ** 2).sum() / max(len(y) - 3, 1) * A
    else:
        K = gp.kernel_(t.reshape(-1, 1))
        S = A @ (XtW @ K @ XtW.T) @ A
        S = 0.5 * (S + S.T)
    rng = np.random.default_rng(seed)
    B = rng.multivariate_normal(beta, S, size=n)
    trend = B @ P.design(tq, t0).T
    resid = gp.sample_y(tq.reshape(-1, 1), n_samples=n, random_state=seed + 1).T
    return np.maximum(0.0, trend + resid)


def drift_draws(t, y, tq, n, seed):
    """Probabilistic drift (random walk with drift), Hyndman and Athanasopoulos:
    sigma_h = sigma * sqrt(h (1 + h / (T - 1)))."""
    m = ~np.isnan(y)
    t, y = t[m], y[m]
    T = len(y)
    c = (y[-1] - y[0]) / (t[-1] - t[0])
    d = np.diff(y) / np.diff(t)
    e = d - c
    sig = np.sqrt((e ** 2).sum() / max(T - 2, 1))
    h = tq - t[-1]
    mu = y[-1] + c * h
    sd = sig * np.sqrt(h * (1 + h / (T - 1)))
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, len(tq)))
    return np.maximum(0.0, mu + sd * Z)


def combine(Dh, Dd, seed):
    """Equal weight mixture of the two predictive distributions."""
    rng = np.random.default_rng(seed)
    pick = rng.random(Dh.shape[0]) < 0.5
    return np.where(pick[:, None], Dh, Dd)


def model_draws(t, y, tq, n, seed, model="combined"):
    if model == "hybrid":
        return hybrid_draws(t, y, tq, n, seed, "sandwich")
    if model == "drift":
        return drift_draws(t, y, tq, n, seed)
    return combine(hybrid_draws(t, y, tq, n, seed, "sandwich"), drift_draws(t, y, tq, n, seed), seed + 3)


# ------------------------------------------------------------------ folds
def fold(t, Y, cutoff, end, seed, model, cov="sandwich", n=1000):
    rows = []
    for k, s in enumerate(P.SERIES):
        y = Y[s]
        tr = t <= cutoff
        te = (t > cutoff) & (t <= end) & ~np.isnan(y)
        tt, yt = t[te], y[te]
        if model == "hybrid":
            D = hybrid_draws(t[tr], y[tr], tt, n, seed + 97 * k, cov)
        elif model == "drift":
            D = drift_draws(t[tr], y[tr], tt, n, seed + 97 * k)
        else:
            D = combine(hybrid_draws(t[tr], y[tr], tt, n, seed + 97 * k, cov),
                        drift_draws(t[tr], y[tr], tt, n, seed + 97 * k), seed + 3)
        ytr = y[tr][~np.isnan(y[tr])]
        rows.append({"series": s, "cutoff": cutoff, "h": (tt - cutoff).tolist(), "y": yt.tolist(),
                     "D": D, "R": float(ytr.max() - ytr.min())})
    return rows


def run_all(t, Y, model, cov="sandwich"):
    short = []
    for j, c in enumerate(P.SHORT_CUTOFFS):
        short += fold(t, Y, c, c + P.SHORT_H, 1000 + 10 * j, model, cov)
    long_ = []
    for j, (c, e) in enumerate(P.LONG_SPLITS):
        long_ += fold(t, Y, c, e, 2000 + 10 * j, model, cov)
    return short, long_


def stack(rows, key):
    return np.concatenate([np.asarray(r[key], float) for r in rows])


def cal_interval(rows, a, k):
    L, U, M = [], [], []
    for r in rows:
        D = r["D"]
        med = np.median(D, axis=0)
        lo = np.percentile(D, 50 * (1 - a), axis=0)
        hi = np.percentile(D, 50 * (1 + a), axis=0)
        L.append(med - k * (med - lo))
        U.append(med + k * (hi - med))
        M.append(med)
    return np.concatenate(L), np.concatenate(U), np.concatenate(M)


def picp(rows, a, k):
    y = stack(rows, "y")
    L, U, _ = cal_interval(rows, a, k)
    return float(np.mean((y >= L) & (y <= U)))


def pinaw(rows, a, k):
    L, U, _ = cal_interval(rows, a, k)
    R = np.concatenate([np.full(len(r["y"]), r["R"]) for r in rows])
    return float(np.mean((U - L) / R))


def interval_score(rows, a, k):
    """Winkler interval score for the central a interval, scaled by R (lower is better)."""
    y = stack(rows, "y")
    L, U, _ = cal_interval(rows, a, k)
    al = 1 - a
    s = (U - L) + 2 / al * (L - y) * (y < L) + 2 / al * (y - U) * (y > U)
    R = np.concatenate([np.full(len(r["y"]), r["R"]) for r in rows])
    return float(np.mean(s / R))


def crps(rows, k):
    """Sample CRPS of the calibrated predictive distribution, scaled by R."""
    out = []
    for r in rows:
        D = r["D"]
        med = np.median(D, axis=0)
        Dk = np.maximum(0, med + k * (D - med))
        for j, yv in enumerate(r["y"]):
            x = Dk[:400, j]
            out.append((np.mean(np.abs(x - yv)) - 0.5 * np.mean(np.abs(x[:, None] - x[None, :]))) / r["R"])
    return float(np.mean(out))


def mae(rows):
    return float(np.mean([np.mean(np.abs(np.asarray(r["y"]) - np.median(r["D"], axis=0))) for r in rows]))


def kstar(rows, a=0.9):
    cov = np.array([picp(rows, a, k) for k in P.KAPPA_GRID])
    return float(P.KAPPA_GRID[int(np.argmin(np.abs(cov - a)))])


def loco(short):
    ins = []
    det = []
    for c in P.SHORT_CUTOFFS:
        fit = [r for r in short if r["cutoff"] != c]
        te = [r for r in short if r["cutoff"] == c]
        k = kstar(fit)
        y = stack(te, "y")
        L, U, _ = cal_interval(te, 0.9, k)
        ins.append((y >= L) & (y <= U))
        det.append({"held_out": c, "kappa": k, "picp": float(np.mean(ins[-1]))})
    return float(np.mean(np.concatenate(ins))), det


def by_horizon(rows, k, bands=((1, 5), (6, 10), (11, 16))):
    y = stack(rows, "y")
    h = stack(rows, "h")
    L, U, _ = cal_interval(rows, 0.9, k)
    res = {}
    for a, b in bands:
        m = (h >= a) & (h <= b)
        if m.sum():
            res[f"{a}-{b}"] = {"n": int(m.sum()), "picp": float(np.mean((y[m] >= L[m]) & (y[m] <= U[m])))}
    return res


def kappa_for(rows, mask_fn):
    """kappa that brings 90% coverage to target on a subset of points."""
    y = stack(rows, "y")
    h = stack(rows, "h")
    m = mask_fn(h)
    best, bd = None, 9
    for k in P.KAPPA_GRID:
        L, U, _ = cal_interval(rows, 0.9, k)
        c = np.mean((y[m] >= L[m]) & (y[m] <= U[m]))
        if abs(c - 0.9) < bd:
            best, bd = float(k), abs(c - 0.9)
    return best


def summary_block(short, long_, name):
    k = kstar(short)
    lc, ld = loco(short)
    d = {"kappa": k, "mae_short": mae(short), "picp_raw": picp(short, 0.9, 1.0), "picp_cal": picp(short, 0.9, k),
         "picp_loco": lc, "loco": ld, "pinaw_raw": pinaw(short, 0.9, 1.0), "pinaw_cal": pinaw(short, 0.9, k),
         "is_raw": interval_score(short, 0.9, 1.0), "is_cal": interval_score(short, 0.9, k),
         "crps_cal": crps(short, k),
         "reliability_raw": [picp(short, a, 1.0) for a in NOM], "reliability_cal": [picp(short, a, k) for a in NOM],
         "long": {}}
    for c in (2003, 2007):
        rows = [r for r in long_ if r["cutoff"] == c]
        d["long"][str(c)] = {"mae": mae(rows), "picp_raw": picp(rows, 0.9, 1.0), "picp_cal": picp(rows, 0.9, k),
                             "pinaw_cal": pinaw(rows, 0.9, k), "is_cal": interval_score(rows, 0.9, k),
                             "crps_cal": crps(rows, k)}
    d["horizon"] = by_horizon(short + long_, k)
    print(f"[{name}] k*={k:.2f} PICP raw {d['picp_raw']:.3f} cal {d['picp_cal']:.3f} LOCO {lc:.3f} "
          f"PINAW {d['pinaw_raw']:.2f}->{d['pinaw_cal']:.2f} IS {d['is_cal']:.2f} CRPS {d['crps_cal']:.3f} MAE {d['mae_short']:.2f}")
    return d


# ------------------------------------------------------------------ final forecasts
def final_forecast(t, Y, k_of_h, model="combined", n=4000, tq=None, series=None):
    tq = np.arange(2020, 2041, dtype=float) if tq is None else tq
    FC = {}
    for idx, s in enumerate(P.SERIES):
        y = Y[s] if series is None or s not in series else series[s]
        tt = t if series is None or s not in series else series[s + "_t"]
        D = model_draws(tt, y, tq, n, 5000 + 31 * idx, model)
        med = np.median(D, axis=0)
        h = tq - np.nanmax(tt[~np.isnan(y)])
        kk = np.array([k_of_h(x) for x in h])
        FC[s] = np.maximum(0.0, med + kk * (D - med))
    return tq, FC


def q(x, p):
    return float(np.percentile(x, p))


def propagate(tq, FC, phi=None, corr=None, seed=777, n=4000, eff_scale=None):
    """Emissions and IMP. phi: dict subsector -> array (n, len(tq)) or scalar.
    corr: 7x7 correlation applied by reordering whole trajectories (Iman Conover).
    eff_scale: optional dict series -> (len(tq),) multiplier on demand."""
    FC = {s: v.copy() for s, v in FC.items()}
    if eff_scale:
        for s in FC:
            FC[s] = FC[s] * eff_scale.get(s, 1.0)
    if corr is not None:
        rng = np.random.default_rng(seed + 5)
        Z = rng.multivariate_normal(np.zeros(len(P.SERIES)), corr, size=n)
        for j, s in enumerate(P.SERIES):
            score = FC[s].mean(axis=1)
            order = np.argsort(score)
            ranks = rankdata(Z[:, j], method="ordinal").astype(int) - 1
            FC[s] = FC[s][order][ranks]
    rng = np.random.default_rng(seed)
    fuels = list(P.EF)
    Zef = {(f, p): rng.standard_normal(n) for f in fuels for p in ["CO2", "SO2", "NOx", "CO"]}
    Zg = {f: rng.standard_normal(n) for f in fuels}
    dmd = {f: np.zeros((n, len(tq))) for f in fuels}
    for s in P.SERIES:
        dmd[P.SERIES_FUEL[s]] += FC[s]

    def emis(dm, p, efu=True):
        tot = np.zeros((n, len(tq)))
        for f in fuels:
            base = P.EF[f][p] / (1.0 if p == "CO2" else 1000.0)
            e = base * (1 + P.U_EF[p] * Zef[(f, p)]) if efu else np.full(n, base)
            tot += dm[f] * 1e6 * np.clip(e, 0, None)[:, None] / 1e9
        return tot

    res = {"CO2": emis(dmd, "CO2")}
    for p in ["NOx", "CO", "SO2"]:
        res[p] = emis(dmd, p)
    imp = np.zeros((n, len(tq)))
    imp_sub = {}
    for sub, ss in P.SUBSECTOR_SERIES.items():
        ex = np.zeros((n, len(tq)))
        for s in ss:
            g = P.GAMMA[P.SERIES_FUEL[s]]
            ex += FC[s] * (g * (1 + P.U_GAMMA * Zg[P.SERIES_FUEL[s]]))[:, None]
        ph = P.EXERGY_FLOWS[sub][1] / P.EXERGY_FLOWS[sub][0] if phi is None else phi[sub]
        imp_sub[sub] = (1 - ph) ** 2 * ex
        imp += imp_sub[sub]
    res["IMP"] = imp
    res["demand"] = sum(FC[s] for s in P.SERIES)
    return res, dmd, Zef, imp_sub


def qtrip(x, j=-1):
    return [q(x[:, j], 5), q(x[:, j], 50), q(x[:, j], 95)]


# ------------------------------------------------------------------ main
def point_mae(t, Y):
    """Equal weight MAE (series x origin) for point forecasts."""
    out = {}
    def folds():
        for c in P.SHORT_CUTOFFS:
            yield "short", c, c + P.SHORT_H
        for c, e in P.LONG_SPLITS:
            yield f"long_{c}", c, e
    acc = {}
    for tag, c, e in folds():
        for s in P.SERIES:
            y = Y[s]; tr = t <= c; te = (t > c) & (t <= e) & ~np.isnan(y)
            tt, yt = t[te], y[te]
            for name, pred in (("quadratic", P.quad_point(t[tr], y[tr], tt)), ("drift", P.drift_point(t[tr], y[tr], tt))):
                acc.setdefault(tag, {}).setdefault(name, []).append(np.mean(np.abs(yt - pred)))
    return {tag: {k: float(np.mean(v)) for k, v in d.items()} for tag, d in acc.items()}


def block(sh, lo, name):
    d = summary_block(sh, lo, name)
    return d


def outlook(tq, FC, label):
    res, dmd, Zef, imp_sub = propagate(tq, FC)
    o = {"years": tq.tolist(), "label": label}
    for key in ("demand", "CO2", "NOx", "CO", "SO2", "IMP"):
        o[key] = {str(int(y)): qtrip(res[key], j) for j, y in enumerate(tq)}
        o[key + "_fan"] = {p: np.percentile(res[key], v, axis=0).tolist() for p, v in
                           [("p5", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95)]}
    o["IMP_sub_2040"] = {s: qtrip(v) for s, v in imp_sub.items()}
    o["IMP_share_2040"] = {s: float(np.median(v[:, -1] / res["IMP"][:, -1])) for s, v in imp_sub.items()}
    o["series_2040"] = {s: qtrip(FC[s]) for s in P.SERIES}
    o["series_by_year"] = {s: {str(int(y)): qtrip(FC[s], j) for j, y in enumerate(tq)} for s in P.SERIES}
    return o, res, dmd, Zef


def main():
    t, Y = P.load_data()
    OUT["point_mae"] = point_mae(t, Y)

    # ---- A, B, C: model comparison
    print("backtests ...")
    folds = {}
    for name, model, cov in (("hybrid_wls", "hybrid", "wls"), ("hybrid", "hybrid", "sandwich"),
                             ("drift", "drift", "sandwich"), ("combined", "combined", "sandwich")):
        folds[name] = run_all(t, Y, model, cov)
        OUT[name] = block(*folds[name], name)
    k_c = OUT["combined"]["kappa"]
    sh_c = folds["combined"][0]
    OUT["combined_by_series"] = {}
    for s_ in P.SERIES:
        rr = [r for r in sh_c if r["series"] == s_]
        OUT["combined_by_series"][s_] = {"mae": mae(rr), "picp_raw": picp(rr, 0.9, 1.0), "picp_cal": picp(rr, 0.9, k_c),
                                         "pinaw_cal": pinaw(rr, 0.9, k_c)}

    # ---- D: horizon dependent kappa for the combined model
    rows = folds["combined"][0] + folds["combined"][1]
    OUT["kappa_by_horizon"] = {"1-5": k_c, "6-10": kappa_for(rows, lambda h: (h >= 6) & (h <= 10)),
                               "11-16": kappa_for(rows, lambda h: h >= 11)}
    print("kappa by horizon (combined)", OUT["kappa_by_horizon"])
    kh = OUT["kappa_by_horizon"]
    def k_h(h):
        return kh["1-5"] if h <= 5 else (kh["6-10"] if h <= 10 else kh["11-16"])

    # ---- E: external validation, trained to 2019
    tq = np.arange(2020, 2041, dtype=float)
    _, FC19 = final_forecast(t, Y, lambda h: k_c)
    y19 = Y["Road_PMS"][-1]
    ext = {}
    for j, yr in enumerate(range(2020, 2024)):
        ratio = FC19["Road_PMS"][:, j] / y19
        obs = TO[yr] / TO[2019]
        ext[str(yr)] = {"obs_index": obs, "p5": q(ratio, 5), "p25": q(ratio, 25), "p50": q(ratio, 50),
                        "p75": q(ratio, 75), "p95": q(ratio, 95), "pit": float(np.mean(ratio <= obs)),
                        "in90": bool(q(ratio, 5) <= obs <= q(ratio, 95)), "in50": bool(q(ratio, 25) <= obs <= q(ratio, 75))}
    OUT["external_validation"] = ext
    ext_fan = {p: (np.percentile(FC19["Road_PMS"], v, axis=0) / y19).tolist() for p, v in
               [("p5", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95)]}
    OUT["external_validation_fan"] = {"years": tq.tolist(), **ext_fan}
    print("external validation", {k: (round(v["obs_index"], 3), round(v["p5"], 2), round(v["p50"], 2), round(v["p95"], 2),
                                      v["in90"]) for k, v in ext.items()})
    OUT["overlap_2015_2019"] = {str(y): {"recon_index": float(Y["Road_PMS"][y - 1988] / y19), "nbs_index": TO[y] / TO[2019]}
                                for y in range(2015, 2020)}
    o19, _, _, _ = outlook(tq, FC19, "trained to 2019")
    OUT["outlook_2019"] = {k: o19[k] for k in ("demand", "CO2", "IMP", "series_2040")}

    # ---- F: main outlook, Road PMS trained to 2023 (NBS index), others to 2019
    t_ext = np.arange(1988, 2024, dtype=float)
    obs_pms = np.array([y19 * TO[y] / TO[2019] for y in range(2020, 2024)])
    y_ext = np.concatenate([Y["Road_PMS"], obs_pms])
    tq2 = np.arange(2024, 2041, dtype=float)
    D = model_draws(t_ext, y_ext, tq2, 4000, 5000)
    med = np.median(D, axis=0)
    D = np.maximum(0, med + k_c * (D - med))
    FC = {s: v.copy() for s, v in FC19.items()}
    FC["Road_PMS"] = np.concatenate([np.tile(obs_pms, (4000, 1)), D], axis=1)
    main_o, base, dmd, Zef = outlook(tq, FC, "Road PMS trained to 2023")
    OUT["main"] = main_o
    # share of 2040 IMP variance due to the grade function
    import copy as _c
    U0 = P.U_GAMMA; P.U_GAMMA = 0.0
    r0, _, _, _ = propagate(tq, FC); P.U_GAMMA = U0
    OUT["main"]["gamma_var_share_2040"] = float(max(0.0, 1 - np.var(r0["IMP"][:, -1]) / np.var(base["IMP"][:, -1])))
    OUT["main"]["leverage_PJ_per_pp"] = {sub: float(2 * (1 - P.EXERGY_FLOWS[sub][1] / P.EXERGY_FLOWS[sub][0]) *
                                               np.median(sum(FC[s_][:, -1] * P.GAMMA[P.SERIES_FUEL[s_]] for s_ in ss)) / 100)
                                         for sub, ss in P.SUBSECTOR_SERIES.items()}
    OUT["main"]["series_2019"] = {s_: (None if np.isnan(Y[s_][-1]) else float(Y[s_][-1])) for s_ in P.SERIES}
    OUT["main"]["hist_total"] = np.nansum(np.vstack([Y[s_] for s_ in P.SERIES]), axis=0).tolist()
    OUT["main"]["road_pms_series"] = {"years": t_ext.tolist(), "values": y_ext.tolist()}
    for key in ("demand", "CO2", "IMP"):
        print("MAIN", key, "2040", [round(v, 1) for v in main_o[key]["2040"]], " | trained to 2019",
              [round(v, 1) for v in o19[key]["2040"]])

    # constant versus horizon kappa (sensitivity on main)
    Dh = model_draws(t_ext, y_ext, tq2, 4000, 5000)
    medh = np.median(Dh, axis=0)
    Dh = np.maximum(0, medh + np.array([k_h(x) for x in tq2 - 2023]) * (Dh - medh))
    _, FCh_ = final_forecast(t, Y, k_h)
    FCh_["Road_PMS"] = np.concatenate([np.tile(obs_pms, (4000, 1)), Dh], axis=1)
    rh, _, _, _ = propagate(tq, FCh_)
    OUT["horizon_kappa_effect_2040"] = {"CO2": qtrip(rh["CO2"]), "IMP": qtrip(rh["IMP"]), "demand": qtrip(rh["demand"])}

    # ---- G: structural change scenarios (illustrative)
    yrs = tq
    ramp = np.clip((yrs - 2024) / (2040 - 2024), 0, 1)
    scen = {}
    for name, mult in {"S1": np.where(yrs >= 2024, 0.83, 1.0), "S3": 1 - 0.20 * ramp}.items():
        r, _, _, _ = propagate(yrs, FC, eff_scale={"Road_PMS": mult})
        scen[name] = {"demand_2040": qtrip(r["demand"]), "CO2_2040": qtrip(r["CO2"]), "IMP_2040": qtrip(r["IMP"])}
    FCs = {s: v.copy() for s, v in FC.items()}
    j23 = list(yrs).index(2023)
    b23 = FCs["Road_PMS"][:, j23:j23 + 1]
    FCs["Road_PMS"][:, j23:] = b23 + 0.5 * (FCs["Road_PMS"][:, j23:] - b23)
    r, _, _, _ = propagate(yrs, FCs)
    scen["S2"] = {"demand_2040": qtrip(r["demand"]), "CO2_2040": qtrip(r["CO2"]), "IMP_2040": qtrip(r["IMP"])}
    OUT["scenarios"] = {"definitions": {"S1": "road PMS 17% below the outlook from 2024 (size of the 2022 to 2023 fall)",
                                        "S2": "road PMS growth after 2023 halved",
                                        "S3": "road PMS displaced linearly to 20% by 2040 (fuel switching)"},
                        "results": scen}
    for k in ("S1", "S2", "S3"):
        v = scen[k]
        print(k, "CO2", [round(x, 1) for x in v["CO2_2040"]], "IMP", [round(x) for x in v["IMP_2040"]],
              "demand", [round(x) for x in v["demand_2040"]])

    # ---- H: exergy efficiency sensitivity
    n = 4000
    phi0 = {s: P.EXERGY_FLOWS[s][1] / P.EXERGY_FLOWS[s][0] for s in P.EXERGY_FLOWS}
    rng = np.random.default_rng(31)
    cases = {
        "E0": {s: phi0[s] for s in phi0},
        "E1": {s: np.clip(phi0[s] + 0.02 * rng.standard_normal(n), 0.05, 0.6)[:, None] for s in phi0},
        "E2": {s: (phi0[s] + 0.0025 * (yrs - 2019))[None, :] for s in phi0},
        "E3": {s: (phi0[s] - 0.0010 * (yrs - 2019))[None, :] for s in phi0},
    }
    eff = {}
    for name, ph in cases.items():
        r, _, _, _ = propagate(yrs, FC, phi=ph)
        eff[name] = {"IMP_2040": qtrip(r["IMP"]), "CO2_2040": qtrip(r["CO2"])}
    sc = {s: phi0[P.SUBSECTOR_OF[s]] / (phi0[P.SUBSECTOR_OF[s]] + 0.0025 * (yrs - 2019)) for s in P.SERIES}
    r, _, _, _ = propagate(yrs, FC, phi=cases["E2"], eff_scale=sc)
    eff["E4"] = {"IMP_2040": qtrip(r["IMP"]), "CO2_2040": qtrip(r["CO2"])}
    OUT["efficiency_sensitivity"] = {"definitions": {
        "E0": "constant at 1999 to 2000 values (base)", "E1": "uncertain, normal with 2 pp standard deviation",
        "E2": "improving 0.25 pp per year from 2020", "E3": "deteriorating 0.10 pp per year from 2020",
        "E4": "improving 0.25 pp per year with useful transport service held at the outlook level"}, "results": eff}
    for k, v in eff.items():
        print(k, "IMP", [round(x) for x in v["IMP_2040"]], "CO2", [round(x, 1) for x in v["CO2_2040"]])

    # ---- I: correlation and interaction
    resid = []
    for s in P.SERIES:
        y = Y[s]; m = ~np.isnan(y)
        beta, _, t0 = P.wls_quadratic(t[m], y[m])
        r = np.full(len(t), np.nan); r[m] = y[m] - P.design(t[m], t0) @ beta
        resid.append(r)
    resid = np.array(resid)
    C = np.eye(7)
    for i in range(7):
        for j in range(7):
            if i != j:
                mm = ~np.isnan(resid[i]) & ~np.isnan(resid[j])
                C[i, j] = np.corrcoef(resid[i, mm], resid[j, mm])[0, 1]
    w, Vv = np.linalg.eigh(C)
    C = Vv @ np.diag(np.clip(w, 1e-3, None)) @ Vv.T
    dd = np.sqrt(np.diag(C)); C = C / np.outer(dd, dd)
    off = C[~np.eye(7, dtype=bool)]
    corr = {}
    for name, M in {"C0": None, "C1": C, "C2": 0.5 * np.ones((7, 7)) + 0.5 * np.eye(7),
                    "C3": 0.95 * np.ones((7, 7)) + 0.05 * np.eye(7)}.items():
        r, _, _, _ = propagate(yrs, FC, corr=M)
        corr[name] = {"demand_2040": qtrip(r["demand"]), "CO2_2040": qtrip(r["CO2"]), "IMP_2040": qtrip(r["IMP"])}
    OUT["correlation"] = {"definitions": {"C0": "independent (base)", "C1": "empirical residual correlation",
                                          "C2": "uniform correlation 0.5", "C3": "near comonotonic, correlation 0.95"},
                          "offdiag_mean": float(off.mean()), "offdiag_min": float(off.min()), "offdiag_max": float(off.max()),
                          "results": corr}
    for k, v in corr.items():
        print(k, "CO2", [round(x, 1) for x in v["CO2_2040"]], "IMP", [round(x) for x in v["IMP_2040"]])
    fuels = list(P.EF)
    inter = {}
    for p in ["CO2", "NOx", "CO", "SO2"]:
        yf = base[p][:, -1]; Vt = np.var(yf)
        efb = {f: P.EF[f][p] / (1.0 if p == "CO2" else 1000.0) for f in fuels}
        ed = {f: np.clip(efb[f] * (1 + P.U_EF[p] * Zef[(f, p)]), 0, None) for f in fuels}
        yD = sum(dmd[f][:, -1] * 1e6 * np.mean(ed[f]) / 1e9 for f in fuels)
        yE = sum(np.mean(dmd[f][:, -1]) * 1e6 * ed[f] / 1e9 for f in fuels)
        SD, SE = np.var(yD) / Vt, np.var(yE) / Vt
        inter[p] = {"S_demand": float(SD), "S_factor": float(SE), "interaction": float(max(0.0, 1 - SD - SE)),
                    "cv": float(np.std(yf) / np.mean(yf))}
    OUT["variance_decomposition_2040"] = inter
    print("variance decomposition", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in inter.items()})
    # gamma share of IMP variance
    with open(os.path.join(HERE, "revision_results.json"), "w") as f:
        json.dump(OUT, f, indent=1)
    print("saved revision_results.json")


if __name__ == "__main__":
    P.SUBSECTOR_OF = {s: sub for sub, ss in P.SUBSECTOR_SERIES.items() for s in ss}
    main()

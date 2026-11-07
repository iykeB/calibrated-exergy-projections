"""
simulation.py

Simulation benchmark with known truth. Synthetic panels of 33 countries over 35 years are
generated, and the leave one country out calibration of driver projections is scored
exactly as on the real data (projection_calibration.py functions).

Data generating process, per country g:
  log income per capita   random walk, drift mu_g ~ N(0.02, 0.01), sd 0.02; with probability
                          p_break the drift changes once, at a random year, by N(0, 0.02)
  log population          drift N(0.003, 0.004), sd 0.002
  log exergy per capita   eps_g x log income + AR(1) noise (rho 0.7, sd 0.03), with
                          eps_g ~ N(0.8, s_eps)
  common shock            in scenario S2 every country receives the same shocks:
                          a 6% fall in year 19 (recession) and a 10% fall in year 31
                          with half recovered next year (pandemic)
Scenarios:
  S0  exchangeable countries, no breaks          (method assumptions hold)
  S1  growth breaks in 40% of countries          (structural change, country specific)
  S2  S1 plus common shocks                       (shocks shared by all countries)
  S3  S1 with strongly heterogeneous elasticities (s_eps 0.4 instead of 0.15)
  S4  a block of one third of the countries shares one growth break: contraction in
      years 0 to 9, then faster growth together (a transition and catch up pattern)
Writes ../results/simulation.json.
"""

import json
import os

import numpy as np

import projection_calibration as PC

HERE = os.path.dirname(os.path.abspath(__file__))
OUTF = os.path.join(HERE, "..", "results", "simulation.json")
NC, NY = 33, 35
REPS = 100
SCEN = {"S0": dict(p_break=0.0, common=False, s_eps=0.15),
        "S1": dict(p_break=0.4, common=False, s_eps=0.15),
        "S2": dict(p_break=0.4, common=True, s_eps=0.15),
        "S3": dict(p_break=0.4, common=False, s_eps=0.40),
        "S4": dict(p_break=0.0, common=False, s_eps=0.15, block=True)}


def panel(rng, p_break, common, s_eps, block=False):
    ex, gdp, pop = {}, {}, {}
    for c in range(NC):
        mu = np.full(NY, rng.normal(0.02, 0.01))
        if rng.random() < p_break:
            mu[rng.integers(8, NY - 3):] += rng.normal(0, 0.02)
        if block and c < NC // 3:
            mu[:10] -= 0.03  # a block of countries contracts in years 0 to 9,
            mu[10:] += 0.02  # then grows faster together from year 10 (catch up)
        lgpc = np.cumsum(mu + rng.normal(0, 0.02, NY)) + np.log(2e4)
        lpop = np.cumsum(rng.normal(0.003, 0.004) + rng.normal(0, 0.002, NY)) + np.log(1e7)
        noise = np.zeros(NY)
        for t in range(1, NY):
            noise[t] = 0.7 * noise[t - 1] + rng.normal(0, 0.03)
        eps = rng.normal(0.8, s_eps)
        lex = eps * (lgpc - lgpc[0]) + noise + lpop - lpop[0] + np.log(300.0)
        if common:
            lex[19:] += np.log(0.94)
            lex[31] += np.log(0.90)
            lex[32:] += np.log(0.95)
        ex[c], gdp[c], pop[c] = np.exp(lex), np.exp(lgpc + lpop), np.exp(lpop)
    return ex, gdp, pop


def score(ex, gdp, pop, alpha=0.9):
    cs = list(ex)
    el = {(c, int(o)): PC.elasticity(ex[c], gdp[c], pop[c], o) for c in cs for o in PC.ORIGINS}
    rows = []
    for c in cs:
        for o in PC.ORIGINS:
            eps = float(np.median([el[(k, int(o))] for k in cs if k != c and np.isfinite(el[(k, int(o))])]))
            h = min(PC.HMAX, int(PC.YEARS[-1] - o))
            pr = PC.project(ex[c], gdp[c], pop[c], o, h, eps)["driver"]
            obs = ex[c][int(o - PC.YEARS[0]) + 1:int(o - PC.YEARS[0]) + 1 + h]
            rows += [(c, j + 1, float(np.log(obs[j] / pr[j])), int(o)) for j in range(h)]
    flags = {"all": [], "1-5": [], "6-10": [], "11-24": [], "o2000": [], "o2005": [], "o2010": [], "o2015": []}
    for c in cs:
        E = {}
        for r in rows:
            if r[0] != c:
                E.setdefault(r[1], []).append(r[2])
        Q = PC.smooth_quantiles(E, alpha)
        for r in rows:
            if r[0] == c:
                lo, _, hi = Q(r[1])
                f = lo <= r[2] <= hi
                flags["all"].append(f)
                flags["1-5" if r[1] <= 5 else ("6-10" if r[1] <= 10 else "11-24")].append(f)
                if f"o{r[3]}" in flags:
                    flags[f"o{r[3]}"].append(f)
    return {k: float(np.mean(v)) for k, v in flags.items()}


def main():
    res = {}
    for name, kw in SCEN.items():
        rng = np.random.default_rng(100 + list(SCEN).index(name))
        runs = [score(*panel(rng, **kw)) for _ in range(REPS)]
        res[name] = {k: {"mean": float(np.mean([r[k] for r in runs])),
                         "p05": float(np.percentile([r[k] for r in runs], 5)),
                         "p95": float(np.percentile([r[k] for r in runs], 95))} for k in runs[0]}
        s = res[name]
        print(f"{name}: cover90 all {s['all']['mean']:.3f} (5 to 95% of runs {s['all']['p05']:.3f} to {s['all']['p95']:.3f}) | "
              f"1-5 {s['1-5']['mean']:.3f} 6-10 {s['6-10']['mean']:.3f} 11-24 {s['11-24']['mean']:.3f} | by origin "
              + " ".join(f"{o[1:]} {s[o]['mean']:.2f} [{s[o]['p05']:.2f},{s[o]['p95']:.2f}]" for o in ("o2000", "o2005", "o2010", "o2015")))
    json.dump(res, open(OUTF, "w"), indent=1)


if __name__ == "__main__":
    main()

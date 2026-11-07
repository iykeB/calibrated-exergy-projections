"""
realtime_check.py

Real time calibration check. In the main validation, a country is calibrated with the errors
of the other countries from all origins, including errors whose outcomes occur after the
forecast origin. Here a forecast made at origin O uses only errors already observable at O,
that is errors with o + h <= O, from the other countries. Origins O run from 2008 to 2019,
and horizons up to the largest horizon observable at O (O - 2000). Coverage is compared with
the main calibration on exactly the same forecasts. Widening factors applied to the real
time bounds about their median show how much widening restores nominal coverage.
Writes ../results/realtime_check.json.
"""

import json
import os

import numpy as np

import framework as F
import projection_calibration as PC

HERE = os.path.dirname(os.path.abspath(__file__))
OUTF = os.path.join(HERE, "..", "results", "realtime_check.json")


def main():
    A = F.accounts()
    _, gdp, pop = PC.load()
    rec = [r for r in F.collect(A, gdp, pop) if "ex_driver" in r]
    err = lambda r: np.log(r["ex"] / r["ex_driver"])
    rt, full, by_h = [], [], {}
    for g in F.ISO3:
        other = [r for r in rec if r["g"] != g]
        Qfull = PC.smooth_quantiles({h: [err(r) for r in other if r["h"] == h] for h in range(1, F.HMAX + 1)}, 0.9)
        for O in range(2008, 2020):
            avail = [r for r in other if r["o"] + r["h"] <= O]
            hmax = O - 2000
            E = {}
            for r in avail:
                E.setdefault(r["h"], []).append(err(r))
            Qrt = PC.smooth_quantiles(E, 0.9)
            for r in rec:
                if r["g"] == g and r["o"] == O and r["h"] <= hmax:
                    e = err(r)
                    lo, _, hi = Qrt(r["h"])
                    f_rt = lo <= e <= hi
                    lo2, _, hi2 = Qfull(r["h"])
                    f_full = lo2 <= e <= hi2
                    rt.append(f_rt)
                    full.append(f_full)
                    b = "1-5" if r["h"] <= 5 else ("6-10" if r["h"] <= 10 else "11-19")
                    by_h.setdefault(b, {"rt": [], "full": []})
                    by_h[b]["rt"].append(f_rt)
                    by_h[b]["full"].append(f_full)
    res = {"n": len(rt), "realtime": float(np.mean(rt)), "full": float(np.mean(full)),
           "by_horizon": {b: {"realtime": float(np.mean(v["rt"])), "full": float(np.mean(v["full"])), "n": len(v["rt"])}
                          for b, v in by_h.items()}}
    wid = {}
    for k in (1.0, 1.1, 1.15, 1.2, 1.3):
        f = []
        for g in F.ISO3:
            other = [r for r in rec if r["g"] != g]
            for O in range(2008, 2020):
                E = {}
                for r in other:
                    if r["o"] + r["h"] <= O:
                        E.setdefault(r["h"], []).append(err(r))
                Q = PC.smooth_quantiles(E, 0.9)
                for r in rec:
                    if r["g"] == g and r["o"] == O and r["h"] <= O - 2000:
                        lo, m, hi = Q(r["h"])
                        f.append(m + k * (lo - m) <= err(r) <= m + k * (hi - m))
        wid[str(k)] = float(np.mean(f))
    res["widening"] = wid
    print("widening:", {k: round(100 * v, 1) for k, v in wid.items()})
    print(f"n={res['n']} real time coverage {100*res['realtime']:.1f}% vs main calibration {100*res['full']:.1f}%")
    for b, v in res["by_horizon"].items():
        print(f"  {b}: real time {100*v['realtime']:.1f}% main {100*v['full']:.1f}% (n={v['n']})")
    json.dump(res, open(OUTF, "w"), indent=1)


if __name__ == "__main__":
    main()

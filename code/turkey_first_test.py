"""
turkey_first_test.py

First test of the calibrated framework on Turkiye transport fuel use
(Eurostat nrg_bal_c, 1990 to 2024). Same fold design as the Nigeria paper,
trained and calibrated on 1990 to 2019 only. The years 2020 to 2024 are held
out completely and used as a true out of sample test.
Writes ../results/turkey_first_test.json.
"""

import json
import os

import numpy as np

import pipeline as P
import revision as R

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "data", "eurostat_nrg_bal_c_TR_transport_1990_2024.json")
OUTF = os.path.join(HERE, "..", "results", "turkey_first_test.json")

# (series name, Eurostat balance code, Eurostat product code)
SERIES_DEF = [
    ("Road_diesel", "FC_TRA_ROAD_E", "O4671XR5220B"),
    ("Road_gasoline", "FC_TRA_ROAD_E", "O4652XR5210B"),
    ("Road_LPG", "FC_TRA_ROAD_E", "O4630"),
    ("Rail_diesel", "FC_TRA_RAIL_E", "O4671XR5220B"),
    ("Rail_electricity", "FC_TRA_RAIL_E", "E7000"),
    ("Air_jet", "FC_TRA_DAVI_E", "O4661XR5230B"),
    ("Sea_diesel", "FC_TRA_DNAVI_E", "O4671XR5220B"),
    ("Sea_fueloil", "FC_TRA_DNAVI_E", "O4680"),
]
HOLDOUT = (2020, 2024)


def load():
    d = json.load(open(SRC))
    ids, size = d["id"], d["size"]
    idx = {k: d["dimension"][k]["category"]["index"] for k in ids}

    def val(**kw):
        p = 0
        for k, s in zip(ids, size):
            p = p * s + (idx[k][kw[k]] if k in kw else 0)
        return d["value"].get(str(p))

    years = np.arange(1990, 2025, dtype=float)
    Y = {}
    for name, bal, siec in SERIES_DEF:
        v = np.array([val(nrg_bal=bal, siec=siec, time=str(int(y))) for y in years], dtype=float) / 1000.0  # PJ
        v[(v == 0) | np.isnan(v)] = np.nan  # fuel not yet in use, or not reported
        Y[name] = v
    return years, Y


def holdout_scores(t, Y, k, model):
    """Train to 2019, forecast 2020 to 2024, score against the held out data."""
    tr = t <= 2019
    tq = np.arange(HOLDOUT[0], HOLDOUT[1] + 1, dtype=float)
    rows = []
    for j, s in enumerate(P.SERIES):
        y = Y[s]
        te = (t >= HOLDOUT[0]) & ~np.isnan(y)
        D = R.model_draws(t[tr], y[tr], t[te], 1000, 9000 + 31 * j, model)
        ytr = y[tr][~np.isnan(y[tr])]
        rows.append({"series": s, "cutoff": 2019, "h": (t[te] - 2019).tolist(), "y": y[te].tolist(),
                     "D": D, "R": float(ytr.max() - ytr.min())})
    per = {}
    for r in rows:
        L, U, M = R.cal_interval([r], 0.9, k)
        L5, U5, _ = R.cal_interval([r], 0.5, k)
        y = np.asarray(r["y"])
        per[r["series"]] = {"in90": int(np.sum((y >= L) & (y <= U))), "in50": int(np.sum((y >= L5) & (y <= U5))),
                            "n": int(len(y)), "mape_median": float(np.mean(np.abs(M - y) / y))}
    return {"picp90": R.picp(rows, 0.9, k), "picp50": R.picp(rows, 0.5, k),
            "picp90_raw": R.picp(rows, 0.9, 1.0), "pinaw": R.pinaw(rows, 0.9, k),
            "is": R.interval_score(rows, 0.9, k), "crps": R.crps(rows, k), "mae": R.mae(rows),
            "per_series": per}


def main():
    t, Y = load()
    P.SERIES = [s for s, _, _ in SERIES_DEF]
    t_tr = t[t <= 2019]
    Ytr = {s: Y[s][t <= 2019] for s in P.SERIES}
    out = {"data": {s: dict(zip(map(int, t), [None if np.isnan(v) else round(float(v), 3) for v in Y[s]]))
                    for s in P.SERIES}}
    for model in ("hybrid", "drift", "combined"):
        short, long_ = R.run_all(t_tr, Ytr, model)
        blk = R.summary_block(short, long_, model)
        blk["holdout_2020_2024"] = holdout_scores(t, Y, blk["kappa"], model)
        h = blk["holdout_2020_2024"]
        print(f"   holdout 2020-24: PICP90 {h['picp90']:.3f} (raw {h['picp90_raw']:.3f}) PICP50 {h['picp50']:.3f} "
              f"IS {h['is']:.2f} CRPS {h['crps']:.3f}")
        out[model] = blk
    json.dump(out, open(OUTF, "w"), indent=1, default=float)
    print("written", OUTF)


if __name__ == "__main__":
    main()

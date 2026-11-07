"""
sensitivity.py

Robustness of the validation to the fixed account parameters.
  elec75, elec95   electric drive efficiency 75% and 95% instead of 85%
  ng25             natural gas vehicles at 25% instead of 20%
  bio_lo, bio_hi   blended biofuel grade factors 0.05 below and above their fossil equivalents
  tech025          technical efficiency gain of 0.25 points per year in every mode after 1990,
                   which adds a technical trend to phi on top of the mix effect
  ncv              exergy to energy ratios on a net calorific basis: 1.06 for oil products and
                   1.04 for natural gas (Brockway et al. 2024, Table 1, after Szargut et al. 1988);
                   the base set follows Utlu and Hepbasli (2006)
For each case the driver | persist validation of framework.py is rerun, together with the
Turkiye 2020 accounts. Writes ../results/sensitivity.json.
"""

import copy
import json
import os

import numpy as np

import framework as F
import projection_calibration as PC

HERE = os.path.dirname(os.path.abspath(__file__))
OUTF = os.path.join(HERE, "..", "results", "sensitivity.json")
G0, E0 = copy.deepcopy(F.GAMMA), copy.deepcopy(F.ETA)
LIQUID = ("O4652XR5210B", "O4671XR5220B", "O4661XR5230B", "O4680", "O4630", "R5210B", "R5220B")


def set_case(name):
    F.TECH = 0.0
    F.GAMMA.clear()
    F.GAMMA.update(G0)
    for b in F.ETA:
        F.ETA[b].clear()
        F.ETA[b].update(E0[b])
    if name in ("elec75", "elec95"):
        for b in ("FC_TRA_ROAD_E", "FC_TRA_RAIL_E"):
            F.ETA[b]["E7000"] = int(name[-2:])
    elif name == "ng25":
        F.ETA["FC_TRA_ROAD_E"]["G3000"] = 25
    elif name in ("bio_lo", "bio_hi"):
        d = -0.05 if name == "bio_lo" else 0.05
        F.GAMMA["R5210B"] = G0["R5210B"] + d
        F.GAMMA["R5220B"] = G0["R5220B"] + d
    elif name == "tech025":
        F.TECH = 0.25
    elif name == "ncv":
        for s in LIQUID:
            F.GAMMA[s] = 1.06
        F.GAMMA["G3000"] = 1.04


def main():
    _, gdp, pop = PC.load()
    res = {}
    for name in ("base", "elec75", "elec95", "ng25", "bio_lo", "bio_hi", "ncv", "tech025"):
        set_case(name)
        A = F.accounts()
        rec = F.collect(A, gdp, pop)
        ev = F.evaluate(rec, "driver", "persist")
        evd = F.evaluate(rec, "driver", "drift")
        i = int(np.where(F.YEARS == 2020)[0][0])
        tr = {k: float(A["TR"][k][i]) for k in ("ex", "phi", "imp")}
        res[name] = {v: {"cover": ev[v]["all"]["cover"], "width": ev[v]["mean_relative_width"]} for v in ev}
        res[name]["phi_trend_projector"] = {"cover": evd["phi"]["all"]["cover"], "interval_score": evd["phi"]["interval_score"]}
        res[name]["phi_persist_interval_score"] = ev["phi"]["interval_score"]
        res[name]["turkiye_2020"] = tr
        res[name]["phi_2024_mean"] = float(np.mean([A[g]["phi"][-1] for g in F.ISO3]))
        print(f"{name:7s} cover90 Ex {ev['ex']['all']['cover']:.3f} phi {ev['phi']['all']['cover']:.3f} "
              f"IMP {ev['imp']['all']['cover']:.3f} | TR 2020 Ex {tr['ex']:.0f} PJ phi {100*tr['phi']:.2f}% "
              f"IMP {tr['imp']:.0f} PJ | mean phi 2024 {100*res[name]['phi_2024_mean']:.2f}% | phi IS persist "
              f"{ev['phi']['interval_score']:.2f} trend {evd['phi']['interval_score']:.2f} (trend cover {evd['phi']['all']['cover']:.3f})")
    set_case("base")
    json.dump(res, open(OUTF, "w"), indent=1)


if __name__ == "__main__":
    main()

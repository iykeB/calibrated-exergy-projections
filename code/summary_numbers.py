"""
summary_numbers.py

Two summaries quoted in the manuscript:
  mae_by_horizon.json        mean absolute log error of exergy input by horizon and projector
  uh_band_by_projector.json  90% interval around the Utlu and Hepbasli (2006) 2020 exergy input
                             (1385.73 PJ, horizon 20 from data to 2000), calibrated with the
                             drift, driver and oracle errors of the 32 countries other than Türkiye
"""

import json
import os

import numpy as np

import framework as F
import projection_calibration as PC

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
UH_EX_2020 = 1385.73


def main():
    A = F.accounts()
    _, gdp, pop = PC.load()
    rec = [r for r in F.collect(A, gdp, pop) if "ex_driver" in r]  # same forecasts for every projector
    mae = {k: {h: float(np.mean([abs(np.log(r["ex"] / r["ex_" + k])) for r in rec if r["h"] == h and "ex_" + k in r]))
               for h in range(1, F.HMAX + 1)} for k in ("drift", "driver", "oracle")}
    json.dump(mae, open(os.path.join(RES, "mae_by_horizon.json"), "w"), indent=1)
    band = {}
    for k in ("drift", "driver", "oracle"):
        E = {}
        for r in rec:
            if r["g"] != "TR" and "ex_" + k in r:
                E.setdefault(r["h"], []).append(np.log(r["ex"] / r["ex_" + k]))
        lo, _, hi = PC.smooth_quantiles(E, 0.9)(20)
        band[k] = [UH_EX_2020 * float(np.exp(lo)), UH_EX_2020 * float(np.exp(hi))]
    json.dump(band, open(os.path.join(RES, "uh_band_by_projector.json"), "w"), indent=1)
    print({k: round(v[20], 3) for k, v in mae.items()}, {k: [round(x) for x in v] for k, v in band.items()})


if __name__ == "__main__":
    main()

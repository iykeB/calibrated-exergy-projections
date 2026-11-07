"""
build_unsd.py

Converts the UN Energy Statistics Database downloads (data/unsd/unsd_*.csv, SDMX CSV) into
energy use in TJ by year, transaction (transport mode) and commodity (fuel).
Fuels in metric tons use the conversion factor reported with each value (TJ per kt);
electricity in GWh uses 3.6 TJ per GWh. Writes data/unsd/unsd_transport_TJ.json.
"""

import collections
import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
UNSD = os.path.join(HERE, "..", "data", "unsd")
NAMES = {"840": "USA", "124": "Canada", "392": "Japan", "036": "Australia", "410": "Korea", "554": "New Zealand",
         "484": "Mexico", "152": "Chile", "076": "Brazil", "710": "South Africa", "032": "Argentina", "458": "Malaysia",
         "764": "Thailand", "356": "India", "566": "Nigeria", "818": "Egypt", "156": "China", "360": "Indonesia",
         "792": "Türkiye", "276": "Germany"}


def to_tj(r):
    v = float(r["OBS_VALUE"]) * 10 ** int(r["UNIT_MULT"] or 0)
    unit = r["UNIT_MEASURE"]
    if unit == "GWHR":
        return v * 3.6 / 1000.0
    if unit == "TN":
        return v / 1000.0 * float(r["CONVERSION_FACTOR"])
    if unit == "TJ":
        return v
    return float("nan")


def main():
    out = {}
    for f in sorted(glob.glob(os.path.join(UNSD, "unsd_*.csv"))):
        area = os.path.basename(f)[5:8]
        d = collections.defaultdict(float)
        for r in csv.DictReader(open(f, encoding="utf-8")):
            d[f"{r['TIME_PERIOD']}|{r['TRANSACTION']}|{r['COMMODITY']}"] += to_tj(r)
        out[NAMES[area]] = dict(d)
    json.dump(out, open(os.path.join(UNSD, "unsd_transport_TJ.json"), "w"))
    print("countries:", len(out))


if __name__ == "__main__":
    main()

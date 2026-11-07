# Calibrated uncertainty for point projections of national energy and exergy accounts

Code, data and results for the manuscript

> B.I. Simeon, S.D. Ahmadu, N. Aghilizadeh, *From Point Projections to Calibrated Uncertainty:
> A Transferable Probabilistic Framework for Energy and Exergy Forecasting of National
> Transport Systems* (manuscript in preparation for *Energy*).

The framework learns how far a kind of point projection misses, by forecast horizon, from
comparable countries, and turns any point projection of exergy input, exergy efficiency and
improvement potential into a calibrated interval. It is validated on the transport sectors of
33 European countries (Eurostat, 1990 to 2024), eight countries outside Europe (UN Energy
Statistics Database), a real time check and a simulation with known truth.

## Setup

```
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Tested with Python 3.12 and the versions in `requirements.txt`.

## Reproducing the results

Run from the `code` folder. Each script writes a JSON file to `results/`.

| Script | Purpose | Output |
|---|---|---|
| `projection_calibration.py` | point projectors (drift, driver, oracle) and leave one country out calibration of exergy input | `projection_calibration.json` |
| `framework.py` | exergy accounts (Ex, efficiency, improvement potential), calibration, paired errors, interval score, country bootstrap | `framework.json` |
| `simulation.py` | simulation benchmark, scenarios S0 to S4 | `simulation.json` |
| `sensitivity.py` | sensitivity to account parameters | `sensitivity.json` |
| `external_check.py` | United Kingdom and Ukraine; raw versus smoothed quantiles | `external_check.json` |
| `build_unsd.py`, `noneu_check.py` | eight countries outside Europe, all panel errors and 20 income peers | `noneu_check.json` |
| `realtime_check.py` | calibration with only the errors observable at each origin | `realtime_check.json` |
| `summary_numbers.py` | error by horizon; calibrated bands for the published 2020 projection | `mae_by_horizon.json`, `uh_band_by_projector.json` |
| `make_figures.py` | figures of the manuscript | `figures/` |
| `turkey_first_test.py`, `turkey_projection_2020.py` | exploratory Türkiye runs (trend only forecast from 2000) | JSON in `results/` |

`pipeline.py` and `revision.py` come from the authors' earlier study (Thermo, 2026) and supply
the hybrid and drift models used by the exploratory Türkiye scripts.

To download the data again from the original sources, run `data/fetch_data.sh` and then
`code/build_unsd.py`.

## Licence

Code: MIT licence (see `LICENSE`). Data: see `data/README.md` for the terms of each source.

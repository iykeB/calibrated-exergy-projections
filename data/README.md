# Data

All data are public. Retrieved on 6 November 2026 with `fetch_data.sh`.

| File | Source | Licence / terms |
|---|---|---|
| `eurostat_nrg_bal_c_*.json` | Eurostat, Complete energy balances (nrg_bal_c), https://doi.org/10.2908/NRG_BAL_C. Final energy use in TJ by transport mode (road, rail, domestic aviation, domestic navigation) and fuel, 1990 to 2024 | Reuse permitted with acknowledgement of the source (Commission Decision 2011/833/EU) |
| `wb_*.json` | World Bank, World Development Indicators: NY.GDP.MKTP.KD (GDP, constant 2015 US$) and SP.POP.TOTL (population), 1985 to 2025 | CC BY 4.0 |
| `unsd/unsd_*.csv` | United Nations Statistics Division, Energy Statistics Database, SDMX dataflow DF_UNDATA_ENERGY (last updated 15 December 2025), transport by mode and fuel | Free to copy and redistribute provided UNdata is cited as the reference |
| `unsd/unsd_transport_TJ.json` | Built from the CSV files by `code/build_unsd.py` | as above |
| `../code/nbs_series.json` | National Bureau of Statistics of Nigeria, PMS truck out 2015 to 2023 (used by the earlier Thermo study code) | public statistics |

#!/bin/sh
# Downloads all input data again from the original public sources (run from this folder).
set -e
EU="https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/nrg_bal_c"
curl -s -o eurostat_nrg_bal_c_all_transport_1990_2024.json "$EU?unit=TJ&nrg_bal=FC_TRA_ROAD_E&nrg_bal=FC_TRA_RAIL_E&nrg_bal=FC_TRA_DNAVI_E&nrg_bal=FC_TRA_DAVI_E&siec=O4652XR5210B&siec=O4671XR5220B&siec=O4661XR5230B&siec=O4680&siec=O4630&siec=E7000&siec=G3000&siec=R5210B&siec=R5220B"
curl -s -o eurostat_nrg_bal_c_TR_transport_1990_2024.json "$EU?geo=TR&unit=TJ&nrg_bal=FC_TRA_ROAD_E&nrg_bal=FC_TRA_RAIL_E&nrg_bal=FC_TRA_DNAVI_E&nrg_bal=FC_TRA_DAVI_E&siec=O4652XR5210B&siec=O4671XR5220B&siec=O4661XR5230B&siec=O4680&siec=O4630&siec=E7000&siec=N900H"
WB="https://api.worldbank.org/v2/country"
EUC="BEL;BGR;CZE;DNK;DEU;EST;IRL;GRC;ESP;FRA;HRV;ITA;CYP;LVA;LTU;LUX;HUN;MLT;NLD;AUT;POL;PRT;ROU;SVN;SVK;FIN;SWE;ISL;NOR;GBR;BIH;MNE;MDA;MKD;GEO;ALB;SRB;TUR;UKR;XKX"
NEU="USA;CAN;JPN;AUS;KOR;NZL;MEX;CHL;BRA;ZAF;ARG;MYS;THA;IND;NGA;EGY;CHN;IDN"
for ind in NY.GDP.MKTP.KD SP.POP.TOTL; do
  curl -s -o "wb_$ind.json" "$WB/$EUC/indicator/$ind?format=json&date=1985:2025&per_page=5000"
  curl -s -o "wb_nonEU_$ind.json" "$WB/$NEU/indicator/$ind?format=json&date=1985:2025&per_page=5000"
done
mkdir -p unsd
for a in 840 124 392 036 410 554 484 152 076 710 032 458 764 356 566 818 156 360 792 276; do
  curl -s -L -H "Accept: application/vnd.sdmx.data+csv" -o "unsd/unsd_$a.csv" \
    "https://data.un.org/ws/rest/data/UNSD,DF_UNDATA_ENERGY,1.2/A.$a.4652+4661+4670+4680+4630+3000+7000+5210+5220.1221+1222+1223+1224?startPeriod=1990"
done
echo "done; now run: python ../code/build_unsd.py"

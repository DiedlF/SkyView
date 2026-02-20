# DWD ICON reference bundle (downloaded)

Downloaded for project context on **2026-02-18 (UTC)**.

## Files

- `icon_database_main.pdf`
  - Source: https://www.dwd.de/DWD/forschung/nwv/fepub/icon_database_main.pdf
  - Purpose: authoritative ICON database reference (model variables, metadata, diagnostics, semantics).

- `icon_description.html`
  - Source: https://www.dwd.de/EN/research/weatherforecasting/num_modelling/01_num_weather_prediction_modells/icon_description.html
  - Purpose: high-level ICON model description from DWD.

- `opendata.html`
  - Source: https://www.dwd.de/EN/ourservices/opendata/opendata.html
  - Purpose: DWD open data service context and access/legal framing.

- `icon-d2-grib-00-index.html`
  - Source: https://opendata.dwd.de/weather/nwp/icon-d2/grib/00/
  - Purpose: concrete open data index showing ICON-D2 variable/parameter directory naming for run `00`.

- `icon-eu-grib-00-index.html`
  - Source: https://opendata.dwd.de/weather/nwp/icon-eu/grib/00/
  - Purpose: concrete open data index showing ICON-EU variable/parameter directory naming for run `00`.

## Notes

- The PDF is the key source for variable semantics.
- The two opendata index files are useful for validating which shortNames/parameters are actually published for D2/EU.
- If needed, add a follow-up extraction step (PDF -> markdown snippets) focused on Skyview-used variables (`tot_prec`, `rain_gsp`, `rain_con`, `snow_gsp`, `snow_con`, `grau_gsp`, `mh`, `hbas_sc`, `htop_sc`, `lpi`, etc.).

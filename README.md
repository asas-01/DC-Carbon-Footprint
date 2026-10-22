# Data Center Carbon Footprint: NREL ESIF Analysis

Code and data for *Carbon Footprint of Cloud Computing: The Hidden Environmental Cost of Digital Infrastructure* (asas).

The analysis uses the NREL Energy Systems Integration Facility (ESIF) high-performance computing data center dataset (Clark & Strelka, OEDI submission 8616, CC BY 4.0).

## Reproducing the results

```bash
pip install pandas pyarrow numpy scipy scikit-learn matplotlib
python run_all.py
```

### Raw data

Download the two Parquet files from https://data.openei.org/submissions/8616 and save them as:

| File on OEDI | Save as |
|---|---|
| `esif.influx.buildingData.PUE.combined.parquet` | `data/esif_pue_raw.parquet` |
| `esif.influx.buildingData.outside.combined_2.parquet` | `data/esif_weather_raw.parquet` |

The raw files (~170 MB) are not committed. `data/world_grid_carbon_intensity_ember.csv` is the world lifecycle carbon intensity of electricity (Ember, via Our World in Data, https://ourworldindata.org/grapher/carbon-intensity-electricity), used only as a historical comparison (2015–2025 decline of ~1.5%/yr); it does not set any scenario parameter. The cleaned hourly file `esif_hourly_merged.csv` is committed, so `02`–`04` can run without them.

## Pipeline

| Script | What it does |
|---|---|
| `01_prepare_data.py` | Filters invalid one-minute readings, aggregates **all** readings to hourly means, and keeps hours with at least 50% of their readings. Then it joins power and weather data and recomputes hourly PUE. It also rebuilds the original every-10th-row version for a sensitivity check. |
| `02_analysis.py` | Descriptive statistics and Pearson correlations with Fisher and moving-block bootstrap 95% CIs. Fits Models A/B/C with Linear Regression, Random Forest and Gradient Boosting under three validation schemes, and checks feature-importance stability, sampling sensitivity and the April 2024 subsystem change. |
| `03_projections.py` | 2024–2030 scenarios built from one baseline (415 TWh, 0.445 kg CO₂/kWh), with a year-by-year emission-factor table and a sensitivity table. |
| `04_generate_figures.py` | All figures. The ESIF figures are drawn from saved results so they match the reported numbers. |

Outputs: `output/results.json`, `output/tables/*.csv`, `output/figures/*.png`.

## Changes from the earlier version

| Topic | Earlier version | Current version |
|---|---|---|
| 2024 baseline | Two electricity baselines (415 and 460 TWh); emission factor of 0.49 kg/kWh in the code | One baseline: 415 TWh × 0.445 kg/kWh ≈ 185 Mt CO₂ |
| Emission-factor path | Fixed year-by-year values with no stated source | `EF(t) = 0.445 × (1 − d)^(t−2024)` with *d* = 3.6%/yr (Business as Usual, IEA Electricity 2025 forecast), 1%/yr (Accelerated AI, assumed) and 11%/yr (Green Transition, 2020–2030 rate in the IEA Net Zero by 2050 pathway, Table A.5, applied from 2024). Year-by-year values are in `projections_by_year.csv`; assumptions are in `scenario_assumptions.csv`. |
| Grid-intensity reduction | Reported without a reference year | Calculated against 2024 (`ef_2030_reduction_vs_2024_pct`) |
| Model validation | Random 80/20 split only | Random split (for comparison), chronological 80/20 split and rolling-origin validation by year (2019–2025) |
| Feature importance | Impurity importance from one split | Impurity and permutation importance, mean ± SD across rolling folds and 10 repeated random splits |
| Algorithms | Gradient Boosting not run for Model A | Linear Regression, Random Forest and Gradient Boosting for Models A, B and C |
| Correlation uncertainty | p-values only | Fisher and one-week moving-block bootstrap CIs, r² and approximate effective sample size |
| Sampling | Every 10th raw reading kept before hourly averaging | All readings aggregated; the every-10th method is kept only as a sensitivity check |
| PUE filter | 0.95–1.50 | 1.00–1.50, applied to one-minute readings before aggregation |
| Overhead definition | Cooling + HVAC + pumps | Plug and light load also included, matching how NREL computes PUE |
| Figure 3 (embodied carbon) | Single grid intensity | Adds an embodied-share vs grid-intensity panel with break-even intensities; server power assumptions are in `config.py` |
| Figure 5 (PUE trend) | PUE only | Adds overhead power by subsystem, showing the April 2024 step in HVAC power (`regime_change` in `results.json`) |
| Figures 7, 8, 10 | Axis titles said *energy* | Titles say *power* (kW); Figure 10 shows random and chronological test sets |
| Figure 2 | Single values | Ranges from published estimates (0.3–2.9 Wh per LLM query; ~2.9 Wh per image) |

The raw timestamps are UTC. Month and hour of day are converted to Golden, CO local time (`America/Denver`).

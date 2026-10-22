"""
Author: asas
Description: Data Center Carbon Footprint Analysis

config.py - Shared configuration for the carbon footprint analysis.
All scripts import from here so paths, constants and assumptions live in one place.
"""

import os
import matplotlib

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output")
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "figures")
TABLE_DIR = os.path.join(OUTPUT_ROOT, "tables")
RESULTS_FILE = os.path.join(OUTPUT_ROOT, "results.json")

# Raw NREL ESIF files (OEDI submission 8616). See README.md for download links.
POWER_RAW = os.path.join(DATA_DIR, "esif_pue_raw.parquet")
WEATHER_RAW = os.path.join(DATA_DIR, "esif_weather_raw.parquet")

# Cleaned hourly outputs
MERGED_CSV = os.path.join(PROJECT_ROOT, "esif_hourly_merged.csv")
# Same pipeline, but keeping only every 10th raw row (the original method);
# used only for the sampling sensitivity check.
MERGED_SUBSAMPLED_CSV = os.path.join(DATA_DIR, "esif_hourly_merged_every10th.csv")

# ── Processing constants ─────────────────────────────────────────────────
SUBSAMPLE_STEP = 10       # only used for the sensitivity comparison
PUE_MIN = 1.00            # PUE < 1 is physically impossible
PUE_MAX = 1.50
TEMP_MIN_F = -50
TEMP_MAX_F = 130
HUMIDITY_MIN = 0
HUMIDITY_MAX = 100
# An hour is kept only if at least this share of its expected readings survived cleaning
MIN_HOURLY_COVERAGE = 0.5
LOCAL_TZ = "America/Denver"   # raw timestamps are UTC; Golden, CO local time

POWER_COMPONENTS = ["it_power_kw", "cooling_kw", "hvac_kw", "pump_kw", "plug_and_light_kw"]
OVERHEAD_COMPONENTS = ["cooling_kw", "hvac_kw", "pump_kw", "plug_and_light_kw"]

# ── Modeling constants ───────────────────────────────────────────────────
TEST_SIZE = 0.20
RANDOM_STATE = 42
N_REPEATED_SPLITS = 10              # repeated random splits for importance stability
ROLLING_TEST_YEARS = list(range(2019, 2026))   # rolling-origin (expanding window) folds
PERMUTATION_REPEATS = 5
BOOTSTRAP_REPS = 1000
BOOTSTRAP_BLOCK_HOURS = 168         # one-week moving blocks for autocorrelated data

RF_PARAMS = {"n_estimators": 200, "max_depth": 15, "random_state": RANDOM_STATE, "n_jobs": -1}
GB_PARAMS = {"n_estimators": 200, "max_depth": 5, "random_state": RANDOM_STATE}

# ── Projection assumptions ───────────────────────────────────────────────
# 2024 global data center electricity use: IEA, Energy and AI (2025).
BASE_YEAR = 2024
END_YEAR = 2030
BASE_TWH = 415
# 2024 global average CO2 intensity of electricity generation (kg CO2/kWh).
# Value used in the manuscript (~445 g/kWh, IEA). 415 TWh x 0.445 = ~185 Mt, which is
# close to the IEA's own estimate of ~180 Mt CO2 for data centers in 2024.
BASE_EMISSION_FACTOR = 0.445
# IEA Energy and AI base case for 2030 (used only as a reference point)
IEA_BASE_CASE_2030_TWH = 945

# Each scenario = annual electricity growth rate + annual decline rate of the grid
# emission factor. EF(t) = EF_2024 * (1 - decline) ** (t - 2024).
# Historical comparison for the decline rate (not used in any scenario): world lifecycle carbon
# intensity of electricity (Ember, via Our World in Data), compound annual change between these years.
HIST_GRID_FILE = os.path.join(DATA_DIR, "world_grid_carbon_intensity_ember.csv")
HIST_START, HIST_END = 2015, 2025
SCENARIOS = {
    "business_as_usual": {
        "label": "Business as Usual",
        "description": "Recent trends continue",
        "growth": 0.12,          # recent historical growth (IEA: ~12%/yr over the last five years)
        "ef_decline": 0.036,     # IEA Electricity 2025 forecast decline of global grid CO2 intensity (3.6%/yr)
    },
    "accelerated_ai": {
        "label": "Accelerated AI",
        "description": "AI drives faster growth; part of new demand met by fossil generation",
        "growth": 0.15,          # IEA base case (~15%/yr, 415 -> ~945 TWh)
        "ef_decline": 0.01,      # slower than the historical decline
    },
    "green_transition": {
        "label": "Green Transition",
        "description": "Efficiency slows growth; rapid renewable expansion",
        "growth": 0.08,          # efficiency gains slow demand growth
        "ef_decline": 0.11,      # IEA Net Zero by 2050 (2021), Table A.5: CO2 intensity of electricity
                                 # generation 0.438 (2020) -> 0.138 (2030), -11%/yr. Applied from the
                                 # 2024 baseline, so EF reaches ~0.22 (not 0.138) by 2030.
    },
}
# Sensitivity ranges for the projection
SENS_BASE_EF = [0.40, 0.445, 0.49]
SENS_GROWTH_DELTA = [-0.02, 0.0, 0.02]

# US average grid emission factor (kg CO2/kWh), used for the facility and Figure 3
US_GRID_EMISSION_FACTOR = 0.348  # EPA eGRID2023 U.S. average (348 g/kWh)

# ── Figure 3 (embodied vs operational carbon) assumptions ────────────────
# Embodied carbon values: Weppe et al. (HotCarbon '25).
# Average power draw values are illustrative assumptions (not measured).
SERVERS = [
    {"name": "2015 Intel server", "embodied_kg": 345, "avg_power_kw": 0.50},
    {"name": "2024 ARM server", "embodied_kg": 1024, "avg_power_kw": 0.60},
]
SERVER_LIFETIME_YEARS = 5
# Reference grid intensities (kg CO2/kWh) marked on Figure 3b
REFERENCE_GRIDS = {
    "Low-carbon grid": 0.03,
    "U.S. average": US_GRID_EMISSION_FACTOR,
    "Global average": BASE_EMISSION_FACTOR,
}

# ── Plot style ───────────────────────────────────────────────────────────
COLORS = {
    "blue":       "#2166AC",
    "red":        "#B2182B",
    "green":      "#1B7837",
    "orange":     "#E08214",
    "gray":       "#666666",
    "light_blue": "#92C5DE",
    "purple":     "#762A83",
    "teal":       "#01665E",
}


def setup_plot_style():
    """Apply consistent matplotlib styling for publication figures."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


def ensure_dirs():
    for d in (DATA_DIR, OUTPUT_DIR, TABLE_DIR):
        os.makedirs(d, exist_ok=True)

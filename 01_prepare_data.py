"""
Author: asas
Description: Data Center Carbon Footprint Analysis

01_prepare_data.py - Clean, aggregate, and merge the NREL ESIF datasets.

Pipeline (applied to every raw one-minute reading; no subsampling):
  1. Drop physically invalid readings (negative power, PUE outside 1.0-1.5,
     temperature outside -50 to 130 F, humidity outside 0-100%).
  2. Aggregate to hourly means (timestamps are UTC in the raw files).
  3. Keep hours where at least MIN_HOURLY_COVERAGE of the expected readings survived.
  4. Inner-join power and weather on the hourly timestamp.
  5. Recompute hourly PUE as total facility power / IT power.

The original every-10th-row method is also run so the two can be compared
(sampling sensitivity check, see 02_analysis.py).
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    POWER_RAW, WEATHER_RAW, MERGED_CSV, MERGED_SUBSAMPLED_CSV, OUTPUT_ROOT,
    SUBSAMPLE_STEP, PUE_MIN, PUE_MAX, TEMP_MIN_F, TEMP_MAX_F, HUMIDITY_MIN, HUMIDITY_MAX,
    MIN_HOURLY_COVERAGE, LOCAL_TZ, POWER_COMPONENTS, OVERHEAD_COMPONENTS, ensure_dirs,
)

CLEANING_SUMMARY = os.path.join(OUTPUT_ROOT, "cleaning_summary.json")


def load_raw():
    for path in (POWER_RAW, WEATHER_RAW):
        if not os.path.exists(path):
            sys.exit(f"Missing raw file: {path}\nSee README.md for download instructions.")
    power = pd.read_parquet(POWER_RAW, columns=["ts"] + POWER_COMPONENTS + ["pue"])
    weather = pd.read_parquet(WEATHER_RAW, columns=["ts", "outdoor_air_temp", "outdoor_air_humidity"])
    power = power.sort_values("ts").reset_index(drop=True)
    weather = weather.sort_values("ts").reset_index(drop=True)
    return power, weather


def clean_minute_level(power, weather, log):
    log["raw_power_rows"] = int(len(power))
    log["raw_weather_rows"] = int(len(weather))

    weather = weather[weather["ts"] >= "2015-01-01"]
    w_ok = (
        weather["outdoor_air_temp"].between(TEMP_MIN_F, TEMP_MAX_F, inclusive="neither")
        & weather["outdoor_air_humidity"].between(HUMIDITY_MIN, HUMIDITY_MAX)
    )
    log["weather_rows_removed"] = int(log["raw_weather_rows"] - w_ok.sum())
    weather = weather[w_ok]

    p_ok = power[POWER_COMPONENTS].notna().all(axis=1) & power["pue"].notna()
    p_ok &= (power[POWER_COMPONENTS] >= 0).all(axis=1) & (power["it_power_kw"] > 0)
    log["power_rows_missing_or_negative"] = int((~p_ok).sum())
    in_range = power["pue"].between(PUE_MIN, PUE_MAX)
    log["power_rows_pue_out_of_range"] = int((p_ok & ~in_range).sum())
    power = power[p_ok & in_range]
    return power, weather


def to_hourly(power, weather, expected_per_hour):
    """Hourly means, keeping only hours with enough surviving readings."""
    min_count = max(1, int(np.ceil(MIN_HOURLY_COVERAGE * expected_per_hour)))

    p = power.set_index("ts")
    p_mean = p[POWER_COMPONENTS].resample("h").mean()
    p_count = p["it_power_kw"].resample("h").count()
    p_mean = p_mean[p_count >= min_count]

    w = weather.set_index("ts")
    w_mean = w[["outdoor_air_temp", "outdoor_air_humidity"]].resample("h").mean()
    w_count = w["outdoor_air_temp"].resample("h").count()
    w_mean = w_mean[w_count >= min_count]

    df = p_mean.join(w_mean, how="inner").dropna()
    df["overhead_kw"] = df[OVERHEAD_COMPONENTS].sum(axis=1)
    df["total_power_kw"] = df["it_power_kw"] + df["overhead_kw"]
    df["pue"] = df["total_power_kw"] / df["it_power_kw"]
    df = df[df["pue"].between(PUE_MIN, PUE_MAX)]

    local = df.index.tz_localize("UTC").tz_convert(LOCAL_TZ)
    df["local_month"] = local.month
    df["local_hour"] = local.hour
    df.index.name = "timestamp_utc"
    return df


def main():
    print("=" * 60)
    print("  STEP 1: Prepare Data (all raw readings, hourly aggregation)")
    print("=" * 60)
    ensure_dirs()
    log = {}

    power, weather = load_raw()
    print(f"Raw power rows:   {len(power):,}")
    print(f"Raw weather rows: {len(weather):,}")

    median_step_s = power["ts"].diff().dt.total_seconds().median()
    expected_per_hour = 3600 / median_step_s
    log["median_reading_interval_s"] = float(median_step_s)

    power_c, weather_c = clean_minute_level(power, weather, log)
    df = to_hourly(power_c, weather_c, expected_per_hour)
    df.to_csv(MERGED_CSV)
    log["hourly_observations"] = int(len(df))
    log["date_range_utc"] = [str(df.index.min()), str(df.index.max())]
    span_hours = (df.index.max() - df.index.min()) / pd.Timedelta(hours=1) + 1
    log["hourly_coverage_of_span"] = round(len(df) / span_hours, 4)
    print(f"\nFull-resolution hourly dataset: {len(df):,} hours "
          f"({df.index.min()} -> {df.index.max()} UTC)")

    # Original method (every 10th raw row, then hourly) for the sensitivity check
    power_s = power.iloc[::SUBSAMPLE_STEP]
    weather_s = weather.iloc[::SUBSAMPLE_STEP]
    power_sc, weather_sc = clean_minute_level(power_s, weather_s, {})
    df_s = to_hourly(power_sc, weather_sc, expected_per_hour / SUBSAMPLE_STEP)
    df_s.to_csv(MERGED_SUBSAMPLED_CSV)
    log["subsampled_hourly_observations"] = int(len(df_s))
    print(f"Every-{SUBSAMPLE_STEP}th-row hourly dataset: {len(df_s):,} hours")

    with open(CLEANING_SUMMARY, "w") as f:
        json.dump(log, f, indent=2)
    print("\nCleaning summary:")
    for k, v in log.items():
        print(f"  {k}: {v}")
    print("\nDescriptive statistics (full dataset):")
    print(df.describe().round(3).T.to_string())


if __name__ == "__main__":
    main()

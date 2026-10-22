"""
Author: asas
Description: Data Center Carbon Footprint Analysis

03_projections.py - Global data center electricity and CO2 projections, 2024-2030.

One consistent baseline is used everywhere:
  2024 electricity  = BASE_TWH (415 TWh, IEA Energy and AI 2025)
  2024 grid factor  = BASE_EMISSION_FACTOR (0.445 kg CO2/kWh)
  2024 emissions    = 415 TWh x 0.445 kg/kWh = ~185 Mt CO2

For each scenario and year t:
  TWh(t) = 415 * (1 + growth) ** (t - 2024)
  EF(t)  = 0.445 * (1 - ef_decline) ** (t - 2024)       [kg CO2/kWh]
  CO2(t) = TWh(t) * EF(t)                              [Mt CO2]
(1 TWh = 1e9 kWh and 1 Mt = 1e9 kg, so TWh x kg/kWh = Mt.)

Outputs a year-by-year table (output/tables/projections_by_year.csv),
a sensitivity table, and facility-level annual energy/CO2.
"""

import itertools
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    MERGED_CSV, RESULTS_FILE, TABLE_DIR, BASE_YEAR, END_YEAR, BASE_TWH, BASE_EMISSION_FACTOR,
    IEA_BASE_CASE_2030_TWH, SCENARIOS, SENS_BASE_EF, SENS_GROWTH_DELTA,
    US_GRID_EMISSION_FACTOR, HIST_GRID_FILE, HIST_START, HIST_END, ensure_dirs,
)


def historical_decline():
    """Compound annual change in world grid carbon intensity between HIST_START and HIST_END."""
    h = pd.read_csv(HIST_GRID_FILE).set_index("year").iloc[:, 0]
    a, b = h.loc[HIST_START], h.loc[HIST_END]
    rate = 1 - (b / a) ** (1 / (HIST_END - HIST_START))
    return {"start_year": HIST_START, "end_year": HIST_END, "start_g_per_kwh": float(a),
            "end_g_per_kwh": float(b), "annual_decline": round(float(rate), 4)}


def project(growth, ef_decline, base_twh=BASE_TWH, base_ef=BASE_EMISSION_FACTOR):
    rows = []
    for t in range(BASE_YEAR, END_YEAR + 1):
        k = t - BASE_YEAR
        twh = base_twh * (1 + growth) ** k
        ef = base_ef * (1 - ef_decline) ** k
        rows.append({"year": t, "twh": twh, "ef_kg_per_kwh": ef, "co2_mt": twh * ef})
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("  STEP 3: Carbon Footprint Projections")
    print("=" * 60)
    ensure_dirs()

    base_co2 = BASE_TWH * BASE_EMISSION_FACTOR
    print(f"2024 baseline: {BASE_TWH} TWh x {BASE_EMISSION_FACTOR} kg/kWh = {base_co2:.1f} Mt CO2")
    iea_cagr = (IEA_BASE_CASE_2030_TWH / BASE_TWH) ** (1 / (END_YEAR - BASE_YEAR)) - 1
    print(f"Growth rate implied by the IEA base case ({IEA_BASE_CASE_2030_TWH} TWh in 2030): {iea_cagr:.1%}/yr")
    hist = historical_decline()
    print(f"World grid carbon intensity {HIST_START}-{HIST_END}: {hist['start_g_per_kwh']:.0f} -> "
          f"{hist['end_g_per_kwh']:.0f} g/kWh, {hist['annual_decline']:.2%}/yr decline "
          f"(Business as Usual uses {SCENARIOS['business_as_usual']['ef_decline']:.1%})")

    tables, projections = [], {}
    for key, s in SCENARIOS.items():
        p = project(s["growth"], s["ef_decline"])
        p.insert(0, "scenario", key)
        tables.append(p)
        last = p.iloc[-1]
        projections[key] = {
            "label": s["label"],
            "growth_rate": s["growth"],
            "ef_annual_decline": s["ef_decline"],
            "years": p["year"].tolist(),
            "twh": p["twh"].round(1).tolist(),
            "ef_kg_per_kwh": p["ef_kg_per_kwh"].round(4).tolist(),
            "co2_mt": p["co2_mt"].round(1).tolist(),
            "ef_2030_reduction_vs_2024_pct": round(100 * (1 - last["ef_kg_per_kwh"] / BASE_EMISSION_FACTOR), 1),
            "co2_2030_change_vs_2024_pct": round(100 * (last["co2_mt"] / base_co2 - 1), 1),
        }
        print(f"\n  {s['label']} (growth {s['growth']:.0%}/yr, EF decline {s['ef_decline']:.1%}/yr)")
        for _, r in p.iterrows():
            print(f"    {int(r.year)}: {r.twh:7.1f} TWh  x {r.ef_kg_per_kwh:.4f} kg/kWh = {r.co2_mt:6.1f} Mt CO2")

    by_year = pd.concat(tables)
    by_year.round(4).to_csv(os.path.join(TABLE_DIR, "projections_by_year.csv"), index=False)

    # Scenario assumptions table (paper Table 1)
    assumptions = pd.DataFrame([{
        "scenario": s["label"], "assumption": s["description"],
        "growth_pct_per_yr": round(100 * s["growth"], 1), "ef_decline_pct_per_yr": round(100 * s["ef_decline"], 1),
        "ef_2030_kg_per_kwh": round(projections[k]["ef_kg_per_kwh"][-1], 3),
        "twh_2030": round(projections[k]["twh"][-1]), "co2_2030_mt": round(projections[k]["co2_mt"][-1]),
    } for k, s in SCENARIOS.items()])
    assumptions.to_csv(os.path.join(TABLE_DIR, "scenario_assumptions.csv"), index=False)
    print("\nScenario assumptions:")
    print(assumptions.to_string(index=False))

    # Sensitivity of 2030 emissions to the baseline EF and growth rate
    sens = []
    for key, s in SCENARIOS.items():
        for ef0, dg in itertools.product(SENS_BASE_EF, SENS_GROWTH_DELTA):
            p = project(s["growth"] + dg, s["ef_decline"], base_ef=ef0)
            sens.append({"scenario": key, "base_ef": ef0, "growth": round(s["growth"] + dg, 3),
                         "twh_2030": round(p.iloc[-1]["twh"], 1), "co2_2030_mt": round(p.iloc[-1]["co2_mt"], 1)})
    sens = pd.DataFrame(sens)
    sens.to_csv(os.path.join(TABLE_DIR, "projection_sensitivity.csv"), index=False)
    sens_range = sens.groupby("scenario")["co2_2030_mt"].agg(["min", "max"]).round(1)
    print("\n2030 CO2 range across sensitivity cases (Mt):")
    print(sens_range.to_string())

    # Facility-level annual energy and CO2 (U.S. average grid factor)
    df = pd.read_csv(MERGED_CSV, index_col=0, parse_dates=True)
    yearly = df.groupby(df.index.year).agg(mean_total_kw=("total_power_kw", "mean"),
                                           mean_it_kw=("it_power_kw", "mean"),
                                           hours=("total_power_kw", "count"))
    yearly["pue_energy_weighted"] = yearly["mean_total_kw"] / yearly["mean_it_kw"]
    yearly = yearly[yearly["hours"] >= 4000]
    # Scale to a full year so years with missing hours are comparable
    yearly["annual_energy_mwh"] = yearly["mean_total_kw"] * 8760 / 1000
    yearly["annual_co2_tonnes"] = yearly["annual_energy_mwh"] * US_GRID_EMISSION_FACTOR
    yearly.round(3).to_csv(os.path.join(TABLE_DIR, "facility_yearly.csv"))

    with open(RESULTS_FILE) as f:
        results = json.load(f)
    results["projections"] = projections
    results["historical_grid_intensity"] = hist
    results["projection_baseline"] = {
        "base_year": BASE_YEAR, "base_twh": BASE_TWH, "base_ef_kg_per_kwh": BASE_EMISSION_FACTOR,
        "base_co2_mt": round(base_co2, 1),
        "iea_base_case_2030_twh": IEA_BASE_CASE_2030_TWH,
        "iea_base_case_implied_growth": round(iea_cagr, 4),
    }
    results["projection_sensitivity_2030_range"] = sens_range.to_dict(orient="index")
    results["facility_yearly"] = yearly.round(3).to_dict(orient="index")
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nProjections added to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()

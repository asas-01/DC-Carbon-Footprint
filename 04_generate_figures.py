"""
Author: asas
Description: Data Center Carbon Footprint Analysis

04_generate_figures.py - Generate all figures for the paper.

Figures 1-4:  literature context (values from published sources, see comments)
Figures 5-10: original NREL ESIF analysis (read from output/results.json and output/tables)
Figure 11:    projections (from 03_projections.py)
Figure S1:    rolling-origin R2 by test year (supplementary)
"""

import json
import os
import sys

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    MERGED_CSV, OUTPUT_DIR, RESULTS_FILE, TABLE_DIR, COLORS, SERVERS, SERVER_LIFETIME_YEARS,
    REFERENCE_GRIDS, US_GRID_EMISSION_FACTOR, IEA_BASE_CASE_2030_TWH, setup_plot_style, ensure_dirs,
)

plt = setup_plot_style()

BLUE, RED, GREEN, ORANGE = COLORS["blue"], COLORS["red"], COLORS["green"], COLORS["orange"]
GRAY, PURPLE, TEAL, LT_BLUE = COLORS["gray"], COLORS["purple"], COLORS["teal"], COLORS["light_blue"]
FEATURE_LABELS = {"outdoor_air_temp": "Temperature", "outdoor_air_humidity": "Humidity",
                  "it_power_kw": "IT Power"}
MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"  Saved {name} ({os.path.getsize(path) / 1024:.0f} KB)")


def fig01_global_electricity():
    # Historical and projected values approximated from IEA (2025) Energy and AI.
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    hist_years = list(range(2015, 2025))
    hist_twh = [200, 215, 225, 235, 260, 260, 290, 340, 375, 415]
    proj_years = list(range(2025, 2031))
    proj_low = [465, 520, 585, 655, 735, 820]
    proj_high = [500, 580, 680, 790, 910, 1050]
    proj_mid = [(l + h) / 2 for l, h in zip(proj_low, proj_high)]
    ax.bar(hist_years, hist_twh, color=BLUE, width=0.7, label="Historical (IEA)", zorder=3)
    ax.bar(proj_years, proj_mid, color=LT_BLUE, width=0.7, label="Projected (IEA mid-range)", zorder=3)
    ax.fill_between(proj_years, [l - 20 for l in proj_low], [h + 20 for h in proj_high],
                    alpha=0.15, color=BLUE, label="Projection range")
    # IEA (2025): ~945 TWh in 2030 is "slightly more than the entire electricity consumption of Japan today"
    ax.axhline(y=IEA_BASE_CASE_2030_TWH, color=RED, linestyle="--", lw=1.5, alpha=0.7)
    ax.text(2019.6, IEA_BASE_CASE_2030_TWH + 15,
            "IEA 2030 base case \u2248 945 TWh (slightly more than Japan's current electricity consumption)",
            fontsize=7, color=RED, va="bottom", ha="left")
    ax.set_xlabel("Year")
    ax.set_ylabel("Electricity Consumption (TWh)")
    ax.set_title("Global Data Center Electricity Consumption (2015\u20132030)", fontweight="bold", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlim(2014.2, 2031.5)
    ax.set_ylim(0, 1200)
    fig.tight_layout()
    save_fig(fig, "fig01_global_dc_electricity.png")


def fig02_ai_energy():
    # (a) Energy per request, Wh. Ranges reflect disagreement in published estimates:
    #   web search ~0.3 Wh (Google, 2009, as cited by de Vries 2023)
    #   LLM chat query: ~0.3 Wh (You, Epoch AI 2025) to ~2.9 Wh (de Vries 2023; EPRI 2024)
    #   image generation: ~2.9 Wh per image, mean across models (Luccioni et al., FAccT 2024)
    # (b) Rack power: 5-10 kW traditional vs 40-100 kW GPU training racks (EPRI).
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3))
    cats = ["Web\nsearch", "LLM chat\nquery", "AI image\ngeneration"]
    low = np.array([0.3, 0.3, 2.9])
    high = np.array([0.3, 2.9, 2.9])
    mid = (low + high) / 2
    ax1.bar(cats, high, color=[GREEN, ORANGE, RED], width=0.5, alpha=0.85, edgecolor="white")
    ax1.errorbar([1], [mid[1]], yerr=[[mid[1] - low[1]], [high[1] - mid[1]]],
                 fmt="none", ecolor="black", capsize=5, lw=1.2)
    labels = ["0.3 Wh", "0.3\u20132.9 Wh", "~2.9 Wh"]
    for i, t in enumerate(labels):
        ax1.text(i, high[i] + 0.12, t, ha="center", fontsize=8, fontweight="bold")
    ax1.set_ylabel("Energy per Request (Wh)")
    ax1.set_title("A) Energy per Request", fontweight="bold", fontsize=10)
    ax1.set_ylim(0, 3.8)

    racks = ["Traditional\nrack", "GPU training\nrack"]
    r_low, r_high = np.array([5, 40]), np.array([10, 100])
    r_mid = (r_low + r_high) / 2
    ax2.bar(racks, r_mid, color=[BLUE, RED], width=0.4, alpha=0.85, edgecolor="white")
    ax2.errorbar([0, 1], r_mid, yerr=[r_mid - r_low, r_high - r_mid], fmt="none",
                 ecolor="black", capsize=5, lw=1.2)
    for i in range(2):
        ax2.text(i, r_high[i] + 3, f"{r_low[i]}\u2013{r_high[i]} kW", ha="center", fontsize=8, fontweight="bold")
    ax2.set_ylabel("Power per Rack (kW)")
    ax2.set_title("B) Rack Power Density", fontweight="bold", fontsize=10)
    ax2.set_ylim(0, 120)
    fig.suptitle("Published Estimates of AI Energy Intensity", fontweight="bold", fontsize=11, y=1.03)
    fig.tight_layout()
    save_fig(fig, "fig02_ai_energy_comparison.png")


def fig03_embodied_vs_operational():
    hours = 8760 * SERVER_LIFETIME_YEARS
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.4), gridspec_kw={"width_ratios": [1, 1.25]})

    names = [s["name"].replace(" server", "\nserver") for s in SERVERS]
    emb = np.array([s["embodied_kg"] for s in SERVERS])
    op = np.array([s["avg_power_kw"] * hours * US_GRID_EMISSION_FACTOR for s in SERVERS])
    x = np.arange(len(SERVERS))
    w = 0.32
    ax1.bar(x - w / 2, emb, w, label="Embodied", color=ORANGE, edgecolor="white")
    ax1.bar(x + w / 2, op, w, label=f"Operational ({SERVER_LIFETIME_YEARS} yr)", color=BLUE, edgecolor="white")
    for i in range(len(SERVERS)):
        ax1.text(x[i] - w / 2, emb[i] + 150, f"{emb[i]:,.0f}", ha="center", fontsize=7)
        ax1.text(x[i] + w / 2, op[i] + 150, f"{op[i]:,.0f}", ha="center", fontsize=7)
        ax1.text(x[i], -1500, f"Embodied share\n{emb[i] / (emb[i] + op[i]):.0%}", ha="center",
                 fontsize=7, style="italic")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=8)
    ax1.set_ylabel("kg CO\u2082e over lifetime")
    ax1.set_ylim(-2300, 12500)
    ax1.axhline(0, color="black", lw=0.5)
    ax1.set_title(f"A) At U.S. average grid\n({US_GRID_EMISSION_FACTOR} kg CO\u2082/kWh)",
                  fontweight="bold", fontsize=9)
    ax1.legend(fontsize=7, loc="upper left")

    ef = np.linspace(0.0, 0.8, 400)
    label_offsets = [(8, -18), (10, 8)]
    for s, color, off in zip(SERVERS, [GRAY, PURPLE], label_offsets):
        op_curve = s["avg_power_kw"] * hours * ef
        share = s["embodied_kg"] / (s["embodied_kg"] + op_curve)
        ax2.plot(ef * 1000, share * 100, color=color, lw=2,
                 label=f"{s['name']} ({s['avg_power_kw'] * 1000:.0f} W)")
        breakeven = s["embodied_kg"] / (s["avg_power_kw"] * hours)
        ax2.plot(breakeven * 1000, 50, "o", color=color, ms=5)
        ax2.annotate(f"50% at {breakeven * 1000:.0f} g/kWh", (breakeven * 1000, 50), xytext=off,
                     textcoords="offset points", fontsize=7, color=color,
                     ha="left" if off[0] > 0 else "right")
    ax2.axhline(50, color="black", ls=":", lw=1)
    for name, val in REFERENCE_GRIDS.items():
        ax2.axvline(val * 1000, color=RED, ls="--", lw=0.8, alpha=0.6)
        ax2.text(val * 1000 + 6, 92, name, rotation=90, fontsize=6.5, color=RED, va="top")
    ax2.set_xlabel("Grid carbon intensity (g CO\u2082/kWh)")
    ax2.set_ylabel("Embodied share of lifetime emissions (%)")
    ax2.set_ylim(0, 100)
    ax2.set_xlim(0, 800)
    ax2.set_title("B) Embodied share vs. grid intensity", fontweight="bold", fontsize=9)
    ax2.legend(fontsize=6.5, loc="upper right")
    fig.tight_layout()
    save_fig(fig, "fig03_embodied_vs_operational.png")


def fig04_water():
    # Han et al. (2026), arXiv:2603.02705, Table 3, Baseline scenario, total U.S. data centers (MGD):
    #   2024 capacity 306 (low growth) / 357 (high growth); 2030 capacity 1,003 / 1,809;
    #   new capacity 2024->2030: 697 / 1,451. NYC average daily supply ~1,000 MGD (cited in Han et al.).
    cases = ["Low growth", "High growth"]
    cap_2024 = np.array([306, 357])
    new_cap = np.array([697, 1451])
    cap_2030 = np.array([1003, 1809])
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.arange(len(cases))
    ax.bar(x, cap_2024, width=0.45, color=BLUE, edgecolor="white", label="2024 peak capacity")
    ax.bar(x, new_cap, width=0.45, bottom=cap_2024, color=ORANGE, edgecolor="white",
           label="New capacity needed, 2024\u20132030")
    for i in range(len(cases)):
        ax.text(x[i], cap_2024[i] / 2, f"{cap_2024[i]:,}", ha="center", va="center", fontsize=9,
                color="white", fontweight="bold")
        ax.text(x[i], cap_2024[i] + min(new_cap[i] / 2, 330), f"+{new_cap[i]:,}", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
        ax.text(x[i], cap_2030[i] + 45, f"2030 total: {cap_2030[i]:,} MGD", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.axhline(1000, color=RED, ls="--", lw=1.4)
    ax.text(0.5, 1000 - 30, "New York City average\ndaily supply (~1,000 MGD)", color=RED, fontsize=8,
            ha="center", va="top")
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 2100)
    ax.set_ylabel("Peak water capacity (million gallons/day)")
    ax.set_title("U.S. Data Center Peak Water Capacity, 2024 and 2030 (Baseline Scenario)",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    save_fig(fig, "fig04_water_demand.png")


def fig05_pue_and_subsystems(df):
    monthly = pd.read_csv(os.path.join(TABLE_DIR, "monthly_subsystems.csv"), index_col=0, parse_dates=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 5.6), sharex=True)
    daily = df["pue"].resample("D").mean().dropna()
    ax1.plot(daily.index, daily.values, lw=0.5, color=BLUE, alpha=0.5, label="Daily average")
    ax1.plot(daily.index, daily.rolling(30, center=True, min_periods=10).mean(), lw=2, color=RED,
             label="30-day rolling mean")
    ax1.set_ylabel("PUE")
    ax1.set_title("A) Hourly-derived PUE, daily averages", fontweight="bold", fontsize=10)
    ax1.legend(fontsize=7, loc="upper left")

    for col, color, label in [("hvac_kw", RED, "HVAC"), ("pump_kw", TEAL, "Pumps"),
                              ("cooling_kw", BLUE, "Cooling"), ("plug_and_light_kw", GRAY, "Plug & light")]:
        ax2.plot(monthly.index, monthly[col], color=color, lw=1.6, label=label)
    ax2.set_ylabel("Overhead power (kW)")
    ax3 = ax2.twinx()
    ax3.plot(monthly.index, monthly["it_power_kw"], color="black", ls="--", lw=1.2, label="IT power (right)")
    ax3.set_ylabel("IT power (kW)")
    ax3.spines["right"].set_visible(True)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left", ncol=2)
    ax2.set_title("B) Monthly mean power by subsystem", fontweight="bold", fontsize=10)
    for a in (ax1, ax2):
        a.axvline(pd.Timestamp("2024-04-01"), color=PURPLE, ls=":", lw=1.2)
    ax2.text(pd.Timestamp("2024-04-25"), 62, "Apr 2024", fontsize=7, color=PURPLE, ha="left", va="center",
             bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9), zorder=5)
    for a in (ax1, ax2):
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        a.xaxis.set_major_locator(mdates.YearLocator())
        a.set_xlabel("Year")
    ax1.tick_params(axis="x", labelbottom=True)
    fig.tight_layout()
    save_fig(fig, "fig05_pue_timeseries.png")


def fig06_correlations(df, results):
    cols = ["outdoor_air_temp", "outdoor_air_humidity", "it_power_kw", "cooling_kw", "hvac_kw",
            "pump_kw", "pue"]
    labels = ["Temperature", "Humidity", "IT Power", "Cooling", "HVAC", "Pump", "PUE"]
    corr = df[cols].corr().to_numpy()
    fig, (ax_hm, ax_bar) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1.3, 1]})
    im = ax_hm.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax_hm.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7,
                       color="white" if abs(corr[i, j]) > 0.5 else "black")
    ax_hm.set_xticks(range(len(cols)))
    ax_hm.set_yticks(range(len(cols)))
    ax_hm.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax_hm.set_yticklabels(labels, fontsize=8)
    ax_hm.set_title("A) Pearson correlation matrix", fontweight="bold", fontsize=10)
    plt.colorbar(im, ax=ax_hm, shrink=0.8, label="Pearson r")

    c = results["correlations"]
    targets = [("cooling_kw", "Cooling"), ("pump_kw", "Pump"), ("hvac_kw", "HVAC"), ("pue", "PUE")]
    r = np.array([c[t]["r"] for t, _ in targets])
    lo = np.array([c[t]["block_bootstrap_ci_low"] for t, _ in targets])
    hi = np.array([c[t]["block_bootstrap_ci_high"] for t, _ in targets])
    y = np.arange(len(targets))
    ax_bar.barh(y, r, color=LT_BLUE, edgecolor=BLUE, height=0.5)
    ax_bar.errorbar(r, y, xerr=[r - lo, hi - r], fmt="none", ecolor="black", capsize=4, lw=1)
    for i, (t, _) in enumerate(targets):
        ax_bar.text(max(hi[i], 0) + 0.01, i, f"r = {r[i]:.3f} [{lo[i]:.3f}, {hi[i]:.3f}]\nr\u00B2 = {r[i] ** 2:.3f}",
                    va="center", fontsize=7)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels([l for _, l in targets])
    ax_bar.axvline(0, color="gray", lw=0.5)
    xmin = min(0, lo.min()) - 0.05
    ax_bar.set_xlim(xmin, max(0.45, hi.max() + 0.25))
    ax_bar.set_xlabel("Pearson r with outdoor temperature (95% block-bootstrap CI)")
    ax_bar.set_title("B) Temperature vs. energy subsystems", fontweight="bold", fontsize=10)
    fig.suptitle(f"Pearson Correlation Analysis (n = {len(df):,} hours)", fontweight="bold", fontsize=11, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig06_correlation_analysis.png")


def fig07_temp_vs_cooling(df):
    feats = ["outdoor_air_temp", "outdoor_air_humidity", "it_power_kw"]
    lr = LinearRegression().fit(df[feats], df["cooling_kw"])
    sample = df.sample(min(5000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    sc = ax.scatter(sample["outdoor_air_temp"], sample["cooling_kw"], c=sample["it_power_kw"],
                    alpha=0.35, s=8, cmap="viridis", edgecolors="none")
    plt.colorbar(sc, ax=ax, label="IT Power (kW)", shrink=0.8)
    t = np.linspace(sample["outdoor_air_temp"].min(), sample["outdoor_air_temp"].max(), 100)
    grid = pd.DataFrame({"outdoor_air_temp": t,
                         "outdoor_air_humidity": df["outdoor_air_humidity"].median(),
                         "it_power_kw": df["it_power_kw"].median()})
    ax.plot(t, lr.predict(grid), color=RED, lw=2.2,
            label="Linear fit (humidity and IT power at median)")
    ax.set_xlabel("Outside Air Temperature (\u00B0F)")
    ax.set_ylabel("Cooling Power (kW)")
    ax.set_title("Outside Air Temperature vs. Cooling Power", fontweight="bold", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    save_fig(fig, "fig07_temp_vs_cooling.png")


def fig08_monthly(df):
    colors = ["#4575B4", "#4575B4", "#74ADD1", "#ABD9E9", "#ABD9E9", "#F46D43",
              "#D73027", "#D73027", "#F46D43", "#ABD9E9", "#ABD9E9", "#4575B4"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3))
    for ax, col, ylabel, title in [(ax1, "cooling_kw", "Cooling Power (kW)", "A) Monthly Cooling Power"),
                                   (ax2, "pue", "PUE", "B) Monthly PUE")]:
        bp = ax.boxplot([df[df["local_month"] == m][col] for m in range(1, 13)], tick_labels=MONTHS,
                        patch_artist=True, widths=0.6, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Month")
        ax.set_title(title, fontweight="bold", fontsize=10)
    fig.suptitle("Monthly Distributions of Cooling Power and PUE", fontweight="bold", fontsize=11, y=1.03)
    fig.tight_layout()
    save_fig(fig, "fig08_seasonal_patterns.png")


def fig09_model_performance(results):
    mb = results["model_b"]
    algos = [("linear", "Linear\nRegression"), ("random_forest", "Random\nForest"),
             ("gradient_boosting", "Gradient\nBoosting")]
    schemes = [("random_split", "Random 80/20", LT_BLUE), ("chronological_split", "Chronological 80/20", BLUE),
               ("rolling_origin", "Rolling-origin (mean)", ORANGE)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.6))
    x = np.arange(len(algos))
    w = 0.26
    for k, (scheme, label, color) in enumerate(schemes):
        vals = [mb[a][scheme]["mean_r2"] if scheme == "rolling_origin" else mb[a][scheme]["r2"] for a, _ in algos]
        bars = ax1.bar(x + (k - 1) * w, vals, w, color=color, label=label, edgecolor="white")
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, max(v, 0) + 0.015, f"{v:.2f}", ha="center", fontsize=6.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([l for _, l in algos], fontsize=8)
    ax1.axhline(0, color="black", lw=0.5)
    ax1.set_ylabel("Test R\u00B2")
    ax1.set_ylim(min(-0.1, ax1.get_ylim()[0]), 1.0)
    ax1.set_title("A) Model B accuracy by validation method", fontweight="bold", fontsize=9)
    ax1.legend(fontsize=6.5, loc="upper left")

    imp = mb["rf_importance"]["rolling_origin_folds"]
    feats = mb["features"]
    y = np.arange(len(feats))
    h = 0.36
    imp_mean = [imp["impurity"][f]["share_mean"] for f in feats]
    imp_sd = [imp["impurity"][f]["share_sd"] for f in feats]
    perm_mean = [imp["permutation_r2_drop"][f]["share_mean"] for f in feats]
    perm_sd = [imp["permutation_r2_drop"][f]["share_sd"] for f in feats]
    ax2.barh(y - h / 2, imp_mean, h, xerr=imp_sd, color=GREEN, alpha=0.8, capsize=3, label="Impurity-based")
    ax2.barh(y + h / 2, perm_mean, h, xerr=perm_sd, color=PURPLE, alpha=0.8, capsize=3, label="Permutation")
    ax2.set_yticks(y)
    ax2.set_yticklabels([FEATURE_LABELS[f] for f in feats])
    ax2.set_xlim(0, 1.05)
    ax2.set_xlabel("Share of total importance (mean \u00B1 SD across folds)")
    ax2.set_title("B) Random Forest feature importance\n(rolling-origin folds)", fontweight="bold", fontsize=9)
    ax2.legend(fontsize=6.5, loc="lower right")
    fig.suptitle("Predicting Cooling Power (Model B)", fontweight="bold", fontsize=11, y=0.99)
    fig.tight_layout()
    save_fig(fig, "fig09_model_performance.png")


def fig10_actual_vs_predicted(results):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8), sharex=True, sharey=True)
    rf = results["model_b"]["random_forest"]
    for ax, key, scheme, title in [(axes[0], "random", "random_split", "A) Random 80/20 split"),
                                   (axes[1], "chrono", "chronological_split", "B) Chronological split")]:
        p = pd.read_csv(os.path.join(TABLE_DIR, f"model_b_rf_predictions_{key}.csv"), index_col=0)
        ax.scatter(p["actual"], p["predicted"], s=4, alpha=0.15, color=GREEN, edgecolors="none")
        mx = max(p["actual"].max(), p["predicted"].max())
        ax.plot([0, mx], [0, mx], "--", color=RED, lw=1.2, label="1:1 line")
        m = rf[scheme]
        ax.set_title(f"{title}\nR\u00B2 = {m['r2']:.3f}, MAE = {m['mae']:.2f} kW", fontweight="bold", fontsize=9)
        ax.set_xlabel("Actual Cooling Power (kW)")
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("Predicted Cooling Power (kW)")
    fig.suptitle("Random Forest: Actual vs. Predicted Cooling Power (test sets)", fontweight="bold",
                 fontsize=11, y=1.03)
    fig.tight_layout()
    save_fig(fig, "fig10_actual_vs_predicted.png")


def fig11_projections(results):
    proj = results["projections"]
    styles = {"business_as_usual": (BLUE, "-"), "accelerated_ai": (RED, "--"), "green_transition": (GREEN, "-.")}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.5))
    for key, (color, ls) in styles.items():
        d = proj[key]
        ax1.plot(d["years"], d["twh"], color=color, ls=ls, lw=2, marker="o", ms=3.5, label=d["label"])
        ax2.plot(d["years"], d["co2_mt"], color=color, ls=ls, lw=2, marker="o", ms=3.5, label=d["label"])
    ax1.plot(2030, IEA_BASE_CASE_2030_TWH, marker="*", color="black", ms=10, ls="none", label="IEA base case 2030")
    ax1.set_ylabel("Electricity (TWh)")
    ax1.set_title("A) Electricity demand", fontweight="bold", fontsize=10)
    ax2.set_ylabel("CO\u2082 Emissions (Mt)")
    ax2.set_title("B) CO\u2082 emissions", fontweight="bold", fontsize=10)
    for a in (ax1, ax2):
        a.set_xlabel("Year")
        a.grid(True, alpha=0.3)
        a.legend(fontsize=6.5, loc="upper left")
    fig.suptitle("Global Data Center Electricity and CO\u2082 Scenarios (2024\u20132030)",
                 fontweight="bold", fontsize=11, y=1.04)
    fig.tight_layout()
    save_fig(fig, "fig11_carbon_projections.png")


def figS1_rolling_origin(results):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for key, color, label in [("model_b", BLUE, "Model B (cooling)"), ("model_c", ORANGE, "Model C (overhead)")]:
        folds = results[key]["random_forest"]["rolling_origin"]["folds"]
        yrs = [int(y) for y in folds]
        ax.plot(yrs, [folds[str(y)]["r2"] for y in yrs], marker="o", color=color, lw=1.8, label=label)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Test year (model trained on all earlier years)")
    ax.set_ylabel("Random Forest test R\u00B2")
    ax.set_title("Rolling-Origin Validation by Test Year", fontweight="bold", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, "figS1_rolling_origin_r2.png")


def main():
    print("=" * 60)
    print("  STEP 4: Generate All Figures")
    print("=" * 60)
    ensure_dirs()
    df = pd.read_csv(MERGED_CSV, index_col=0, parse_dates=True).sort_index()
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    fig01_global_electricity()
    fig02_ai_energy()
    fig03_embodied_vs_operational()
    fig04_water()
    fig05_pue_and_subsystems(df)
    fig06_correlations(df, results)
    fig07_temp_vs_cooling(df)
    fig08_monthly(df)
    fig09_model_performance(results)
    fig10_actual_vs_predicted(results)
    fig11_projections(results)
    figS1_rolling_origin(results)
    print(f"\nFigures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

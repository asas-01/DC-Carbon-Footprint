"""
Author: asas
Description: Data Center Carbon Footprint Analysis

02_analysis.py - Correlation analysis and machine learning models.

Performs:
  - Descriptive statistics (Table 2)
  - Pearson correlations with Fisher-z and moving-block bootstrap 95% CIs
  - Model A: PUE from temperature + humidity
  - Model B: cooling power from temperature + humidity + IT power
  - Model C: total overhead power from temperature + humidity + IT power
    (each with Linear Regression, Random Forest and Gradient Boosting)
  - Three validation schemes:
      random      80/20 random split (the original method, kept for comparison)
      chrono      first 80% of hours for training, last 20% for testing
      rolling     rolling-origin: for each test year Y, train on all hours before Y
  - Random Forest feature importance (impurity + permutation) across folds
    and repeated random splits, to check how stable the rankings are
  - Sampling sensitivity: full-resolution vs every-10th-row hourly data

Results go to output/results.json and output/tables/*.csv.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    MERGED_CSV, MERGED_SUBSAMPLED_CSV, RESULTS_FILE, TABLE_DIR, OUTPUT_ROOT,
    TEST_SIZE, RANDOM_STATE, N_REPEATED_SPLITS, ROLLING_TEST_YEARS, PERMUTATION_REPEATS,
    BOOTSTRAP_REPS, BOOTSTRAP_BLOCK_HOURS, RF_PARAMS, GB_PARAMS, ensure_dirs,
)

TEMP, HUM, IT = "outdoor_air_temp", "outdoor_air_humidity", "it_power_kw"

MODELS = {
    "model_a": {"label": "PUE", "features": [TEMP, HUM], "target": "pue"},
    "model_b": {"label": "Cooling power (kW)", "features": [TEMP, HUM, IT], "target": "cooling_kw"},
    "model_c": {"label": "Total overhead power (kW)", "features": [TEMP, HUM, IT], "target": "overhead_kw"},
}

ALGORITHMS = {
    "linear": LinearRegression(),
    "random_forest": RandomForestRegressor(**RF_PARAMS),
    "gradient_boosting": GradientBoostingRegressor(**GB_PARAMS),
}


def load(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index()


# ── Correlation helpers ──────────────────────────────────────────────────
def fisher_ci(r, n, alpha=0.05):
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    q = stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - q * se)), float(np.tanh(z + q * se))


def block_bootstrap_ci(x, y, block, reps, seed=RANDOM_STATE):
    """Moving-block bootstrap CI for Pearson r (keeps short-range autocorrelation)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    rs = np.empty(reps)
    for i in range(reps):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = (starts[:, None] + offsets).ravel()[:n]
        rs[i] = np.corrcoef(x[idx], y[idx])[0, 1]
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def effective_n(x, y):
    """Approximate effective sample size for correlated AR(1)-like series."""
    rx = pd.Series(x).autocorr(1)
    ry = pd.Series(y).autocorr(1)
    return int(len(x) * (1 - rx * ry) / (1 + rx * ry)), float(rx), float(ry)


def correlation_table(df):
    rows = []
    for target in ["cooling_kw", "pump_kw", "hvac_kw", "plug_and_light_kw", "overhead_kw", "pue"]:
        x, y = df[TEMP].to_numpy(), df[target].to_numpy()
        r, p = stats.pearsonr(x, y)
        lo, hi = fisher_ci(r, len(x))
        blo, bhi = block_bootstrap_ci(x, y, BOOTSTRAP_BLOCK_HOURS, BOOTSTRAP_REPS)
        neff, rx, ry = effective_n(x, y)
        rows.append({
            "variable": target, "r": round(r, 4), "r_squared": round(r * r, 4),
            "p_value": float(f"{p:.2e}"), "n": len(x),
            "fisher_ci_low": round(lo, 4), "fisher_ci_high": round(hi, 4),
            "block_bootstrap_ci_low": round(blo, 4), "block_bootstrap_ci_high": round(bhi, 4),
            "lag1_autocorr_temp": round(rx, 3), "lag1_autocorr_target": round(ry, 3),
            "effective_n_approx": neff,
        })
    return pd.DataFrame(rows)


# ── Modeling helpers ─────────────────────────────────────────────────────
def metrics(y_true, y_pred):
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
    }


def split_random(X, y, seed=RANDOM_STATE):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)


def split_chrono(X, y):
    cut = int(len(X) * (1 - TEST_SIZE))
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


def rolling_folds(X, y):
    years = X.index.year
    for yr in ROLLING_TEST_YEARS:
        tr, te = years < yr, years == yr
        if tr.sum() and te.sum():
            yield yr, X[tr], X[te], y[tr], y[te]


def importance_record(model, X_te, y_te, features):
    # n_jobs=1: the forest already predicts in parallel, and copying it to workers is slow
    perm = permutation_importance(model, X_te, y_te, n_repeats=PERMUTATION_REPEATS,
                                  random_state=RANDOM_STATE, n_jobs=1)
    return {
        "impurity": dict(zip(features, model.feature_importances_.round(4).tolist())),
        "permutation_r2_drop": dict(zip(features, perm.importances_mean.round(4).tolist())),
    }


def summarize_importance(records, features):
    out = {}
    for kind in ["impurity", "permutation_r2_drop"]:
        mat = np.array([[rec[kind][f] for f in features] for rec in records])
        # Shares use non-negative importances; folds where nothing helps (sum = 0) are skipped
        pos = np.clip(mat, 0, None)
        tot = pos.sum(axis=1, keepdims=True)
        share = np.where(tot > 0, pos / np.where(tot > 0, tot, 1), np.nan)
        out[kind] = {
            f: {"mean": round(float(mat[:, i].mean()), 4), "sd": round(float(mat[:, i].std(ddof=1)), 4),
                "share_mean": round(float(np.nanmean(share[:, i])), 4),
                "share_sd": round(float(np.nanstd(share[:, i], ddof=1)), 4)}
            for i, f in enumerate(features)
        }
        out[kind]["n_folds_used_for_share"] = int(np.isfinite(share[:, 0]).sum())
        ranks = [tuple(np.array(features)[np.argsort(-row)]) for row in mat]
        out[kind]["rank_order_counts"] = {" > ".join(k): ranks.count(k) for k in set(ranks)}
    out["per_fold"] = records
    return out


def run_models(df):
    results, perf_rows, predictions = {}, [], {}
    for key, spec in MODELS.items():
        feats, target = spec["features"], spec["target"]
        data = df[feats + [target]].dropna()
        X, y = data[feats], data[target]
        res = {"label": spec["label"], "features": feats, "target": target, "n": len(data)}
        print(f"\n{key}: {spec['label']} ~ {', '.join(feats)}  (n={len(data):,})")

        for algo_name, algo in ALGORITHMS.items():
            algo_res = {}
            # random split
            Xtr, Xte, ytr, yte = split_random(X, y)
            m = clone(algo).fit(Xtr, ytr)
            algo_res["random_split"] = metrics(yte, m.predict(Xte))
            if key == "model_b" and algo_name == "random_forest":
                predictions["random"] = pd.DataFrame({"actual": yte, "predicted": m.predict(Xte)})
            # chronological split
            Xtr, Xte, ytr, yte = split_chrono(X, y)
            m = clone(algo).fit(Xtr, ytr)
            algo_res["chronological_split"] = metrics(yte, m.predict(Xte))
            algo_res["chronological_split"]["test_start"] = str(Xte.index.min())
            if key == "model_b" and algo_name == "random_forest":
                predictions["chrono"] = pd.DataFrame({"actual": yte, "predicted": m.predict(Xte)})
            # rolling origin
            folds = {}
            for yr, Xtr, Xte, ytr, yte in rolling_folds(X, y):
                m = clone(algo).fit(Xtr, ytr)
                folds[str(yr)] = metrics(yte, m.predict(Xte))
            r2s = [v["r2"] for v in folds.values()]
            algo_res["rolling_origin"] = {
                "folds": folds,
                "mean_r2": round(float(np.mean(r2s)), 4),
                "median_r2": round(float(np.median(r2s)), 4),
                "min_r2": round(float(np.min(r2s)), 4),
                "max_r2": round(float(np.max(r2s)), 4),
            }
            res[algo_name] = algo_res
            print(f"  {algo_name:18s} random R2={algo_res['random_split']['r2']:.3f}  "
                  f"chrono R2={algo_res['chronological_split']['r2']:.3f}  "
                  f"rolling mean R2={algo_res['rolling_origin']['mean_r2']:.3f}", flush=True)
            for scheme in ["random_split", "chronological_split"]:
                perf_rows.append({"model": key, "target": target, "algorithm": algo_name,
                                  "validation": scheme, **{k: algo_res[scheme][k] for k in ["r2", "mae", "rmse"]}})
            perf_rows.append({"model": key, "target": target, "algorithm": algo_name,
                              "validation": "rolling_origin_mean", "r2": algo_res["rolling_origin"]["mean_r2"]})

        # Feature importance stability for the Random Forest
        if key in ("model_b", "model_c"):
            print(f"  feature importance stability for {key}...", flush=True)
            rf = ALGORITHMS["random_forest"]
            rec_rolling = []
            for yr, Xtr, Xte, ytr, yte in rolling_folds(X, y):
                m = clone(rf).fit(Xtr, ytr)
                rec_rolling.append(importance_record(m, Xte, yte, feats))
            rec_random = []
            for s in range(N_REPEATED_SPLITS):
                Xtr, Xte, ytr, yte = split_random(X, y, seed=RANDOM_STATE + s)
                m = clone(rf).fit(Xtr, ytr)
                rec_random.append(importance_record(m, Xte, yte, feats))
            Xtr, Xte, ytr, yte = split_random(X, y)
            single = importance_record(clone(rf).fit(Xtr, ytr), Xte, yte, feats)
            res["rf_importance"] = {
                "original_random_split_seed42": single,
                "rolling_origin_folds": summarize_importance(rec_rolling, feats),
                "repeated_random_splits": summarize_importance(rec_random, feats),
            }
        results[key] = res
    return results, pd.DataFrame(perf_rows), predictions


def sampling_sensitivity(df_full):
    df_sub = load(MERGED_SUBSAMPLED_CSV)
    out = {"n_hours_full": len(df_full), "n_hours_every10th": len(df_sub)}
    common = df_full.index.intersection(df_sub.index)
    out["n_common_hours"] = len(common)
    diffs = {}
    for col in ["it_power_kw", "cooling_kw", "hvac_kw", "pump_kw", "pue", "outdoor_air_temp"]:
        d = (df_sub.loc[common, col] - df_full.loc[common, col])
        diffs[col] = {
            "mean_full": round(float(df_full[col].mean()), 4),
            "mean_every10th": round(float(df_sub[col].mean()), 4),
            "hourly_mean_abs_diff": round(float(d.abs().mean()), 4),
            "hourly_corr_full_vs_every10th": round(float(np.corrcoef(df_sub.loc[common, col],
                                                                       df_full.loc[common, col])[0, 1]), 4),
            "within_hour_sd_ratio": round(float(df_sub.loc[common, col].std() / df_full.loc[common, col].std()), 4),
        }
    out["variables"] = diffs
    for name, d in [("full", df_full), ("every10th", df_sub)]:
        r, _ = stats.pearsonr(d[TEMP], d["cooling_kw"])
        feats = [TEMP, HUM, IT]
        X, y = d[feats], d["cooling_kw"]
        rf = RandomForestRegressor(**RF_PARAMS)
        Xtr, Xte, ytr, yte = split_random(X, y)
        r2_rand = r2_score(yte, clone(rf).fit(Xtr, ytr).predict(Xte))
        Xtr, Xte, ytr, yte = split_chrono(X, y)
        r2_chr = r2_score(yte, clone(rf).fit(Xtr, ytr).predict(Xte))
        out[f"temp_cooling_r_{name}"] = round(float(r), 4)
        out[f"model_b_rf_random_r2_{name}"] = round(float(r2_rand), 4)
        out[f"model_b_rf_chrono_r2_{name}"] = round(float(r2_chr), 4)
    return out


def regime_summary(df):
    monthly = df[["it_power_kw", "cooling_kw", "hvac_kw", "pump_kw", "plug_and_light_kw",
                  "overhead_kw", "total_power_kw"]].resample("MS").mean()
    monthly["pue"] = monthly["total_power_kw"] / monthly["it_power_kw"]
    monthly.round(3).to_csv(os.path.join(TABLE_DIR, "monthly_subsystems.csv"))

    def period(a, b):
        sub = df.loc[a:b]
        return {c: round(float(sub[c].mean()), 3) for c in
                ["pue", "it_power_kw", "cooling_kw", "hvac_kw", "pump_kw", "plug_and_light_kw", "overhead_kw"]}

    before = period("2016-01-01", "2024-03-31")
    after = period("2024-04-01", "2025-12-31")
    d_over = after["overhead_kw"] - before["overhead_kw"]
    share = {c: round((after[c] - before[c]) / d_over, 3) for c in
             ["cooling_kw", "hvac_kw", "pump_kw", "plug_and_light_kw"]}
    return {"before_2024_04": before, "from_2024_04": after,
            "share_of_overhead_increase_by_subsystem": share}


def main():
    print("=" * 60)
    print("  STEP 2: Correlation and Machine Learning Analysis")
    print("=" * 60)
    ensure_dirs()
    df = load(MERGED_CSV)
    print(f"Loaded {len(df):,} hourly observations ({df.index.min()} -> {df.index.max()})")

    results = {"n_observations": len(df), "date_range_utc": [str(df.index.min()), str(df.index.max())]}
    clean_path = os.path.join(OUTPUT_ROOT, "cleaning_summary.json")
    if os.path.exists(clean_path):
        with open(clean_path) as f:
            results["cleaning"] = json.load(f)

    cols = ["it_power_kw", "cooling_kw", "hvac_kw", "pump_kw", "plug_and_light_kw", "overhead_kw",
            "pue", "outdoor_air_temp", "outdoor_air_humidity"]
    desc = df[cols].describe().T[["mean", "std", "min", "max"]].round(3)
    desc.to_csv(os.path.join(TABLE_DIR, "table2_descriptive.csv"))
    results["descriptive"] = desc.to_dict(orient="index")
    mean_it, mean_over = df["it_power_kw"].mean(), df["overhead_kw"].mean()
    results["overhead_ratios"] = {
        "overhead_as_pct_of_it": round(100 * mean_over / mean_it, 2),
        "overhead_as_pct_of_total": round(100 * mean_over / (mean_it + mean_over), 2),
        "energy_weighted_pue": round(float((mean_it + mean_over) / mean_it), 4),
    }

    print("\nPearson correlations with outdoor temperature:")
    corr = correlation_table(df)
    corr.to_csv(os.path.join(TABLE_DIR, "correlations_temperature.csv"), index=False)
    print(corr[["variable", "r", "fisher_ci_low", "fisher_ci_high", "block_bootstrap_ci_low",
                "block_bootstrap_ci_high", "r_squared", "effective_n_approx"]].to_string(index=False))
    results["correlations"] = corr.set_index("variable").to_dict(orient="index")

    model_results, perf, preds = run_models(df)
    results.update(model_results)
    perf.to_csv(os.path.join(TABLE_DIR, "model_performance.csv"), index=False)
    for k, v in preds.items():
        v.to_csv(os.path.join(TABLE_DIR, f"model_b_rf_predictions_{k}.csv"))

    print("\nSampling sensitivity (full vs every-10th row)...")
    results["sampling_sensitivity"] = sampling_sensitivity(df)
    print(json.dumps({k: v for k, v in results["sampling_sensitivity"].items() if k != "variables"}, indent=2))

    results["regime_change"] = regime_summary(df)
    print("\nPUE / subsystem change around April 2024:")
    print(json.dumps(results["regime_change"], indent=2))

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nAll results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()

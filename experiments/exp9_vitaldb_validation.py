"""Experiment 9: External validation on VitalDB surgical cases against radial arterial FloTrac reference."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ichd.external import (
    GE_PLETH_HARDWARE_DELAY_MS,
    SELECTED_CASES,
    cluster_bootstrap_ci,
    extract_vitaldb_case_windows,
    train_frozen_auricle_models,
)

OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    models = train_frozen_auricle_models()

    all_dfs = []
    for cid in SELECTED_CASES:
        t0 = time.time()
        df_c = extract_vitaldb_case_windows(cid, shift_ms=GE_PLETH_HARDWARE_DELAY_MS)
        if len(df_c) > 0:
            print(f"Case {cid}: extracted {len(df_c)} evaluation windows in {time.time() - t0:.1f}s")
            all_dfs.append(df_c)

    if not all_dfs:
        raise RuntimeError("No VitalDB cases successfully processed.")

    df_all = pd.concat(all_dfs, ignore_index=True)
    y_true = df_all["dlog_sv"].to_numpy(dtype=float)
    clusters = df_all["caseid"].to_numpy()

    preds = {}
    for name, (cols, reg) in models.items():
        X = df_all[cols].to_numpy(dtype=float)
        preds[name] = reg.predict(X)
        df_all[f"pred_{name}"] = preds[name]

    summary = {
        "n_cases": int(df_all["caseid"].nunique()),
        "n_windows": int(len(df_all)),
        "models": {},
        "comparisons": {},
        "per_subject_distributions": {},
    }

    for name in models:
        y_p = preds[name]
        ci_dict = cluster_bootstrap_ci(y_true, y_p, None, clusters)
        summary["models"][name] = {
            "r_pooled": ci_dict["r_obs"],
            "ci_95": [ci_dict["ci_low"], ci_dict["ci_high"]],
            "mae_log": float(np.mean(np.abs(y_true - y_p))),
            "rmse_log": float(np.sqrt(np.mean((y_true - y_p) ** 2))),
        }

    for base_name in ["HR_Only", "PAT_Finger"]:
        diff_ci = cluster_bootstrap_ci(y_true, preds["PPG_Finger"], preds[base_name], clusters)
        summary["comparisons"][f"PPG_Finger_vs_{base_name}"] = {
            "delta_r": diff_ci["delta_obs"],
            "ci_95": [diff_ci["delta_ci_low"], diff_ci["delta_ci_high"]],
            "p_value": diff_ci["delta_p_val"],
        }

    pat_vs_hr = cluster_bootstrap_ci(y_true, preds["PAT_Finger"], preds["HR_Only"], clusters)
    summary["comparisons"]["PAT_Finger_vs_HR_Only"] = {
        "delta_r": pat_vs_hr["delta_obs"],
        "ci_95": [pat_vs_hr["delta_ci_low"], pat_vs_hr["delta_ci_high"]],
        "p_value": pat_vs_hr["delta_p_val"],
    }

    for name in models:
        sub_rs = []
        for _, grp in df_all.groupby("caseid"):
            yt = grp["dlog_sv"].to_numpy()
            yp = grp[f"pred_{name}"].to_numpy()
            if len(yt) >= 10 and np.std(yt) > 1e-6 and np.std(yp) > 1e-6:
                sub_rs.append(float(pearsonr(yt, yp)[0]))
        sub_rs_arr = np.array(sub_rs)
        n_pos = int(np.sum(sub_rs_arr > 0))
        summary["per_subject_distributions"][name] = {
            "mean_r": float(np.mean(sub_rs_arr)),
            "median_r": float(np.median(sub_rs_arr)),
            "iqr_25_75": [float(np.percentile(sub_rs_arr, 25)), float(np.percentile(sub_rs_arr, 75))],
            "min_max": [float(np.min(sub_rs_arr)), float(np.max(sub_rs_arr))],
            "n_valid_subjects": int(len(sub_rs_arr)),
            "n_positive": n_pos,
            "pct_positive": float(100.0 * n_pos / len(sub_rs_arr)) if len(sub_rs_arr) > 0 else 0.0,
        }

    rhos = []
    for _, grp in df_all.groupby("caseid"):
        y = grp["dlog_sv"].to_numpy()
        if len(y) > 2:
            r1 = np.corrcoef(y[:-1], y[1:])[0, 1]
            if not np.isnan(r1):
                rhos.append(float(r1))
    mean_rho = float(np.mean(rhos)) if rhos else 0.0
    n_eff = int(round(len(df_all) * (1.0 - mean_rho) / (1.0 + mean_rho))) if rhos else len(df_all)
    summary["effective_sample_size"] = {
        "mean_lag1_autocorr": mean_rho,
        "n_effective": n_eff,
        "n_effective_per_case": float(n_eff / len(df_all.groupby("caseid"))) if len(df_all.groupby("caseid")) > 0 else 0.0,
    }

    with open(OUT_DIR / "exp9_vitaldb_validation_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    df_all.to_csv(OUT_DIR / "vitaldb_windows.csv.gz", index=False, compression="gzip")
    print(f"exp9_vitaldb_validation: cases={summary['n_cases']}, windows={summary['n_windows']}, n_eff={n_eff}")


if __name__ == "__main__":
    main()

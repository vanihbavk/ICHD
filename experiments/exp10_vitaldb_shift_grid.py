"""Experiment 10: VitalDB plethysmogram delay shift grid sensitivity audit (0 to 500 ms)."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ichd.external import (
    SELECTED_CASES,
    cache_case_raw,
    cluster_bootstrap_ci,
    extract_vitaldb_case_windows,
    train_frozen_auricle_models,
)

OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHIFT_GRID_MS = [0, 100, 200, 300, 380, 400, 500]


def run_shift_grid():
    for cid in SELECTED_CASES:
        cache_case_raw(cid)

    models = train_frozen_auricle_models()
    grid_results = {}

    for shift_ms in SHIFT_GRID_MS:
        dfs = []
        for cid in SELECTED_CASES:
            df_c = extract_vitaldb_case_windows(cid, shift_ms=shift_ms, allow_fetch=False)
            if len(df_c) > 0:
                dfs.append(df_c)

        if not dfs:
            grid_results[f"shift_{shift_ms}"] = {"n_windows": 0, "n_cases": 0}
            continue

        df_shift = pd.concat(dfs, ignore_index=True)
        y_true = df_shift["dlog_sv"].to_numpy(dtype=float)
        clusters = df_shift["caseid"].to_numpy()

        preds = {}
        for name, (cols, reg) in models.items():
            X = df_shift[cols].to_numpy(dtype=float)
            preds[name] = reg.predict(X)
            df_shift[f"pred_{name}"] = preds[name]

        shift_entry = {
            "shift_ms": shift_ms,
            "n_cases": int(df_shift["caseid"].nunique()),
            "n_windows": int(len(df_shift)),
            "models": {},
            "comparisons": {},
            "per_subject": {},
        }

        for name in models:
            yp = preds[name]
            ci_dict = cluster_bootstrap_ci(y_true, yp, None, clusters)
            shift_entry["models"][name] = {
                "r_pooled": round(ci_dict["r_obs"], 3),
                "ci_95": [round(ci_dict["ci_low"], 3), round(ci_dict["ci_high"], 3)],
                "mae": round(float(np.mean(np.abs(y_true - yp))), 3),
            }

            sub_rs = []
            for _, grp in df_shift.groupby("caseid"):
                yt_s = grp["dlog_sv"].to_numpy()
                yp_s = grp[f"pred_{name}"].to_numpy()
                if len(yt_s) >= 10 and np.std(yt_s) > 1e-6 and np.std(yp_s) > 1e-6:
                    sub_rs.append(float(pearsonr(yt_s, yp_s)[0]))
            sub_rs_arr = np.array(sub_rs)
            n_pos = int(np.sum(sub_rs_arr > 0))
            shift_entry["per_subject"][name] = {
                "median_r": round(float(np.median(sub_rs_arr)), 3) if len(sub_rs_arr) else None,
                "n_positive": n_pos,
                "n_total": len(sub_rs_arr),
                "pct_positive": round(100.0 * n_pos / len(sub_rs_arr), 1) if len(sub_rs_arr) else None,
            }

        comp1 = cluster_bootstrap_ci(y_true, preds["PPG_Finger"], preds["HR_Only"], clusters)
        shift_entry["comparisons"]["PPG_Finger_vs_HR_Only"] = {
            "delta_r": round(comp1["delta_obs"], 3),
            "ci_95": [round(comp1["delta_ci_low"], 3), round(comp1["delta_ci_high"], 3)],
            "p_value": round(comp1["delta_p_val"], 4),
        }
        comp2 = cluster_bootstrap_ci(y_true, preds["PAT_Finger"], preds["HR_Only"], clusters)
        shift_entry["comparisons"]["PAT_Finger_vs_HR_Only"] = {
            "delta_r": round(comp2["delta_obs"], 3),
            "ci_95": [round(comp2["delta_ci_low"], 3), round(comp2["delta_ci_high"], 3)],
            "p_value": round(comp2["delta_p_val"], 4),
        }

        grid_results[f"shift_{shift_ms}"] = shift_entry

    out_file = OUT_DIR / "exp10_vitaldb_shift_grid_results.json"
    with open(out_file, "w") as f:
        json.dump(grid_results, f, indent=2)
    print(f"exp10_vitaldb_shift_grid: wrote {out_file.name}")


if __name__ == "__main__":
    run_shift_grid()

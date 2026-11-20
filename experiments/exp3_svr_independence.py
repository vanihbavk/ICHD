"""Experiment 3: Systemic vascular resistance (SVR) vs. inverted flow (-dlogCO) partial-correlation analysis."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold

from ichd.model import ICHDModel, CONFIG_FEATURES

ROOT = Path(__file__).resolve().parents[1]
TABLES_CACHE = ROOT / "data" / "interim" / "ichd_tables"
OUT_DIR = ROOT / "results"


def partial_corr(x: np.ndarray, y: np.ndarray, covar: np.ndarray) -> tuple[float, float]:
    """Compute partial correlation between x and y controlling for covar."""
    # Linear regression residuals
    slope_x, int_x, _, _, _ = stats.linregress(covar, x)
    res_x = x - (int_x + slope_x * covar)
    
    slope_y, int_y, _, _, _ = stats.linregress(covar, y)
    res_y = y - (int_y + slope_y * covar)
    
    r, p = stats.pearsonr(res_x, res_y)
    return float(r), float(p)


def run_svr_independence_analysis():
    ep = pd.read_pickle(TABLES_CACHE / "episodes.pkl")
    
    # 1. True reference coupling: dlogSVR vs -dlogCO
    y_svr = ep["dlog_svr"].to_numpy(dtype=float)
    y_co = ep["dlog_co"].to_numpy(dtype=float)
    neg_co = -y_co
    
    r_overall, p_overall = stats.pearsonr(y_svr, neg_co)
    
    # Maneuver-specific true coupling
    maneuver_coupling = {}
    for act in ["exercise", "cold_pressor", "valsalva", "breath_hold"]:
        sub = ep[ep["act_name"] == act]
        r_act, p_act = stats.pearsonr(sub["dlog_svr"], -sub["dlog_co"])
        maneuver_coupling[act] = {
            "n": len(sub),
            "r_svr_vs_neg_co": round(float(r_act), 3),
            "p_val": round(float(p_act), 4),
            "dlog_co_mean": round(float(sub["dlog_co"].mean()), 3),
            "dlog_svr_mean": round(float(sub["dlog_svr"].mean()), 3),
            "dlog_map_mean": round(float(sub["dlog_map"].mean()), 3),
        }
        
    # 2. Out-of-fold CV predictions for Chest-SCG and ICHD-Full
    gkf = GroupKFold(n_splits=5)
    groups = ep["subject"].values
    
    pred_svr = {"Chest-ECG+SCG": np.zeros(len(ep)), "ICHD-Full": np.zeros(len(ep))}
    pred_co = {"Chest-ECG+SCG": np.zeros(len(ep)), "ICHD-Full": np.zeros(len(ep))}
    
    for train_idx, test_idx in gkf.split(ep, groups=groups):
        train_df, test_df = ep.iloc[train_idx], ep.iloc[test_idx]
        for cfg in ["Chest-ECG+SCG", "ICHD-Full"]:
            model = ICHDModel(features=CONFIG_FEATURES[cfg], kind="hybrid")
            model.fit(train_df)
            p = model.predict(test_df)
            pred_svr[cfg][test_idx] = p.dlog_svr
            pred_co[cfg][test_idx] = p.dlog_co
            
    model_skills = {}
    for cfg in ["Chest-ECG+SCG", "ICHD-Full"]:
        p_svr = pred_svr[cfg]
        p_co = pred_co[cfg]
        
        # Raw correlations
        r_svr_raw, _ = stats.pearsonr(y_svr, p_svr)
        r_co_raw, _ = stats.pearsonr(y_co, p_co)
        
        # Partial correlation: skill of SVR head controlling for true CO
        part_r_true_co, part_p_true_co = partial_corr(p_svr, y_svr, y_co)
        
        # Partial correlation: skill of SVR head controlling for predicted CO
        part_r_pred_co, part_p_pred_co = partial_corr(p_svr, y_svr, p_co)
        
        model_skills[cfg] = {
            "r_svr_raw": round(float(r_svr_raw), 3),
            "r_co_raw": round(float(r_co_raw), 3),
            "partial_r_svr_given_true_co": round(float(part_r_true_co), 3),
            "partial_p_svr_given_true_co": round(float(part_p_true_co), 4),
            "partial_r_svr_given_pred_co": round(float(part_r_pred_co), 3),
            "partial_p_svr_given_pred_co": round(float(part_p_pred_co), 4),
        }

    results = {
        "true_overall_r_svr_vs_neg_co": round(float(r_overall), 3),
        "true_overall_p_val": round(float(p_overall), 4),
        "maneuver_true_coupling": maneuver_coupling,
        "model_skills": model_skills,
    }
    
    with open(OUT_DIR / "exp3_svr_independence_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"exp3_svr_independence: partial_r_true_co={model_skills['ICHD-Full']['partial_r_svr_given_true_co']:+.3f}")


if __name__ == "__main__":
    run_svr_independence_analysis()

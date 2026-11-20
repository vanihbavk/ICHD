"""Experiment 2: Clinical 4-Quadrant Trending Agreement & Concordance Metrics.

Implements the Intensive Care Medicine / Critchley & Critchley cardiac-output trending standard:
evaluates 4-quadrant concordance rate and polar trending agreement outside a clinical exclusion zone (10%)
for stroke volume and cardiac output across all 5 sensor configurations.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from ichd.model import ICHDModel, CONFIG_FEATURES

ROOT = Path(__file__).resolve().parents[1]
TABLES_CACHE = ROOT / "data" / "interim" / "ichd_tables"
OUT_DIR = ROOT / "results"


def compute_concordance(y_true: np.ndarray, y_pred: np.ndarray, exclusion_threshold: float = 0.10) -> tuple[float, int, int]:
    """Compute 4-quadrant concordance rate outside an exclusion zone.
    
    Returns:
        concordance_rate: fraction of concordant directional changes (in %)
        n_eval: number of pairs outside exclusion zone
        n_total: total number of valid pairs
    """
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[m], y_pred[m]
    
    # Exclusion zone based on reference magnitude
    eval_mask = np.abs(yt) > exclusion_threshold
    n_eval = int(np.sum(eval_mask))
    n_total = len(yt)
    
    if n_eval == 0:
        return 0.0, 0, n_total
    
    concordant = np.sum((yt[eval_mask] * yp[eval_mask]) > 0)
    concordance_rate = float((concordant / n_eval) * 100.0)
    return round(concordance_rate, 1), n_eval, n_total


def run_trending_analysis():
    ep = pd.read_pickle(TABLES_CACHE / "episodes.pkl")
    
    # 5-fold subject-disjoint split
    gkf = GroupKFold(n_splits=5)
    groups = ep["subject"].values
    
    configs = ["ECG-Only", "PAT-Only", "Chest-ECG+SCG", "PPG-Only", "ICHD-Full"]
    preds = {cfg: {"dlog_sv": np.zeros(len(ep)), "dlog_co": np.zeros(len(ep)), "dlog_svr": np.zeros(len(ep))} for cfg in configs}
    
    for train_idx, test_idx in gkf.split(ep, groups=groups):
        train_df, test_df = ep.iloc[train_idx], ep.iloc[test_idx]
        for cfg in configs:
            feats = CONFIG_FEATURES[cfg]
            model = ICHDModel(features=feats, kind="hybrid")
            model.fit(train_df)
            pred = model.predict(test_df)
            preds[cfg]["dlog_sv"][test_idx] = pred.dlog_sv
            preds[cfg]["dlog_co"][test_idx] = pred.dlog_co
            preds[cfg]["dlog_svr"][test_idx] = pred.dlog_svr

    results = {}
    # Exclusion zones: 10% change (|dlog| > 0.095) and 15% change (|dlog| > 0.14)
    thresholds = {"10pct": 0.095, "15pct": 0.140}
    
    for cfg in configs:
        results[cfg] = {}
        for metric in ["dlog_sv", "dlog_co", "dlog_svr"]:
            y_true = ep[metric].values
            y_pred = preds[cfg][metric]
            
            # Overall Pearson correlation
            valid = np.isfinite(y_true) & np.isfinite(y_pred)
            r = float(np.corrcoef(y_true[valid], y_pred[valid])[0, 1])
            
            conc_10, n10, ntot = compute_concordance(y_true, y_pred, thresholds["10pct"])
            conc_15, n15, _ = compute_concordance(y_true, y_pred, thresholds["15pct"])
            
            results[cfg][metric] = {
                "r": round(r, 3),
                "concordance_10pct": conc_10,
                "n_eval_10pct": n10,
                "concordance_15pct": conc_15,
                "n_eval_15pct": n15,
                "n_total": ntot,
            }
            
    with open(OUT_DIR / "exp2_trending_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"exp2_trending_concordance: full_conc_10pct={results['ICHD-Full']['dlog_sv']['concordance_10pct']:.1f}%")


if __name__ == "__main__":
    run_trending_analysis()

"""Experiment 6: Post-Hoc 20% Subject-Stratified Holdout Evaluation.

Evaluates a post-hoc 20% subject-stratified holdout split (seed 20261110)
alongside 5-fold CV to verify consistent modality ordering.
Trains exclusively on 80% development subjects and tests once on the 20% holdout subjects.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, accuracy_score

from ichd.model import ICHDModel, CONFIG_FEATURES

ROOT = Path(__file__).resolve().parents[1]
TABLES_CACHE = ROOT / "data" / "interim" / "ichd_tables"
OUT_DIR = ROOT / "results"


def run_locked_holdout_analysis(seed: int = 20261110):
    ep = pd.read_pickle(TABLES_CACHE / "episodes.pkl")
    
    # Stratified split by cohort across unique subjects
    sub_df = ep.groupby("subject")["cohort"].first().reset_index()
    rng = np.random.default_rng(seed)
    
    hc_subs = sub_df[sub_df["cohort"] == "HC"]["subject"].values
    hfc_subs = sub_df[sub_df["cohort"] == "HFC"]["subject"].values
    
    n_test_hc = int(np.round(len(hc_subs) * 0.20))
    n_test_hfc = int(np.round(len(hfc_subs) * 0.20))
    
    test_hc = rng.choice(hc_subs, size=n_test_hc, replace=False)
    test_hfc = rng.choice(hfc_subs, size=n_test_hfc, replace=False)
    test_subjects = set(np.concatenate([test_hc, test_hfc]))
    
    train_mask = ~ep["subject"].isin(test_subjects)
    test_mask = ep["subject"].isin(test_subjects)
    
    train_df = ep[train_mask].copy()
    test_df = ep[test_mask].copy()
    
    configs = ["ECG-Only", "PAT-Only", "Chest-ECG+SCG", "PPG-Only", "ICHD-Full"]
    holdout_results = {}
    
    for cfg in configs:
        feats = CONFIG_FEATURES[cfg]
        model = ICHDModel(features=feats, kind="hybrid", enforce_identity=True)
        model.fit(train_df)
        pred = model.predict(test_df)
        
        y_sv = test_df["dlog_sv"].values
        y_co = test_df["dlog_co"].values
        y_svr = test_df["dlog_svr"].values
        y_dom = test_df["y_res_dom"].values
        
        r_sv, _ = stats.pearsonr(y_sv, pred.dlog_sv)
        r_co, _ = stats.pearsonr(y_co, pred.dlog_co)
        r_svr, _ = stats.pearsonr(y_svr, pred.dlog_svr)
        
        auc = float(roc_auc_score(y_dom, pred.prob_res_dom))
        acc = float(accuracy_score(y_dom, (pred.prob_res_dom > 0.5).astype(int)))
        
        # Verify exact manifold closure
        gap = float(np.mean(np.abs(pred.dlog_map - (test_df["dlog_hr"].values + pred.dlog_sv + pred.dlog_svr))))
        
        holdout_results[cfg] = {
            "r_sv": round(float(r_sv), 3),
            "r_co": round(float(r_co), 3),
            "r_svr": round(float(r_svr), 3),
            "auc": round(auc, 3),
            "accuracy_pct": round(acc * 100.0, 1),
            "closure_gap_nat": round(gap, 15),
        }

    n_total_subjects = int(ep["subject"].nunique())
    summary = {
        "seed": seed,
        "n_total_subjects": n_total_subjects,
        "n_train_subjects": int(np.sum(~sub_df["subject"].isin(test_subjects))),
        "n_test_subjects": len(test_subjects),
        "n_test_hc": int(n_test_hc),
        "n_test_hfc": int(n_test_hfc),
        "n_train_episodes": len(train_df),
        "n_test_episodes": len(test_df),
        "test_subjects": sorted(list(test_subjects)),
        "holdout_results": holdout_results,
    }

    with open(OUT_DIR / "exp6_locked_holdout_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"exp6_locked_holdout: test={len(test_subjects)} subj ({len(test_df)} ep), r_sv={holdout_results['ICHD-Full']['r_sv']:.3f}")


if __name__ == "__main__":
    run_locked_holdout_analysis()

"""Master evaluation suite for Identity-Constrained Hemodynamic Decomposition (Experiments 1-5).

Computes statistical benchmarks, subject-clustered bootstrap 95% confidence
intervals, subject-disjoint Leave-One-Maneuver-Out (SD-LOMO) evaluations,
projected vs. unprojected manifold ablations, conformal risk-coverage curves,
and exact 2D conformal box certifications; writes JSON to `results/ichd_results.json`.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as _mp
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GroupKFold

from .episodes import build_deltas_and_episodes
from .model import (
    ICHDModel, ICHDPrediction, CONFIG_FEATURES,
    FEATURES_SCG_ONLY, FEATURES_EAR_PPG, FEATURES_CHEST_EAR_PPG,
    FEATURES_EAR_OPTICAL_ONLY, FEATURES_FINGER_OPTICAL_ONLY,
    FEATURES_EAR_NO_AMP, FEATURES_FINGER_NO_AMP,
)
from .conformal import confidence_score, risk_coverage_curve, calibrate_2d_conformal, quality_col_for_config
from .stats import (
    _paired_dr_ci,
    _paired_did_dr_ci,
    _paired_within_maneuver_dr_ci,
    _incremental_r_sv_beyond_oracle,
    _holm_bonferroni_correction,
    _eval_predictions,
    _fit_predict_fold,
    _oof_cv,
    _maneuver_oracle_oof,
)

ROOT = Path(__file__).resolve().parents[2]
TABLES_CACHE = ROOT / "data" / "interim" / "ichd_tables"
RESULTS_DIR = ROOT / "results"


def _hr_only_oof_metrics(df_in: pd.DataFrame) -> dict:
    gkf_hr = GroupKFold(n_splits=5)
    prob_hr = np.zeros(len(df_in))
    sv_hr = np.zeros(len(df_in))
    x_hr = df_in[["dlog_hr"]].fillna(0.0).to_numpy(dtype=float)
    y_hr = df_in["y_res_dom"].to_numpy(dtype=int)
    y_sv_in = df_in["dlog_sv"].to_numpy(dtype=float)
    g_hr = df_in["subject"].to_numpy()
    for tr_i, te_i in gkf_hr.split(df_in, groups=g_hr):
        clf_hr = LogisticRegression(max_iter=1000).fit(x_hr[tr_i], y_hr[tr_i])
        prob_hr[te_i] = clf_hr.predict_proba(x_hr[te_i])[:, 1]
        sv_hr[te_i] = Ridge(alpha=1.0).fit(x_hr[tr_i], y_sv_in[tr_i]).predict(x_hr[te_i])
    return {
        "dom_auc": float(roc_auc_score(y_hr, prob_hr)),
        "dom_acc": float(accuracy_score(y_hr, prob_hr >= 0.5)),
        "r_sv": float(pearsonr(y_sv_in, sv_hr)[0]),
        "res_dom_majority_frac": float(y_hr.mean()),
    }


def _run_exp1_biophysics(w: pd.DataFrame, ep: pd.DataFrame) -> dict:
    """Dataset accounting, PCG-SCG biophysical verification, and maneuver summaries."""
    hc_subs = w[w["cohort"] == "HC"]["subject"].nunique()
    hfc_subs = w[w["cohort"] == "HFC"]["subject"].nunique()
    clean_w = w[w["scg_q"] >= 0.40].copy()

    valid_ac_s2 = clean_w[
        clean_w["d_ac_lag"].notna() & clean_w["d_pcg_s2"].notna() & (clean_w["d_pcg_s2"].abs() < 60.0)
    ]
    ac_s2_shift_err = (valid_ac_s2["d_ac_lag"] - valid_ac_s2["d_pcg_s2"]).abs()
    valid_ao_s1 = clean_w[
        clean_w["d_pep_lag"].notna() & clean_w["d_pcg_s1"].notna() & (clean_w["d_pcg_s1"].abs() < 40.0)
    ]
    ao_s1_shift_err = (valid_ao_s1["d_pep_lag"] - valid_ao_s1["d_pcg_s1"]).abs()

    lvet_scg_val = clean_w["lvet_lag"]
    lvet_pcg_val = clean_w["pcg_s2"] - clean_w["pcg_s1"]
    m_lv = (
        (clean_w["scg_q"] >= 0.55)
        & (clean_w["scg_tc_dia"] >= 0.65)
        & lvet_scg_val.notna()
        & lvet_pcg_val.notna()
        & lvet_pcg_val.between(180, 400)
    )

    co_closure_err = float(np.max(np.abs(ep["dlog_co"] - ep["dlog_co_cnap"])))
    map_closure_err = float(np.max(np.abs(ep["dlog_map"] - (ep["dlog_co"] + ep["dlog_svr"]))))
    pat_valid = ep["d_ppgf_pat"].notna() & ep["d_pep_lag"].notna() & ep["d_ppgf_ptt"].notna()
    pat_closure_err = float(np.max(np.abs(
        ep.loc[pat_valid, "d_ppgf_pat"] - (ep.loc[pat_valid, "d_pep_lag"] + ep.loc[pat_valid, "d_ppgf_ptt"])
    )))
    corr_hr_ecg_cnap = float(pearsonr(ep["dlog_hr"], ep["dlog_hr_ref"])[0])

    maneuver_summary = {}
    for act_name, grp in ep.groupby("act_name"):
        maneuver_summary[str(act_name)] = {
            "n_episodes": int(len(grp)),
            "d_map_mmhg_mean": float(grp["d_map_mmhg"].mean()),
            "d_sbp_mmhg_mean": float(grp["d_sbp_mmhg"].mean()),
            "d_hr_bpm_mean": float(grp["d_hr_bpm"].mean()),
            "d_sv_ml_mean": float(grp["d_sv_ml"].mean()),
            "d_co_lpm_mean": float(grp["d_co_lpm"].mean()),
            "d_svr_dyn_mean": float(grp["d_svr_dyn"].mean()),
            "dlog_map_mean": float(grp["dlog_map"].mean()),
            "dlog_hr_mean": float(grp["dlog_hr"].mean()),
            "dlog_sv_mean": float(grp["dlog_sv"].mean()),
            "dlog_co_mean": float(grp["dlog_co"].mean()),
            "dlog_svr_mean": float(grp["dlog_svr"].mean()),
            "res_dom_frac": float(grp["y_res_dom"].mean()),
            "d_pep_ms_mean": float(grp["d_pep_lag"].mean()),
            "d_lvet_ms_mean": float(grp["d_lvet_lag"].mean()),
            "d_ppgf_pat_ms_mean": float(grp["d_ppgf_pat"].mean()),
            "d_ppgf_ptt_ms_mean": float(grp["d_ppgf_ptt"].mean()),
            "d_ppge_pat_ms_mean": float(grp["d_ppge_pat"].mean()),
            "d_ppge_ptt_ms_mean": float(grp["d_ppge_ptt"].mean()),
        }

    ex_hc = ep[(ep["cohort"] == "HC") & (ep["act"] == 3)]
    ex_hfc = ep[(ep["cohort"] == "HFC") & (ep["act"] == 3)]
    excl_recs = sorted(set(w["record"].unique()) - set(ep["record"].unique()))
    return {
        "n_sessions": int(w["record"].nunique()),
        "n_subjects": int(w["subject"].nunique()),
        "n_hc_subjects": int(hc_subs),
        "n_hfc_subjects": int(hfc_subs),
        "n_episode_sessions": int(ep["record"].nunique()),
        "n_episode_subjects": int(ep["subject"].nunique()),
        "n_episode_hc_subjects": int(ep[ep["cohort"] == "HC"]["subject"].nunique()),
        "n_episode_hfc_subjects": int(ep[ep["cohort"] == "HFC"]["subject"].nunique()),
        "excluded_rest_only_sessions": excl_recs,
        "n_windows": int(len(w)),
        "n_active_windows": int((w["act"].isin([1, 2, 3, 4]) & (w["scg_q"] >= 0.25)).sum()),
        "n_episodes": int(len(ep)),
        "n_hc_episodes": int((ep["cohort"] == "HC").sum()),
        "n_hfc_episodes": int((ep["cohort"] == "HFC").sum()),
        "max_co_closure_err": co_closure_err,
        "max_map_closure_err": map_closure_err,
        "max_pat_closure_err": pat_closure_err,
        "corr_hr_ecg_vs_cnap": corr_hr_ecg_cnap,
        "scg_amp_pp_mg_median": float(np.exp(w["scg_amp_log"].median()) * 1000.0),
        "scg_rms_sys_mg_median": float(np.exp(w["scg_rms_sys_log"].median()) * 1000.0),
        "ac_vs_s2_median_abs_ms": float(np.median(ac_s2_shift_err)),
        "ac_vs_s2_iqr_ms": [float(np.percentile(ac_s2_shift_err, 25)), float(np.percentile(ac_s2_shift_err, 75))],
        "ao_vs_s1_median_abs_ms": float(np.median(ao_s1_shift_err)),
        "ao_vs_s1_iqr_ms": [float(np.percentile(ao_s1_shift_err, 25)), float(np.percentile(ao_s1_shift_err, 75))],
        "lvet_scg_vs_pcg_r": float(pearsonr(lvet_scg_val[m_lv], lvet_pcg_val[m_lv])[0]),
        "corr_pep_sv_ep": float(pearsonr(ep["d_pep_lag"].fillna(0), ep["dlog_sv"])[0]),
        "corr_lvet_sv_ep": float(pearsonr(ep["d_lvet_lag"].fillna(0), ep["dlog_sv"])[0]),
        "corr_weissler_sv_ep": float(pearsonr(ep["d_weissler"].fillna(0), ep["dlog_sv"])[0]),
        "corr_scg_amp_sv_ep": float(pearsonr(ep["d_scg_amp_log"].fillna(0), ep["dlog_sv"])[0]),
        "corr_ppgf_amp_sv_ep": float(pearsonr(ep["dlog_ppgf_amp"].fillna(0), ep["dlog_sv"])[0]),
        "corr_ppge_amp_sv_ep": float(pearsonr(ep["dlog_ppge_amp"].fillna(0), ep["dlog_sv"])[0]),
        "corr_hr_sv_ep": float(pearsonr(ep["dlog_hr"], ep["dlog_sv"])[0]),
        "maneuvers": maneuver_summary,
        "hc_vs_hfc_exercise": {
            "hc_d_co_lpm": float(ex_hc["d_co_lpm"].median()),
            "hfc_d_co_lpm": float(ex_hfc["d_co_lpm"].median()),
            "hc_dlog_co": float(ex_hc["dlog_co"].mean()),
            "hfc_dlog_co": float(ex_hfc["dlog_co"].mean()),
            "hc_d_pep_ms": float(ex_hc["d_pep_lag"].mean()),
            "hfc_d_pep_ms": float(ex_hfc["d_pep_lag"].mean()),
            "hc_d_svr_dyn": float(ex_hc["d_svr_dyn"].median()),
            "hfc_d_svr_dyn": float(ex_hfc["d_svr_dyn"].median()),
        },
    }


def _run_exp2_cv(w: pd.DataFrame, ep: pd.DataFrame) -> tuple[dict, dict[str, ICHDPrediction]]:
    """5-fold subject-disjoint CV, identity ablations, controls, and active-window CV."""
    cv_results: dict = {}
    oof_preds_hybrid: dict[str, ICHDPrediction] = {}
    for kind in ("hybrid", "ridge", "gbm"):
        cv_results[kind] = {}
        for cfg, fcols in CONFIG_FEATURES.items():
            p = _oof_cv(ep, fcols, kind=kind, enforce_identity=True)
            cv_results[kind][cfg] = _eval_predictions(ep, p, n_boot=400 if kind == "hybrid" else 200)
            if kind == "hybrid":
                oof_preds_hybrid[cfg] = p

    p_unproj_ecg = _oof_cv(ep, CONFIG_FEATURES["ECG-Only"], kind="hybrid", enforce_identity=False)
    p_unproj_chest = _oof_cv(ep, CONFIG_FEATURES["Chest-ECG+SCG"], kind="hybrid", enforce_identity=False)
    p_unproj_ppg = _oof_cv(ep, CONFIG_FEATURES["PPG-Only"], kind="hybrid", enforce_identity=False)
    p_unproj_ear = _oof_cv(ep, FEATURES_EAR_PPG, kind="hybrid", enforce_identity=False)
    p_unproj_full = _oof_cv(ep, CONFIG_FEATURES["ICHD-Full"], kind="hybrid", enforce_identity=False)
    p_unproj_ridge = _oof_cv(ep, CONFIG_FEATURES["ICHD-Full"], kind="ridge", enforce_identity=False)
    cv_results["identity_ablation"] = {
        "ECG-Only_unprojected": _eval_predictions(ep, p_unproj_ecg, n_boot=200),
        "Chest-ECG+SCG_unprojected": _eval_predictions(ep, p_unproj_chest, n_boot=200),
        "PPG-Only_unprojected": _eval_predictions(ep, p_unproj_ppg, n_boot=200),
        "Ear-PPG-Only_unprojected": _eval_predictions(ep, p_unproj_ear, n_boot=200),
        "ICHD-Full_unprojected": _eval_predictions(ep, p_unproj_full, n_boot=200),
        "ICHD-Ridge_unprojected": _eval_predictions(ep, p_unproj_ridge, n_boot=200),
    }

    p_scg_only = _oof_cv(ep, FEATURES_SCG_ONLY, kind="hybrid", enforce_identity=False)
    p_ear_ppg = _oof_cv(ep, FEATURES_EAR_PPG, kind="hybrid", enforce_identity=True)
    p_chest_ear = _oof_cv(ep, FEATURES_CHEST_EAR_PPG, kind="hybrid", enforce_identity=True)
    p_ear_opt42 = _oof_cv(ep, FEATURES_EAR_OPTICAL_ONLY, kind="hybrid", enforce_identity=False)
    p_finger_opt42 = _oof_cv(ep, FEATURES_FINGER_OPTICAL_ONLY, kind="hybrid", enforce_identity=False)
    p_man_oracle = _maneuver_oracle_oof(ep, include_hr=False)
    p_man_hr_oracle = _maneuver_oracle_oof(ep, include_hr=True)
    oof_preds_hybrid["Ear-PPG-Only"] = p_ear_ppg

    # Negative Control 1: Subject-shuffled SCG features
    scg_cols = [c for c in CONFIG_FEATURES["Chest-ECG+SCG"] if c not in CONFIG_FEATURES["ECG-Only"]]
    rng_nc = np.random.default_rng(101)
    ep_shuf = ep.copy()
    perm_idx = rng_nc.permutation(len(ep_shuf))
    for c in scg_cols:
        ep_shuf[c] = ep_shuf[c].to_numpy()[perm_idx]
    p_chest_shuf = _oof_cv(ep_shuf, CONFIG_FEATURES["Chest-ECG+SCG"], kind="hybrid", enforce_identity=True)

    # Negative Control 2: Cross-subject/cross-session circular shift (len // 2)
    ep_shift = ep.copy()
    shift_n = max(1, len(ep_shift) // 2)
    for c in scg_cols:
        ep_shift[c] = np.roll(ep_shift[c].to_numpy(), shift_n)
    p_chest_shift = _oof_cv(ep_shift, CONFIG_FEATURES["Chest-ECG+SCG"], kind="hybrid", enforce_identity=True)

    low_hr_m = (ep["d_hr_bpm"].abs() <= 10.0).to_numpy()
    ep_low_hr = ep.iloc[low_hr_m].reset_index(drop=True)
    low_hr_res = {}
    for cfg in list(CONFIG_FEATURES.keys()) + ["Ear-PPG-Only"]:
        p = oof_preds_hybrid[cfg]
        sub_p = ICHDPrediction(
            p.dlog_sv[low_hr_m], p.dlog_co[low_hr_m], p.dlog_svr[low_hr_m],
            p.dlog_map[low_hr_m], p.vres_minus_vflow[low_hr_m], p.prob_res_dom[low_hr_m],
            p.hemo_angle_rad[low_hr_m], p.inconsistency[low_hr_m],
        )
        low_hr_res[cfg] = _eval_predictions(ep_low_hr, sub_p, n_boot=200)

    y_sv_all = ep["dlog_sv"].to_numpy(dtype=float)
    g_all = ep["subject"].to_numpy()
    dr_chest_vs_ecg = _paired_dr_ci(y_sv_all, oof_preds_hybrid["Chest-ECG+SCG"].dlog_sv, oof_preds_hybrid["ECG-Only"].dlog_sv, g_all)
    dr_chest_unproj_vs_ecg_unproj = _paired_dr_ci(y_sv_all, p_unproj_chest.dlog_sv, p_unproj_ecg.dlog_sv, g_all)
    dr_chest_proj_vs_unproj = _paired_dr_ci(y_sv_all, oof_preds_hybrid["Chest-ECG+SCG"].dlog_sv, p_unproj_chest.dlog_sv, g_all)
    dr_full_vs_ppg = _paired_dr_ci(y_sv_all, oof_preds_hybrid["ICHD-Full"].dlog_sv, oof_preds_hybrid["PPG-Only"].dlog_sv, g_all)
    dr_ear_vs_ppg = _paired_dr_ci(y_sv_all, p_ear_ppg.dlog_sv, oof_preds_hybrid["PPG-Only"].dlog_sv, g_all)
    dr_ear_vs_full = _paired_dr_ci(y_sv_all, p_ear_ppg.dlog_sv, oof_preds_hybrid["ICHD-Full"].dlog_sv, g_all)
    dr_chest_ear_vs_ear = _paired_dr_ci(y_sv_all, p_chest_ear.dlog_sv, p_ear_ppg.dlog_sv, g_all)
    dr_ear_opt_vs_finger_opt = _paired_dr_ci(y_sv_all, p_ear_opt42.dlog_sv, p_finger_opt42.dlog_sv, g_all)

    y_dom_all = ep["y_res_dom"].to_numpy(dtype=int)
    cv_within = {}
    for act_name, grp in ep.groupby("act_name"):
        idx = grp.index.to_numpy()
        act_d: dict = {"n_episodes": int(len(idx))}
        for cfg in ("ECG-Only", "PPG-Only", "Ear-PPG-Only", "Chest-ECG+SCG", "ICHD-Full"):
            pc = oof_preds_hybrid[cfg]
            act_d[cfg] = {
                "r_sv": float(pearsonr(ep.loc[idx, "dlog_sv"], pc.dlog_sv[idx])[0]),
                "r_co": float(pearsonr(ep.loc[idx, "dlog_co"], pc.dlog_co[idx])[0]),
                "r_svr": float(pearsonr(ep.loc[idx, "dlog_svr"], pc.dlog_svr[idx])[0]),
                "r_map": float(pearsonr(ep.loc[idx, "dlog_map"], pc.dlog_map[idx])[0]),
                "dom_acc": float(accuracy_score(y_dom_all[idx], pc.prob_res_dom[idx] >= 0.5)),
            }
        cv_within[str(act_name)] = act_d

    cp_ep = ep[ep["act_name"] == "cold_pressor"]
    val_ep = ep[ep["act_name"] == "valsalva"]
    ear_vs_finger_audit = {
        "ppge_q_median": float(ep["ppge_q"].median()),
        "ppgf_q_median": float(ep["ppgf_q"].median()),
        "ppge_q_mean": float(ep["ppge_q"].mean()),
        "ppgf_q_mean": float(ep["ppgf_q"].mean()),
        "cp_dlog_ppgf_amp_mean": float(cp_ep["dlog_ppgf_amp"].mean()),
        "cp_dlog_ppge_amp_mean": float(cp_ep["dlog_ppge_amp"].mean()),
        "val_dlog_ppgf_amp_mean": float(val_ep["dlog_ppgf_amp"].mean()),
        "val_dlog_ppge_amp_mean": float(val_ep["dlog_ppge_amp"].mean()),
        "corr_ppgf_amp_vs_ppge_amp": float(pearsonr(ep["dlog_ppgf_amp"].fillna(0), ep["dlog_ppge_amp"].fillna(0))[0]),
        "corr_ppgf_pat_vs_ppge_pat": float(pearsonr(ep["d_ppgf_pat"].fillna(0), ep["d_ppge_pat"].fillna(0))[0]),
    }

    out10 = np.abs(y_sv_all) > np.log(1.10)
    neg_out10_frac = float((y_sv_all[out10] < 0.0).mean())

    cv_results["controls"] = {
        "SCG-Only_NoHR": _eval_predictions(ep, p_scg_only, n_boot=200),
        "Ear-PPG-Only": _eval_predictions(ep, p_ear_ppg, n_boot=400),
        "Chest+Ear-PPG": _eval_predictions(ep, p_chest_ear, n_boot=200),
        "Ear-Optical-42_NoECG": _eval_predictions(ep, p_ear_opt42, n_boot=200),
        "Finger-Optical-42_NoECG": _eval_predictions(ep, p_finger_opt42, n_boot=200),
        "Maneuver-Mean_Oracle": _eval_predictions(ep, p_man_oracle, n_boot=200),
        "Maneuver+HR_Oracle": _eval_predictions(ep, p_man_hr_oracle, n_boot=200),
        "Chest-ECG+SCG_shuffled": _eval_predictions(ep, p_chest_shuf, n_boot=200),
        "Chest-ECG+SCG_timeshifted": _eval_predictions(ep, p_chest_shift, n_boot=200),
        "HR-Only_1Feat": _hr_only_oof_metrics(ep),
        "ear_vs_finger_audit": ear_vs_finger_audit,
        "cv_within_maneuver": cv_within,
        "sv_direction_skew": {
            "n_outside_10pct": int(out10.sum()),
            "neg_frac_outside_10pct": neg_out10_frac,
        },
        "low_hr_stratum_n": int(low_hr_m.sum()),
        "low_hr_stratum": low_hr_res,
        "paired_dr_sv": {
            "chest_vs_ecg": {"dr": dr_chest_vs_ecg[0], "ci": [dr_chest_vs_ecg[1], dr_chest_vs_ecg[2]], "p_val": dr_chest_vs_ecg[3]},
            "chest_unproj_vs_ecg_unproj": {"dr": dr_chest_unproj_vs_ecg_unproj[0], "ci": [dr_chest_unproj_vs_ecg_unproj[1], dr_chest_unproj_vs_ecg_unproj[2]], "p_val": dr_chest_unproj_vs_ecg_unproj[3]},
            "chest_proj_vs_unproj": {"dr": dr_chest_proj_vs_unproj[0], "ci": [dr_chest_proj_vs_unproj[1], dr_chest_proj_vs_unproj[2]], "p_val": dr_chest_proj_vs_unproj[3]},
            "full_vs_ppg": {"dr": dr_full_vs_ppg[0], "ci": [dr_full_vs_ppg[1], dr_full_vs_ppg[2]], "p_val": dr_full_vs_ppg[3]},
            "ear_vs_ppg": {"dr": dr_ear_vs_ppg[0], "ci": [dr_ear_vs_ppg[1], dr_ear_vs_ppg[2]], "p_val": dr_ear_vs_ppg[3]},
            "ear_vs_full": {"dr": dr_ear_vs_full[0], "ci": [dr_ear_vs_full[1], dr_ear_vs_full[2]], "p_val": dr_ear_vs_full[3]},
            "chest_ear_vs_ear": {"dr": dr_chest_ear_vs_ear[0], "ci": [dr_chest_ear_vs_ear[1], dr_chest_ear_vs_ear[2]], "p_val": dr_chest_ear_vs_ear[3]},
            "ear_opt_vs_finger_opt": {"dr": dr_ear_opt_vs_finger_opt[0], "ci": [dr_ear_opt_vs_finger_opt[1], dr_ear_opt_vs_finger_opt[2]], "p_val": dr_ear_opt_vs_finger_opt[3]},
        },
        "incremental_r_sv_beyond_oracle": {
            cfg: _incremental_r_sv_beyond_oracle(ep, oof_preds_hybrid[cfg].dlog_sv)
            for cfg in ("ECG-Only", "PPG-Only", "Ear-PPG-Only", "Chest-ECG+SCG", "ICHD-Full")
        },
    }

    act_w = w[w["act"].isin([1, 2, 3, 4]) & (w["scg_q"] >= 0.25)].copy().reset_index(drop=True)
    cv_win = {}
    for cfg in CONFIG_FEATURES:
        pw = _oof_cv(act_w, CONFIG_FEATURES[cfg], kind="ridge", enforce_identity=True)
        cv_win[cfg] = _eval_predictions(act_w, pw, n_boot=50)
    cv_results["n_active_windows"] = int(len(act_w))
    cv_results["active_windows"] = cv_win
    cv_results["active_windows_hr_only"] = _hr_only_oof_metrics(act_w)
    return cv_results, oof_preds_hybrid


def _run_exp3_lomo(ep: pd.DataFrame, oof_preds_hybrid: dict[str, ICHDPrediction]) -> dict:
    """Subject-Disjoint Leave-One-Maneuver-Out (SD-LOMO) generalization & ablations."""
    gkf_lomo = GroupKFold(n_splits=5)
    y_sv_all = ep["dlog_sv"].to_numpy(dtype=float)
    g_all = ep["subject"].to_numpy()
    act_arr = ep["act"].to_numpy()

    def _eval_lomo_config(fcols: list[str], enforce_identity: bool = True, n_boot: int = 300) -> tuple[dict, ICHDPrediction]:
        n = len(ep)
        sv, co, svr, mp, vd, pr, ang, inc = (np.zeros(n) for _ in range(8))
        lomo_tasks = []
        for tr_sub_idx, te_sub_idx in gkf_lomo.split(ep, groups=ep["subject"]):
            tr_sub_mask = np.zeros(n, dtype=bool)
            te_sub_mask = np.zeros(n, dtype=bool)
            tr_sub_mask[tr_sub_idx] = True
            te_sub_mask[te_sub_idx] = True
            for act_code in (1, 2, 3, 4):
                tr_m = tr_sub_mask & (act_arr != act_code)
                te_m = te_sub_mask & (act_arr == act_code)
                if te_m.sum() == 0:
                    continue
                te_idx = np.flatnonzero(te_m)
                lomo_tasks.append((ep.iloc[tr_m], ep.iloc[te_m], te_idx, fcols, "hybrid", enforce_identity))
        with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 4), mp_context=_mp.get_context("spawn")) as ex:
            for te_idx, p in ex.map(_fit_predict_fold, lomo_tasks):
                sv[te_idx], co[te_idx], svr[te_idx], mp[te_idx] = p.dlog_sv, p.dlog_co, p.dlog_svr, p.dlog_map
                vd[te_idx], pr[te_idx], ang[te_idx], inc[te_idx] = p.vres_minus_vflow, p.prob_res_dom, p.hemo_angle_rad, p.inconsistency

        per_act = {}
        within_r_co, within_r_sv, within_acc = [], [], []
        for act_code in (1, 2, 3, 4):
            m_act = (act_arr == act_code)
            act_name = str(ep.loc[m_act, "act_name"].iloc[0])
            r_co_a = float(pearsonr(ep.loc[m_act, "dlog_co"], co[m_act])[0])
            r_sv_a = float(pearsonr(ep.loc[m_act, "dlog_sv"], sv[m_act])[0])
            acc_a = float(accuracy_score(ep.loc[m_act, "y_res_dom"], pr[m_act] >= 0.5))
            per_act[act_name] = {"dom_acc": acc_a, "r_co": r_co_a, "r_sv": r_sv_a}
            within_r_co.append(r_co_a)
            within_r_sv.append(r_sv_a)
            within_acc.append(acc_a)

        ex_cp_m = ep["act"].isin([3, 4]).to_numpy()
        ex_cp_auc = float(roc_auc_score(ep.loc[ex_cp_m, "y_res_dom"], pr[ex_cp_m]))
        ex_cp_acc = float(accuracy_score(ep.loc[ex_cp_m, "y_res_dom"], pr[ex_cp_m] >= 0.5))

        p_all = ICHDPrediction(sv, co, svr, mp, vd, pr, ang, inc)
        out = _eval_predictions(ep, p_all, n_boot=n_boot)
        out["mean_within_r_co"] = float(np.mean(within_r_co))
        out["mean_within_r_sv"] = float(np.mean(within_r_sv))
        out["mean_within_dom_acc"] = float(np.mean(within_acc))
        out["ex_vs_cp_auc"] = ex_cp_auc
        out["ex_vs_cp_acc"] = ex_cp_acc
        out["per_maneuver"] = per_act
        return out, p_all

    def _lomo_hr_only_metrics() -> dict:
        n = len(ep)
        prob_hr = np.zeros(n)
        sv_hr = np.zeros(n)
        x_hr = ep[["dlog_hr"]].fillna(0.0).to_numpy(dtype=float)
        y_hr = ep["y_res_dom"].to_numpy(dtype=int)
        for tr_sub_idx, te_sub_idx in gkf_lomo.split(ep, groups=ep["subject"]):
            tr_sub_mask = np.zeros(n, dtype=bool)
            te_sub_mask = np.zeros(n, dtype=bool)
            tr_sub_mask[tr_sub_idx] = True
            te_sub_mask[te_sub_idx] = True
            for act_code in (1, 2, 3, 4):
                tr_m = tr_sub_mask & (act_arr != act_code)
                te_m = te_sub_mask & (act_arr == act_code)
                if te_m.sum() == 0:
                    continue
                prob_hr[te_m] = LogisticRegression(max_iter=1000).fit(x_hr[tr_m], y_hr[tr_m]).predict_proba(x_hr[te_m])[:, 1]
                sv_hr[te_m] = Ridge(alpha=1.0).fit(x_hr[tr_m], y_sv_all[tr_m]).predict(x_hr[te_m])
        return {
            "dom_auc": float(roc_auc_score(y_hr, prob_hr)),
            "dom_acc": float(accuracy_score(y_hr, prob_hr >= 0.5)),
            "r_sv": float(pearsonr(y_sv_all, sv_hr)[0]),
        }

    lomo_results = {}
    lomo_preds = {}
    for cfg, fcols in CONFIG_FEATURES.items():
        lomo_results[cfg], lomo_preds[cfg] = _eval_lomo_config(fcols, enforce_identity=True, n_boot=300)

    lomo_ear_res, lomo_ear_pred = _eval_lomo_config(FEATURES_EAR_PPG, enforce_identity=True, n_boot=400)
    lomo_chest_ear_res, lomo_chest_ear_pred = _eval_lomo_config(FEATURES_CHEST_EAR_PPG, enforce_identity=True, n_boot=200)
    lomo_ear_opt_res, lomo_ear_opt_pred = _eval_lomo_config(FEATURES_EAR_OPTICAL_ONLY, enforce_identity=False, n_boot=200)
    lomo_finger_opt_res, lomo_finger_opt_pred = _eval_lomo_config(FEATURES_FINGER_OPTICAL_ONLY, enforce_identity=False, n_boot=200)

    dr_lomo_chest_ecg = _paired_dr_ci(y_sv_all, lomo_preds["Chest-ECG+SCG"].dlog_sv, lomo_preds["ECG-Only"].dlog_sv, g_all)
    dr_lomo_full_ppg = _paired_dr_ci(y_sv_all, lomo_preds["ICHD-Full"].dlog_sv, lomo_preds["PPG-Only"].dlog_sv, g_all)
    dr_lomo_ear_ppg = _paired_dr_ci(y_sv_all, lomo_ear_pred.dlog_sv, lomo_preds["PPG-Only"].dlog_sv, g_all)
    dr_lomo_ear_full = _paired_dr_ci(y_sv_all, lomo_ear_pred.dlog_sv, lomo_preds["ICHD-Full"].dlog_sv, g_all)
    dr_lomo_chest_ear_vs_ear = _paired_dr_ci(y_sv_all, lomo_chest_ear_pred.dlog_sv, lomo_ear_pred.dlog_sv, g_all)
    dr_lomo_ear_opt_vs_finger_opt = _paired_dr_ci(y_sv_all, lomo_ear_opt_pred.dlog_sv, lomo_finger_opt_pred.dlog_sv, g_all)

    wm_dr_ear_vs_ppg = _paired_within_maneuver_dr_ci(y_sv_all, lomo_ear_pred.dlog_sv, lomo_preds["PPG-Only"].dlog_sv, act_arr, g_all)
    wm_dr_ear_vs_full = _paired_within_maneuver_dr_ci(y_sv_all, lomo_ear_pred.dlog_sv, lomo_preds["ICHD-Full"].dlog_sv, act_arr, g_all)
    wm_dr_chest_vs_ecg = _paired_within_maneuver_dr_ci(y_sv_all, lomo_preds["Chest-ECG+SCG"].dlog_sv, lomo_preds["ECG-Only"].dlog_sv, act_arr, g_all)

    inc_r_lomo = {
        cfg: _incremental_r_sv_beyond_oracle(ep, lomo_preds[cfg].dlog_sv)
        for cfg in ("ECG-Only", "PPG-Only", "Chest-ECG+SCG", "ICHD-Full")
    }
    inc_r_lomo["Ear-PPG-Only"] = _incremental_r_sv_beyond_oracle(ep, lomo_ear_pred.dlog_sv)

    lomo_ear_no_amp_res, lomo_ear_no_amp_pred = _eval_lomo_config(FEATURES_EAR_NO_AMP, enforce_identity=True, n_boot=400)
    lomo_finger_no_amp_res, lomo_finger_no_amp_pred = _eval_lomo_config(FEATURES_FINGER_NO_AMP, enforce_identity=True, n_boot=400)
    dr_no_amp_ear_vs_finger = _paired_dr_ci(y_sv_all, lomo_ear_no_amp_pred.dlog_sv, lomo_finger_no_amp_pred.dlog_sv, g_all)
    wm_dr_no_amp = _paired_within_maneuver_dr_ci(y_sv_all, lomo_ear_no_amp_pred.dlog_sv, lomo_finger_no_amp_pred.dlog_sv, act_arr, g_all)
    did_amp_ear_vs_finger = _paired_did_dr_ci(
        y_sv_all,
        lomo_ear_pred.dlog_sv,
        lomo_preds["PPG-Only"].dlog_sv,
        lomo_ear_no_amp_pred.dlog_sv,
        lomo_finger_no_amp_pred.dlog_sv,
        g_all,
    )

    primary_hyp = {
        "hypothesis": "Ear-PPG-Only vs PPG-Only (LOMO r_sv)",
        "dr": dr_lomo_ear_ppg[0],
        "ci": [dr_lomo_ear_ppg[1], dr_lomo_ear_ppg[2]],
        "p_val": dr_lomo_ear_ppg[3],
    }
    secondary_hyps = [
        {"hypothesis": "Ear-PPG-Only vs ICHD-Full (LOMO r_sv)", "dr": dr_lomo_ear_full[0], "ci": [dr_lomo_ear_full[1], dr_lomo_ear_full[2]], "p_val": dr_lomo_ear_full[3]},
        {"hypothesis": "Chest-ECG+SCG vs ECG-Only (LOMO r_sv)", "dr": dr_lomo_chest_ecg[0], "ci": [dr_lomo_chest_ecg[1], dr_lomo_chest_ecg[2]], "p_val": dr_lomo_chest_ecg[3]},
        {"hypothesis": "Ear-Optical-42 vs Finger-Optical-42 (LOMO r_sv)", "dr": dr_lomo_ear_opt_vs_finger_opt[0], "ci": [dr_lomo_ear_opt_vs_finger_opt[1], dr_lomo_ear_opt_vs_finger_opt[2]], "p_val": dr_lomo_ear_opt_vs_finger_opt[3]},
        {"hypothesis": "Within-Maneuver: Ear-PPG-Only vs PPG-Only", "dr": wm_dr_ear_vs_ppg["mean_dr"], "ci": wm_dr_ear_vs_ppg["ci"], "p_val": wm_dr_ear_vs_ppg["p_val"]},
        {"hypothesis": "Within-Maneuver: Ear-PPG-Only vs ICHD-Full", "dr": wm_dr_ear_vs_full["mean_dr"], "ci": wm_dr_ear_vs_full["ci"], "p_val": wm_dr_ear_vs_full["p_val"]},
        {"hypothesis": "Within-Maneuver: Chest-ECG+SCG vs ECG-Only", "dr": wm_dr_chest_vs_ecg["mean_dr"], "ci": wm_dr_chest_vs_ecg["ci"], "p_val": wm_dr_chest_vs_ecg["p_val"]},
        {"hypothesis": "Amplitude-Ablated: Ear-noAmp vs Finger-noAmp (LOMO r_sv)", "dr": dr_no_amp_ear_vs_finger[0], "ci": [dr_no_amp_ear_vs_finger[1], dr_no_amp_ear_vs_finger[2]], "p_val": dr_no_amp_ear_vs_finger[3]},
        {"hypothesis": "Amplitude-Ablation Interaction: DiD (w/ vs w/o Amp)", "dr": did_amp_ear_vs_finger[0], "ci": [did_amp_ear_vs_finger[1], did_amp_ear_vs_finger[2]], "p_val": did_amp_ear_vs_finger[3]},
    ]
    secondary_adjusted = _holm_bonferroni_correction(secondary_hyps)

    pressor_m = (ep["d_map_mmhg"] > 1.0).to_numpy()
    pressor_ep = ep.iloc[pressor_m].reset_index(drop=True)
    pressor_res = {}
    for cfg, p in oof_preds_hybrid.items():
        sub_p = ICHDPrediction(
            p.dlog_sv[pressor_m], p.dlog_co[pressor_m], p.dlog_svr[pressor_m],
            p.dlog_map[pressor_m], p.vres_minus_vflow[pressor_m], p.prob_res_dom[pressor_m],
            p.hemo_angle_rad[pressor_m], p.inconsistency[pressor_m],
        )
        pressor_res[cfg] = _eval_predictions(pressor_ep, sub_p, n_boot=200)

    return {
        "lomo": lomo_results,
        "controls": {
            "Ear-PPG-Only": lomo_ear_res,
            "Chest+Ear-PPG": lomo_chest_ear_res,
            "Ear-Optical-42_NoECG": lomo_ear_opt_res,
            "Finger-Optical-42_NoECG": lomo_finger_opt_res,
            "HR-Only_1Feat": _lomo_hr_only_metrics(),
        },
        "pressor_subset": pressor_res,
        "n_pressor": int(pressor_m.sum()),
        "paired_dr_sv_chest_vs_ecg": {"dr": dr_lomo_chest_ecg[0], "ci": [dr_lomo_chest_ecg[1], dr_lomo_chest_ecg[2]], "p_val": dr_lomo_chest_ecg[3]},
        "paired_dr_sv": {
            "chest_vs_ecg": {"dr": dr_lomo_chest_ecg[0], "ci": [dr_lomo_chest_ecg[1], dr_lomo_chest_ecg[2]], "p_val": dr_lomo_chest_ecg[3]},
            "full_vs_ppg": {"dr": dr_lomo_full_ppg[0], "ci": [dr_lomo_full_ppg[1], dr_lomo_full_ppg[2]], "p_val": dr_lomo_full_ppg[3]},
            "ear_vs_ppg": {"dr": dr_lomo_ear_ppg[0], "ci": [dr_lomo_ear_ppg[1], dr_lomo_ear_ppg[2]], "p_val": dr_lomo_ear_ppg[3]},
            "ear_vs_full": {"dr": dr_lomo_ear_full[0], "ci": [dr_lomo_ear_full[1], dr_lomo_ear_full[2]], "p_val": dr_lomo_ear_full[3]},
            "chest_ear_vs_ear": {"dr": dr_lomo_chest_ear_vs_ear[0], "ci": [dr_lomo_chest_ear_vs_ear[1], dr_lomo_chest_ear_vs_ear[2]], "p_val": dr_lomo_chest_ear_vs_ear[3]},
            "ear_opt_vs_finger_opt": {"dr": dr_lomo_ear_opt_vs_finger_opt[0], "ci": [dr_lomo_ear_opt_vs_finger_opt[1], dr_lomo_ear_opt_vs_finger_opt[2]], "p_val": dr_lomo_ear_opt_vs_finger_opt[3]},
        },
        "within_maneuver_paired_dr": {
            "ear_vs_ppg": wm_dr_ear_vs_ppg,
            "ear_vs_full": wm_dr_ear_vs_full,
            "chest_vs_ecg": wm_dr_chest_vs_ecg,
            "ear_vs_finger_no_amp": wm_dr_no_amp,
        },
        "incremental_r_sv_beyond_oracle": inc_r_lomo,
        "amplitude_ablation": {
            "Ear-PPG-noAmp": lomo_ear_no_amp_res,
            "Finger-PPG-noAmp": lomo_finger_no_amp_res,
            "paired_dr_sv_ear_vs_finger_no_amp": {
                "dr": dr_no_amp_ear_vs_finger[0],
                "ci": [dr_no_amp_ear_vs_finger[1], dr_no_amp_ear_vs_finger[2]],
                "p_val": dr_no_amp_ear_vs_finger[3],
            },
            "paired_did_sv_amp_vs_no_amp": {
                "did": did_amp_ear_vs_finger[0],
                "ci": [did_amp_ear_vs_finger[1], did_amp_ear_vs_finger[2]],
                "p_val": did_amp_ear_vs_finger[3],
            },
        },
        "multiplicity": {
            "primary": primary_hyp,
            "secondary": secondary_adjusted,
        },
    }


def _run_exp4_transfer(ep: pd.DataFrame) -> dict:
    """Zero-shot clinical transfer from Healthy Controls (HC) to Heart Failure (HFC)."""
    tr_hc = ep[ep["cohort"] == "HC"].reset_index(drop=True)
    te_hfc = ep[ep["cohort"] == "HFC"].reset_index(drop=True)
    transfer_res: dict = {}
    transfer_preds_hybrid: dict[str, ICHDPrediction] = {}
    for kind in ("hybrid", "ridge", "gbm"):
        transfer_res[kind] = {}
        for cfg, fcols in CONFIG_FEATURES.items():
            m = ICHDModel(fcols, kind=kind, enforce_identity=True).fit(tr_hc)
            p = m.predict(te_hfc)
            transfer_res[kind][cfg] = _eval_predictions(te_hfc, p, n_boot=300)
            if kind == "hybrid":
                transfer_preds_hybrid[cfg] = p

    p_tr_ear = ICHDModel(FEATURES_EAR_PPG, kind="hybrid", enforce_identity=True).fit(tr_hc).predict(te_hfc)
    p_tr_chest_ear = ICHDModel(FEATURES_CHEST_EAR_PPG, kind="hybrid", enforce_identity=True).fit(tr_hc).predict(te_hfc)
    p_tr_ear_opt = ICHDModel(FEATURES_EAR_OPTICAL_ONLY, kind="hybrid", enforce_identity=False).fit(tr_hc).predict(te_hfc)
    p_tr_finger_opt = ICHDModel(FEATURES_FINGER_OPTICAL_ONLY, kind="hybrid", enforce_identity=False).fit(tr_hc).predict(te_hfc)

    x_tr_hr = tr_hc[["dlog_hr"]].fillna(0.0).to_numpy(dtype=float)
    x_te_hr = te_hfc[["dlog_hr"]].fillna(0.0).to_numpy(dtype=float)
    prob_tr_hr = LogisticRegression(max_iter=1000).fit(x_tr_hr, tr_hc["y_res_dom"].to_numpy(dtype=int)).predict_proba(x_te_hr)[:, 1]
    sv_tr_hr = Ridge(alpha=1.0).fit(x_tr_hr, tr_hc["dlog_sv"].to_numpy(dtype=float)).predict(x_te_hr)
    tr_hr_only_metrics = {
        "dom_auc": float(roc_auc_score(te_hfc["y_res_dom"].to_numpy(dtype=int), prob_tr_hr)),
        "dom_acc": float(accuracy_score(te_hfc["y_res_dom"].to_numpy(dtype=int), prob_tr_hr >= 0.5)),
        "r_sv": float(pearsonr(te_hfc["dlog_sv"].to_numpy(dtype=float), sv_tr_hr)[0]),
    }

    y_sv_hfc = te_hfc["dlog_sv"].to_numpy(dtype=float)
    g_hfc = te_hfc["subject"].to_numpy()
    dr_tr_full_ppg = _paired_dr_ci(y_sv_hfc, transfer_preds_hybrid["ICHD-Full"].dlog_sv, transfer_preds_hybrid["PPG-Only"].dlog_sv, g_hfc)
    dr_tr_chest_ecg = _paired_dr_ci(y_sv_hfc, transfer_preds_hybrid["Chest-ECG+SCG"].dlog_sv, transfer_preds_hybrid["ECG-Only"].dlog_sv, g_hfc)
    dr_tr_ear_ppg = _paired_dr_ci(y_sv_hfc, p_tr_ear.dlog_sv, transfer_preds_hybrid["PPG-Only"].dlog_sv, g_hfc)
    dr_tr_ear_full = _paired_dr_ci(y_sv_hfc, p_tr_ear.dlog_sv, transfer_preds_hybrid["ICHD-Full"].dlog_sv, g_hfc)
    dr_tr_chest_ear_vs_ear = _paired_dr_ci(y_sv_hfc, p_tr_chest_ear.dlog_sv, p_tr_ear.dlog_sv, g_hfc)
    dr_tr_ear_opt_vs_finger_opt = _paired_dr_ci(y_sv_hfc, p_tr_ear_opt.dlog_sv, p_tr_finger_opt.dlog_sv, g_hfc)

    transfer_res["controls"] = {
        "Ear-PPG-Only": _eval_predictions(te_hfc, p_tr_ear, n_boot=400),
        "Chest+Ear-PPG": _eval_predictions(te_hfc, p_tr_chest_ear, n_boot=200),
        "Ear-Optical-42_NoECG": _eval_predictions(te_hfc, p_tr_ear_opt, n_boot=200),
        "Finger-Optical-42_NoECG": _eval_predictions(te_hfc, p_tr_finger_opt, n_boot=200),
        "HR-Only_1Feat": tr_hr_only_metrics,
        "paired_dr_sv_full_vs_ppg": {"dr": dr_tr_full_ppg[0], "ci": [dr_tr_full_ppg[1], dr_tr_full_ppg[2]], "p_val": dr_tr_full_ppg[3]},
        "paired_dr_sv_chest_vs_ecg": {"dr": dr_tr_chest_ecg[0], "ci": [dr_tr_chest_ecg[1], dr_tr_chest_ecg[2]], "p_val": dr_tr_chest_ecg[3]},
        "paired_dr_sv": {
            "chest_vs_ecg": {"dr": dr_tr_chest_ecg[0], "ci": [dr_tr_chest_ecg[1], dr_tr_chest_ecg[2]], "p_val": dr_tr_chest_ecg[3]},
            "full_vs_ppg": {"dr": dr_tr_full_ppg[0], "ci": [dr_tr_full_ppg[1], dr_tr_full_ppg[2]], "p_val": dr_tr_full_ppg[3]},
            "ear_vs_ppg": {"dr": dr_tr_ear_ppg[0], "ci": [dr_tr_ear_ppg[1], dr_tr_ear_ppg[2]], "p_val": dr_tr_ear_ppg[3]},
            "ear_vs_full": {"dr": dr_tr_ear_full[0], "ci": [dr_tr_ear_full[1], dr_tr_ear_full[2]], "p_val": dr_tr_ear_full[3]},
            "chest_ear_vs_ear": {"dr": dr_tr_chest_ear_vs_ear[0], "ci": [dr_tr_chest_ear_vs_ear[1], dr_tr_chest_ear_vs_ear[2]], "p_val": dr_tr_chest_ear_vs_ear[3]},
            "ear_opt_vs_finger_opt": {"dr": dr_tr_ear_opt_vs_finger_opt[0], "ci": [dr_tr_ear_opt_vs_finger_opt[1], dr_tr_ear_opt_vs_finger_opt[2]], "p_val": dr_tr_ear_opt_vs_finger_opt[3]},
        },
    }
    return transfer_res


def _subset_pred(p: ICHDPrediction, idx: np.ndarray) -> ICHDPrediction:
    return ICHDPrediction(
        p.dlog_sv[idx], p.dlog_co[idx], p.dlog_svr[idx], p.dlog_map[idx],
        p.vres_minus_vflow[idx], p.prob_res_dom[idx], p.hemo_angle_rad[idx], p.inconsistency[idx],
    )


def _run_exp5_conformal(ep: pd.DataFrame, oof_preds_hybrid: dict[str, ICHDPrediction]) -> dict:
    """Subject-weighted conformal risk control, 2D box certification, and Mondrian calibration."""
    conf_res: dict = {}
    conf_configs = dict(CONFIG_FEATURES)
    conf_configs["Ear-PPG-Only"] = FEATURES_EAR_PPG

    for cfg, fcols in conf_configs.items():
        p = oof_preds_hybrid[cfg]
        q_col = quality_col_for_config(cfg, fcols)
        c_score = confidence_score(p, ep, quality_col=q_col)
        y_dom = ep["y_res_dom"].to_numpy(dtype=int)
        y_hat = (p.prob_res_dom >= 0.5).astype(int)
        rcc = risk_coverage_curve(y_dom, y_hat, c_score, n_grid=91)

        gkf_cal = GroupKFold(n_splits=10)
        hw_co_all, hw_svr_all, hw_diff_all = np.zeros(len(ep)), np.zeros(len(ep)), np.zeros(len(ep))
        hw_co_mond, hw_svr_mond, hw_diff_mond = np.zeros(len(ep)), np.zeros(len(ep)), np.zeros(len(ep))

        for tr, te in gkf_cal.split(ep, groups=ep["subject"]):
            reg2d = calibrate_2d_conformal(_subset_pred(p, tr), ep.iloc[tr], alpha=0.10, alpha_margin=0.10, quality_col=q_col)
            p_te = _subset_pred(p, te)
            hw_co_all[te], hw_svr_all[te] = reg2d.half_widths(p_te, ep.iloc[te])
            hw_diff_all[te] = reg2d.margin_half_width(p_te, ep.iloc[te])

            tr_hc_idx = tr[ep.iloc[tr]["cohort"].to_numpy() == "HC"]
            tr_hfc_idx = tr[ep.iloc[tr]["cohort"].to_numpy() == "HFC"]
            reg2d_hc = calibrate_2d_conformal(_subset_pred(p, tr_hc_idx), ep.iloc[tr_hc_idx], alpha=0.10, alpha_margin=0.10, quality_col=q_col)
            reg2d_hfc = calibrate_2d_conformal(_subset_pred(p, tr_hfc_idx), ep.iloc[tr_hfc_idx], alpha=0.10, alpha_margin=0.10, quality_col=q_col)

            for cohort_lbl, reg_c in (("HC", reg2d_hc), ("HFC", reg2d_hfc)):
                te_c_idx = te[ep.iloc[te]["cohort"].to_numpy() == cohort_lbl]
                if len(te_c_idx):
                    p_te_c = _subset_pred(p, te_c_idx)
                    hw_co_mond[te_c_idx], hw_svr_mond[te_c_idx] = reg_c.half_widths(p_te_c, ep.iloc[te_c_idx])
                    hw_diff_mond[te_c_idx] = reg_c.margin_half_width(p_te_c, ep.iloc[te_c_idx])

        y_co = ep["dlog_co"].to_numpy(dtype=float)
        y_svr = ep["dlog_svr"].to_numpy(dtype=float)
        y_diff = y_svr - y_co
        cov_co = np.abs(y_co - p.dlog_co) <= hw_co_all
        cov_svr = np.abs(y_svr - p.dlog_svr) <= hw_svr_all
        cov_diff = np.abs(y_diff - p.vres_minus_vflow) <= hw_diff_all
        joint_cov = cov_co & cov_svr

        cov_co_m = np.abs(y_co - p.dlog_co) <= hw_co_mond
        cov_svr_m = np.abs(y_svr - p.dlog_svr) <= hw_svr_mond
        cov_diff_m = np.abs(y_diff - p.vres_minus_vflow) <= hw_diff_mond
        joint_cov_m = cov_co_m & cov_svr_m

        subj_covs = [float(np.mean(joint_cov[ep["subject"].to_numpy() == s])) for s in ep["subject"].unique()]
        subj_diff_covs = [float(np.mean(cov_diff[ep["subject"].to_numpy() == s])) for s in ep["subject"].unique()]

        hc_m = (ep["cohort"].to_numpy() == "HC")
        hfc_m = (ep["cohort"].to_numpy() == "HFC")
        per_man_cov = {
            str(act_n): float(np.mean(joint_cov[ep["act_name"].to_numpy() == act_n]))
            for act_n in ("breath_hold", "valsalva", "exercise", "cold_pressor")
        }

        mondrian_info = {
            "joint_cov": float(np.mean(joint_cov_m)),
            "hc_joint_cov": float(np.mean(joint_cov_m[hc_m])),
            "hfc_joint_cov": float(np.mean(joint_cov_m[hfc_m])),
            "co_cov": float(np.mean(cov_co_m)),
            "svr_cov": float(np.mean(cov_svr_m)),
            "diff_cov": float(np.mean(cov_diff_m)),
            "median_hw_co_hc": float(np.median(hw_co_mond[hc_m])),
            "median_hw_co_hfc": float(np.median(hw_co_mond[hfc_m])),
            "median_hw_svr_hc": float(np.median(hw_svr_mond[hc_m])),
            "median_hw_svr_hfc": float(np.median(hw_svr_mond[hfc_m])),
            "median_hw_diff_hc": float(np.median(hw_diff_mond[hc_m])),
            "median_hw_diff_hfc": float(np.median(hw_diff_mond[hfc_m])),
        }

        cert_box_mask = np.abs(p.vres_minus_vflow) > (hw_co_all + hw_svr_all)
        cert_margin_mask = np.abs(p.vres_minus_vflow) > hw_diff_all
        y_hat_vec = (p.vres_minus_vflow > 0.0).astype(int)

        cov_arr = rcc["coverages"]
        acc_arr = rcc["accuracies"]
        conf_res[cfg] = {
            "aurc": float(rcc["aurc"]),
            "acc_cov_100": float(acc_arr[-1]),
            "acc_cov_80": float(acc_arr[np.argmin(np.abs(cov_arr - 0.80))]),
            "acc_cov_70": float(acc_arr[np.argmin(np.abs(cov_arr - 0.70))]),
            "acc_cov_60": float(acc_arr[np.argmin(np.abs(cov_arr - 0.60))]),
            "acc_cov_50": float(acc_arr[np.argmin(np.abs(cov_arr - 0.50))]),
            "coverages": [float(x) for x in cov_arr[::5]],
            "accuracies": [float(x) for x in acc_arr[::5]],
            "risks": [float(x) for x in rcc["risks"][::5]],
            "conformal_2d_joint_cov": float(np.mean(joint_cov)),
            "conformal_2d_subject_macro_cov": float(np.mean(subj_covs)),
            "conformal_2d_hc_joint_cov": float(np.mean(joint_cov[hc_m])),
            "conformal_2d_hfc_joint_cov": float(np.mean(joint_cov[hfc_m])),
            "conformal_2d_per_maneuver_cov": per_man_cov,
            "conformal_2d_co_cov": float(np.mean(cov_co)),
            "conformal_2d_svr_cov": float(np.mean(cov_svr)),
            "conformal_1d_diff_cov": float(np.mean(cov_diff)),
            "conformal_1d_diff_subject_macro_cov": float(np.mean(subj_diff_covs)),
            "median_hw_co": float(np.median(hw_co_all)),
            "median_hw_svr": float(np.median(hw_svr_all)),
            "median_hw_diff": float(np.median(hw_diff_all)),
            "certified_frac": float(np.mean(cert_box_mask)),
            "certified_acc": float(np.mean(y_dom[cert_box_mask] == y_hat_vec[cert_box_mask])) if cert_box_mask.sum() > 0 else np.nan,
            "certified_margin_frac": float(np.mean(cert_margin_mask)),
            "certified_margin_acc": float(np.mean(y_dom[cert_margin_mask] == y_hat_vec[cert_margin_mask])) if cert_margin_mask.sum() > 0 else np.nan,
            "mondrian": mondrian_info,
        }
    return conf_res


def run_all() -> dict:
    if (TABLES_CACHE / "episodes.pkl").exists():
        w = pd.read_pickle(TABLES_CACHE / "windows.pkl")
        ep = pd.read_pickle(TABLES_CACHE / "episodes.pkl")
    else:
        w, ep = build_deltas_and_episodes()
        TABLES_CACHE.mkdir(parents=True, exist_ok=True)
        w.to_pickle(TABLES_CACHE / "windows.pkl")
        ep.to_pickle(TABLES_CACHE / "episodes.pkl")

    exp2_cv, oof_preds_hybrid = _run_exp2_cv(w, ep)
    results = {
        "exp1_biophysics": _run_exp1_biophysics(w, ep),
        "exp2_cv": exp2_cv,
        "exp3_lomo": _run_exp3_lomo(ep, oof_preds_hybrid),
        "exp4_transfer": _run_exp4_transfer(ep),
        "exp5_conformal": _run_exp5_conformal(ep, oof_preds_hybrid),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "ichd_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Wrote results/ichd_results.json")
    return results


if __name__ == "__main__":
    run_all()

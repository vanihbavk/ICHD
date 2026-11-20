"""Experiment 11: SAGER Sex/Gender Equity Audit & Cohort Demographics Summary.

Computes:
1. Full cohort demographics (Sex, Age, Height, Weight, BMI, LVEF, NYHA, medications)
   for enrolled (N=70) and active (N=67: 42 HC, 25 HFC) Auricle participants.
2. Sex-stratified performance (Female vs. Male) in 5-fold subject-disjoint CV and
   4x5 subject-disjoint LOMO for ICHD-Full, Ear-PPG-Only, PPG-Only (Finger), and HR-Only,
   with subject-clustered bootstrap 95% CIs and two-sided difference p-values.

Saves results/exp11_sager_sex_audit.json.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from ichd.evaluate import _oof_cv, ROOT, TABLES_CACHE, RESULTS_DIR
from ichd.model import (
    CONFIG_FEATURES,
    FEATURES_EAR_PPG,
    ICHDModel,
)
from ichd.stats import _r_ci


def _parse_lvef(val) -> float:
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
    if not nums:
        return np.nan
    return float(np.mean(nums))


def _unpaired_cluster_diff_r(
    y1: np.ndarray,
    p1: np.ndarray,
    c1: np.ndarray,
    y2: np.ndarray,
    p2: np.ndarray,
    c2: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Subject-clustered bootstrap for unpaired difference r(group1) - r(group2)."""
    rng = np.random.default_rng(seed)

    def _r(y, p):
        m = np.isfinite(y) & np.isfinite(p)
        if m.sum() < 4 or np.std(y[m]) < 1e-12 or np.std(p[m]) < 1e-12:
            return np.nan
        return float(np.corrcoef(y[m], p[m])[0, 1])

    r1_obs = _r(y1, p1)
    r2_obs = _r(y2, p2)
    diff_obs = float(r1_obs - r2_obs)

    u1 = np.unique(c1)
    u2 = np.unique(c2)
    g1 = {u: np.where(c1 == u)[0] for u in u1}
    g2 = {u: np.where(c2 == u)[0] for u in u2}

    diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        s1 = rng.choice(u1, size=len(u1), replace=True)
        idx1 = np.concatenate([g1[u] for u in s1])
        s2 = rng.choice(u2, size=len(u2), replace=True)
        idx2 = np.concatenate([g2[u] for u in s2])
        diffs[b] = _r(y1[idx1], p1[idx1]) - _r(y2[idx2], p2[idx2])

    valid = diffs[np.isfinite(diffs)]
    ci_lo, ci_hi = np.percentile(valid, [2.5, 97.5])
    # Two-sided bootstrap p-value testing H0: diff = 0
    p_pos = (np.sum(valid >= 0.0) + 1) / (len(valid) + 1)
    p_neg = (np.sum(valid <= 0.0) + 1) / (len(valid) + 1)
    p_val = float(min(1.0, 2.0 * min(p_pos, p_neg)))
    return {
        "r_female": r1_obs,
        "r_male": r2_obs,
        "delta_r_f_minus_m": diff_obs,
        "ci_95": [float(ci_lo), float(ci_hi)],
        "p_boot": p_val,
    }


def _lomo_oof_sv(ep: pd.DataFrame, fcols: list[str], enforce_identity: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Run exact 4x5 subject-disjoint LOMO returning out-of-fold dlog_sv and prob_res_dom."""
    n = len(ep)
    pred_sv = np.full(n, np.nan)
    pred_prob = np.full(n, np.nan)
    subjects = ep["subject"].to_numpy()
    acts = sorted(ep["act_name"].unique())
    gkf = GroupKFold(n_splits=5)

    for held_act in acts:
        for tr_sub_idx, te_sub_idx in gkf.split(ep, groups=subjects):
            tr_mask = np.zeros(n, dtype=bool)
            tr_mask[tr_sub_idx] = True
            tr_mask &= (ep["act_name"].to_numpy() != held_act)
            te_mask = np.zeros(n, dtype=bool)
            te_mask[te_sub_idx] = True
            te_mask &= (ep["act_name"].to_numpy() == held_act)
            if te_mask.sum() == 0:
                continue
            mdl = ICHDModel(fcols, kind="hybrid", enforce_identity=enforce_identity, seed=42)
            mdl.fit(ep.iloc[tr_mask])
            pr = mdl.predict(ep.iloc[te_mask])
            pred_sv[te_mask] = pr.dlog_sv
            pred_prob[te_mask] = pr.prob_res_dom
    return pred_sv, pred_prob


def _summarize_demo(sub: pd.DataFrame) -> dict:
    n = int(len(sub))
    n_f = int((sub["Gender"] == "female").sum())
    n_m = int((sub["Gender"] == "male").sum())
    age = sub["Age"].dropna()
    bmi = sub["BMI"].dropna()
    ht = sub["Height (cm)"].dropna()
    wt = sub["Weight (kg)"].dropna()
    lvef = sub["LVEF_num"].dropna()
    bb = int(sub["Beta-adrenoceptor antagonist"].notna().sum())
    htn_med = int(sub["Hypertension medications"].notna().sum())
    diur = int(sub["Diuretic"].notna().sum())
    return {
        "n": n,
        "n_female": n_f,
        "pct_female": round(100.0 * n_f / n, 1) if n else 0.0,
        "n_male": n_m,
        "pct_male": round(100.0 * n_m / n, 1) if n else 0.0,
        "age_mean": round(float(age.mean()), 1) if len(age) else None,
        "age_sd": round(float(age.std()), 1) if len(age) else None,
        "height_mean": round(float(ht.mean()), 1) if len(ht) else None,
        "height_sd": round(float(ht.std()), 1) if len(ht) else None,
        "weight_mean": round(float(wt.mean()), 1) if len(wt) else None,
        "weight_sd": round(float(wt.std()), 1) if len(wt) else None,
        "bmi_mean": round(float(bmi.mean()), 1) if len(bmi) else None,
        "bmi_sd": round(float(bmi.std()), 1) if len(bmi) else None,
        "lvef_n": int(len(lvef)),
        "lvef_mean": round(float(lvef.mean()), 1) if len(lvef) else None,
        "lvef_sd": round(float(lvef.std()), 1) if len(lvef) else None,
        "n_beta_blocker": bb,
        "pct_beta_blocker": round(100.0 * bb / n, 1) if n else 0.0,
        "n_htn_med": htn_med,
        "n_diuretic": diur,
    }


def main() -> None:
    demo = pd.read_csv(ROOT / "data" / "raw" / "auricle" / "demographics.csv")
    demo["BMI"] = demo["Weight (kg)"] / ((demo["Height (cm)"] / 100.0) ** 2)
    demo["LVEF_num"] = demo["LV ejection fraction"].apply(_parse_lvef)

    ep = pd.read_pickle(TABLES_CACHE / "episodes.pkl")
    active_ids = set(ep["subject"].unique())
    demo["active"] = demo["ID"].isin(active_ids)

    sex_map = dict(zip(demo["ID"], demo["Gender"]))
    ep["sex"] = ep["subject"].map(sex_map)
    assert ep["sex"].isin(["female", "male"]).all(), "Missing sex labels in episodes"

    demographics_summary = {
        "enrolled_all": _summarize_demo(demo),
        "active_all": _summarize_demo(demo[demo["active"]]),
        "active_hc": _summarize_demo(demo[demo["active"] & (demo["Cohort"] == "HC")]),
        "active_hfc": _summarize_demo(demo[demo["active"] & (demo["Cohort"] == "HFC")]),
    }

    # Sex-stratified CV and LOMO evaluation
    cfgs = {
        "ICHD-Full": CONFIG_FEATURES["ICHD-Full"],
        "Ear-PPG-Only": FEATURES_EAR_PPG,
        "PPG-Only": CONFIG_FEATURES["PPG-Only"],
    }

    y_sv = ep["dlog_sv"].to_numpy(dtype=float)
    y_dom = ep["y_res_dom"].to_numpy(dtype=int)
    subj = ep["subject"].to_numpy()
    m_f = (ep["sex"].to_numpy() == "female")
    m_m = (ep["sex"].to_numpy() == "male")

    sex_cv = {}
    sex_lomo = {}

    for name, fcols in cfgs.items():
        # 5-fold subject-disjoint CV
        pr_cv = _oof_cv(ep, fcols, kind="hybrid", enforce_identity=True)
        r_f_cv, lo_f_cv, hi_f_cv = _r_ci(y_sv[m_f], pr_cv.dlog_sv[m_f], subj[m_f], n_boot=2000, seed=42)
        r_m_cv, lo_m_cv, hi_m_cv = _r_ci(y_sv[m_m], pr_cv.dlog_sv[m_m], subj[m_m], n_boot=2000, seed=42)
        diff_cv = _unpaired_cluster_diff_r(
            y_sv[m_f], pr_cv.dlog_sv[m_f], subj[m_f],
            y_sv[m_m], pr_cv.dlog_sv[m_m], subj[m_m],
            n_boot=2000, seed=42,
        )
        sex_cv[name] = {
            "female": {
                "n_subjects": int(len(np.unique(subj[m_f]))),
                "n_episodes": int(m_f.sum()),
                "r_sv": round(r_f_cv, 4),
                "r_sv_ci": [round(lo_f_cv, 4), round(hi_f_cv, 4)],
                "dom_auc": round(float(roc_auc_score(y_dom[m_f], pr_cv.prob_res_dom[m_f])), 4),
            },
            "male": {
                "n_subjects": int(len(np.unique(subj[m_m]))),
                "n_episodes": int(m_m.sum()),
                "r_sv": round(r_m_cv, 4),
                "r_sv_ci": [round(lo_m_cv, 4), round(hi_m_cv, 4)],
                "dom_auc": round(float(roc_auc_score(y_dom[m_m], pr_cv.prob_res_dom[m_m])), 4),
            },
            "diff_f_minus_m": diff_cv,
        }

        # 4x5 subject-disjoint LOMO
        sv_lomo, prob_lomo = _lomo_oof_sv(ep, fcols, enforce_identity=True)
        r_f_lo, lo_f_lo, hi_f_lo = _r_ci(y_sv[m_f], sv_lomo[m_f], subj[m_f], n_boot=2000, seed=42)
        r_m_lo, lo_m_lo, hi_m_lo = _r_ci(y_sv[m_m], sv_lomo[m_m], subj[m_m], n_boot=2000, seed=42)
        diff_lo = _unpaired_cluster_diff_r(
            y_sv[m_f], sv_lomo[m_f], subj[m_f],
            y_sv[m_m], sv_lomo[m_m], subj[m_m],
            n_boot=2000, seed=42,
        )
        sex_lomo[name] = {
            "female": {
                "n_subjects": int(len(np.unique(subj[m_f]))),
                "n_episodes": int(m_f.sum()),
                "r_sv": round(r_f_lo, 4),
                "r_sv_ci": [round(lo_f_lo, 4), round(hi_f_lo, 4)],
                "dom_auc": round(float(roc_auc_score(y_dom[m_f], prob_lomo[m_f])), 4),
            },
            "male": {
                "n_subjects": int(len(np.unique(subj[m_m]))),
                "n_episodes": int(m_m.sum()),
                "r_sv": round(r_m_lo, 4),
                "r_sv_ci": [round(lo_m_lo, 4), round(hi_m_lo, 4)],
                "dom_auc": round(float(roc_auc_score(y_dom[m_m], prob_lomo[m_m])), 4),
            },
            "diff_f_minus_m": diff_lo,
        }

    # Also check HC-only sex balance (perfectly 21F/21M) to separate sex from HF cohort confounding
    m_hc_f = m_f & (ep["cohort"].to_numpy() == "HC")
    m_hc_m = m_m & (ep["cohort"].to_numpy() == "HC")
    pr_ear_cv = _oof_cv(ep, FEATURES_EAR_PPG, kind="hybrid", enforce_identity=True)
    hc_sex_ear_cv = _unpaired_cluster_diff_r(
        y_sv[m_hc_f], pr_ear_cv.dlog_sv[m_hc_f], subj[m_hc_f],
        y_sv[m_hc_m], pr_ear_cv.dlog_sv[m_hc_m], subj[m_hc_m],
        n_boot=2000, seed=42,
    )

    out = {
        "demographics": demographics_summary,
        "sex_stratified_cv": sex_cv,
        "sex_stratified_lomo": sex_lomo,
        "hc_balanced_ear_ppg_cv_sex_diff": hc_sex_ear_cv,
    }
    out_path = RESULTS_DIR / "exp11_sager_sex_audit.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

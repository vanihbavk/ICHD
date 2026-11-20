"""Experiment 7: Session 1 -> Session 2 Temporal Generalization & VitalDB Failure-Mode Audit.

1. Evaluates true cross-day temporal generalization on Auricle by training on Session 1
   and testing on Session 2 (both full S1->S2 and strictly subject+session-disjoint
   S1-only->S2).
2. Diagnoses the mechanistic causes of frozen PPG_Finger collapse vs. PAT_Finger/HR_Only
   preservation in VitalDB (raw ADC gain mismatch + within-case sign inversion of
   optical amplitude/slope features vs. sign-preserved HR/PAT timing).
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from ichd.model import ICHDModel, CONFIG_FEATURES, FEATURES_EAR_PPG
from ichd.stats import _r_ci, _auc_ci, _paired_dr_ci

ROOT = Path(__file__).resolve().parents[1]
TABLES_CACHE = ROOT / "data" / "interim" / "ichd_tables"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"


def _eval_split(tr_df: pd.DataFrame, te_df: pd.DataFrame, configs: dict[str, list[str]]) -> tuple[dict, dict]:
    out: dict = {}
    preds: dict = {}
    g = te_df["subject"].to_numpy()
    y_sv = te_df["dlog_sv"].to_numpy(dtype=float)
    y_co = te_df["dlog_co"].to_numpy(dtype=float)
    y_svr = te_df["dlog_svr"].to_numpy(dtype=float)
    y_dom = te_df["y_res_dom"].to_numpy(dtype=int)

    for name, fcols in configs.items():
        m = ICHDModel(fcols, kind="hybrid", enforce_identity=True).fit(tr_df)
        p = m.predict(te_df)
        preds[name] = p
        r_sv, lo_sv, hi_sv = _r_ci(y_sv, p.dlog_sv, g, n_boot=1000)
        r_co, lo_co, hi_co = _r_ci(y_co, p.dlog_co, g, n_boot=1000)
        r_svr, lo_svr, hi_svr = _r_ci(y_svr, p.dlog_svr, g, n_boot=1000)
        auc, lo_auc, hi_auc = _auc_ci(y_dom, p.prob_res_dom, g, n_boot=1000)
        acc = float(accuracy_score(y_dom, p.prob_res_dom >= 0.5))
        out[name] = {
            "r_sv": round(r_sv, 3),
            "r_sv_ci": [round(lo_sv, 3), round(hi_sv, 3)],
            "r_co": round(r_co, 3),
            "r_co_ci": [round(lo_co, 3), round(hi_co, 3)],
            "r_svr": round(r_svr, 3),
            "r_svr_ci": [round(lo_svr, 3), round(hi_svr, 3)],
            "dom_auc": round(auc, 3),
            "dom_auc_ci": [round(lo_auc, 3), round(hi_auc, 3)],
            "dom_acc": round(acc, 4),
        }
    return out, preds


def run_temporal_and_vitaldb_audit() -> dict:
    ep = pd.read_pickle(TABLES_CACHE / "episodes.pkl")
    ep["sess_num"] = ep["record"].str.split("_").str[-1]
    s1 = ep[ep["sess_num"] == "1"].reset_index(drop=True)
    s2 = ep[ep["sess_num"] == "2"].reset_index(drop=True)

    eval_configs = {
        "HR-Only": CONFIG_FEATURES["HR-Only"],
        "ECG-Only": CONFIG_FEATURES["ECG-Only"],
        "PAT-Only": CONFIG_FEATURES["PAT-Only"],
        "Chest-ECG+SCG": CONFIG_FEATURES["Chest-ECG+SCG"],
        "PPG-Only": CONFIG_FEATURES["PPG-Only"],
        "Ear-PPG-Only": FEATURES_EAR_PPG,
        "ICHD-Full": CONFIG_FEATURES["ICHD-Full"],
    }

    # 1A. Cross-day temporal split (all S1 -> all S2)
    s1_to_s2, s1_to_s2_preds = _eval_split(s1, s2, eval_configs)

    # 1B. Strictly subject-disjoint AND session-disjoint split (S1-only subjects -> S2 subjects)
    s1_subs = set(s1["subject"])
    s2_subs = set(s2["subject"])
    only_s1_subs = s1_subs - s2_subs
    s1_disjoint = s1[s1["subject"].isin(only_s1_subs)].reset_index(drop=True)
    s2_disjoint = s2[s2["subject"].isin(s2_subs - only_s1_subs)].reset_index(drop=True)
    s1_disj_res, s1_disj_preds = _eval_split(s1_disjoint, s2_disjoint, eval_configs)

    g_disj = s2_disjoint["subject"].to_numpy()
    y_disj = s2_disjoint["dlog_sv"].to_numpy(dtype=float)
    dr_ear_ppg_disj = _paired_dr_ci(y_disj, s1_disj_preds["Ear-PPG-Only"].dlog_sv, s1_disj_preds["PPG-Only"].dlog_sv, g_disj)
    dr_ear_hr_disj = _paired_dr_ci(y_disj, s1_disj_preds["Ear-PPG-Only"].dlog_sv, s1_disj_preds["HR-Only"].dlog_sv, g_disj)

    # 2. VitalDB mechanistic failure-mode audit
    vdb = pd.read_csv(RESULTS_DIR / "vitaldb_windows.csv.gz")
    audit_feats = [
        "d_ppgf_amp", "d_ppgf_slope", "d_ppgf_area_raw",
        "dlog_ppgf_amp", "dlog_ppgf_slope", "dlog_ppgf_area_raw",
        "dlog_ppgf_dia_decay", "dlog_ppgf_pat", "dlog_hr",
    ]
    feat_shift = {}
    for f in audit_feats:
        ma = ep[f].notna() & ep["dlog_sv"].notna()
        mv = vdb[f].notna() & vdb["dlog_sv"].notna()
        r_aur = float(pearsonr(ep.loc[ma, f], ep.loc[ma, "dlog_sv"])[0])
        sd_aur = float(ep.loc[ma, f].std())
        sd_vdb = float(vdb.loc[mv, f].std())
        case_rs = []
        for _, grp in vdb.groupby("caseid"):
            mg = grp[f].notna() & grp["dlog_sv"].notna()
            if mg.sum() > 10 and grp.loc[mg, f].std() > 1e-9 and grp.loc[mg, "dlog_sv"].std() > 1e-9:
                case_rs.append(float(pearsonr(grp.loc[mg, f], grp.loc[mg, "dlog_sv"])[0]))
        arr_rs = np.asarray(case_rs, dtype=float)
        feat_shift[f] = {
            "r_auricle": round(r_aur, 3),
            "r_vdb_case_mean": round(float(np.mean(arr_rs)), 3),
            "r_vdb_case_median": round(float(np.median(arr_rs)), 3),
            "vdb_negative_case_frac": round(float(np.mean(arr_rs < 0)), 3),
            "sd_auricle": round(sd_aur, 3),
            "sd_vitaldb": round(sd_vdb, 3),
            "sd_ratio_vdb_over_aur": round(sd_vdb / max(sd_aur, 1e-9), 2),
        }

    # Per-case correlation predictors of VitalDB PAT_Finger survival vs PPG_Finger collapse
    case_rows = []
    for cid, grp in vdb.groupby("caseid"):
        case_rows.append({
            "caseid": int(cid),
            "r_ppg": float(pearsonr(grp["dlog_sv"], grp["pred_PPG_Finger"])[0]),
            "r_pat": float(pearsonr(grp["dlog_sv"], grp["pred_PAT_Finger"])[0]),
            "r_hr": float(pearsonr(grp["dlog_sv"], grp["pred_HR_Only"])[0]),
            "r_sv_hr": float(pearsonr(grp["dlog_sv"], grp["dlog_hr"].fillna(0))[0]),
            "r_sv_amp": float(pearsonr(grp["dlog_sv"], grp["dlog_ppgf_amp"].fillna(0))[0]),
            "q_mean": float(grp["q"].mean()),
        })
    cdf = pd.DataFrame(case_rows)
    rho_pat_hr, p_pat_hr = spearmanr(cdf["r_pat"], cdf["r_sv_hr"])
    rho_ppg_q, p_ppg_q = spearmanr(cdf["r_ppg"], cdf["q_mean"])

    results = {
        "temporal_all_s1_to_s2": {
            "n_train_subjects": int(s1["subject"].nunique()),
            "n_train_episodes": int(len(s1)),
            "n_test_subjects": int(s2["subject"].nunique()),
            "n_test_episodes": int(len(s2)),
            "models": s1_to_s2,
        },
        "temporal_disjoint_s1only_to_s2": {
            "n_train_subjects": int(s1_disjoint["subject"].nunique()),
            "n_train_episodes": int(len(s1_disjoint)),
            "n_test_subjects": int(s2_disjoint["subject"].nunique()),
            "n_test_episodes": int(len(s2_disjoint)),
            "models": s1_disj_res,
            "paired_dr_ear_vs_ppg": {
                "dr": round(dr_ear_ppg_disj[0], 3),
                "ci": [round(dr_ear_ppg_disj[1], 3), round(dr_ear_ppg_disj[2], 3)],
                "p_val": round(dr_ear_ppg_disj[3], 4),
            },
            "paired_dr_ear_vs_hr": {
                "dr": round(dr_ear_hr_disj[0], 3),
                "ci": [round(dr_ear_hr_disj[1], 3), round(dr_ear_hr_disj[2], 3)],
                "p_val": round(dr_ear_hr_disj[3], 4),
            },
        },
        "vitaldb_failure_audit": {
            "n_cases": int(len(cdf)),
            "feature_domain_shifts": feat_shift,
            "case_level_predictors": {
                "spearman_r_pat_vs_r_sv_hr": round(float(rho_pat_hr), 3),
                "p_pat_vs_r_sv_hr": round(float(p_pat_hr), 4),
                "spearman_r_ppg_vs_q_mean": round(float(rho_ppg_q), 3),
                "p_ppg_vs_q_mean": round(float(p_ppg_q), 4),
            },
        },
    }

    with open(RESULTS_DIR / "exp7_temporal_vitaldb_audit.json", "w") as f:
        json.dump(results, f, indent=2)
    print(
        f"exp7: S1->S2 Ear={s1_to_s2['Ear-PPG-Only']['r_sv']:.3f} | "
        f"Disjoint S1->S2 Ear={s1_disj_res['Ear-PPG-Only']['r_sv']:.3f} vs PPG={s1_disj_res['PPG-Only']['r_sv']:.3f} | "
        f"VDB dlog_ppgf_amp case_mean_r={feat_shift['dlog_ppgf_amp']['r_vdb_case_mean']:+.3f}"
    )

    from ichd.plots import MINIMAL_RC, PAL_CRIMSON, PAL_GREEN, PAL_NAVY, PAL_SLATE_LT, _clean_grid

    with plt.rc_context(MINIMAL_RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 5.0))
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.14, top=0.89, wspace=0.25)

        cfgs = ["HR-Only", "ECG-Only", "Chest-ECG+SCG", "PPG-Only", "Ear-PPG-Only", "ICHD-Full"]
        lbls = ["HR", "ECG", "Chest", "Finger", "Ear", "Full"]
        x = np.arange(len(cfgs))
        w = 0.36
        v_all = [s1_to_s2[c]["r_sv"] for c in cfgs]
        v_disj = [s1_disj_res[c]["r_sv"] for c in cfgs]
        ax1.bar(x - w / 2, v_all, w, label=r"$\mathrm{S1}\to\mathrm{S2}$ (All, $N=35$ S2 subj)", color=PAL_SLATE_LT, edgecolor="#111111", lw=0.85)
        ax1.bar(x + w / 2, v_disj, w, label=r"$\mathrm{S1\text{-}Only}\to\mathrm{S2}$ (Disjoint)", color=PAL_NAVY, edgecolor="#111111", lw=0.85)
        ax1.set_xticks(x)
        ax1.set_xticklabels(lbls, fontsize=9.6)
        ax1.set_ylim(0.0, 0.86)
        ax1.set_ylabel(r"Pearson $r(\Delta\log\mathrm{SV})$")
        ax1.set_title(r"(a) Session 1 $\to$ Session 2 Temporal Holdout", pad=9)
        _clean_grid(ax1)
        ax1.legend(frameon=False, fontsize=9.0, loc="upper left")

        f_keys = ["dlog_ppgf_area_raw", "dlog_ppgf_amp", "dlog_ppgf_slope", "dlog_ppgf_pat", "dlog_hr"]
        f_lbls = ["Area", "Amp", "Slope", "PAT", "HR"]
        aur_r = [feat_shift[k]["r_auricle"] for k in f_keys]
        vdb_r = [feat_shift[k]["r_vdb_case_mean"] for k in f_keys]
        x2 = np.arange(len(f_keys))
        ax2.bar(x2 - w / 2, aur_r, w, label="Auricle Lab (Episodes)", color=PAL_GREEN, edgecolor="#111111", lw=0.85)
        ax2.bar(x2 + w / 2, vdb_r, w, label="VitalDB OR (Case Mean)", color=PAL_CRIMSON, edgecolor="#111111", lw=0.85)
        ax2.axhline(0.0, color="#222222", lw=0.85)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(f_lbls, fontsize=9.6)
        ax2.set_ylim(-0.36, 0.54)
        ax2.set_ylabel(r"Feature Correlation $r(\cdot, \Delta\log\mathrm{SV})$")
        ax2.set_title("(b) VitalDB Sign Flip: Amplitude vs. Timing", pad=9)
        _clean_grid(ax2)
        ax2.legend(frameon=False, fontsize=9.0, loc="upper right")

        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / "fig7_temporal_vitaldb.pdf", bbox_inches="tight")
        plt.close(fig)
    return results


if __name__ == "__main__":
    run_temporal_and_vitaldb_audit()

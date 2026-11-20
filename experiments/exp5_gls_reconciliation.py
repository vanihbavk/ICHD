"""Experiment 5: Full-covariance Generalized Least Squares (GLS) manifold reconciliation."""
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


def gls_project(h: np.ndarray, z_sv: np.ndarray, z_svr: np.ndarray, z_diff: np.ndarray, z_map: np.ndarray, sigma_inv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve GLS projection theta = (C^T Sigma^{-1} C)^{-1} C^T Sigma^{-1} (z - d)."""
    # C matrix is 4 x 2:
    # row 0: [1,  0]  (SV)
    # row 1: [0,  1]  (SVR)
    # row 2: [-1, 1]  (diff = SVR - CO = SVR - (h + SV) -> SVR - SV - h)
    # row 3: [1,  1]  (MAP = h + SV + SVR -> SV + SVR + h)
    C = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 1.0],
        [1.0, 1.0],
    ])
    
    # Precompute (C^T Sigma^{-1} C)^{-1} C^T Sigma^{-1}
    CTC_inv = np.linalg.inv(C.T @ sigma_inv @ C)
    proj_op = CTC_inv @ C.T @ sigma_inv  # 2 x 4
    
    n = len(h)
    u_sv = np.zeros(n)
    u_svr = np.zeros(n)
    
    for i in range(n):
        d_i = np.array([0.0, 0.0, -h[i], h[i]])
        z_i = np.array([z_sv[i], z_svr[i], z_diff[i], z_map[i]])
        rhs = z_i - d_i
        theta_i = proj_op @ rhs
        u_sv[i] = theta_i[0]
        u_svr[i] = theta_i[1]
        
    return u_sv, u_svr


def run_gls_analysis():
    ep = pd.read_pickle(TABLES_CACHE / "episodes.pkl")
    gkf = GroupKFold(n_splits=5)
    groups = ep["subject"].values
    
    # We test on ICHD-Full features
    feats = CONFIG_FEATURES["ICHD-Full"]
    
    # Run 5-fold CV without manifold projection to get unprojected predictions and residuals
    oof_z_sv = np.zeros(len(ep))
    oof_z_svr = np.zeros(len(ep))
    oof_z_diff = np.zeros(len(ep))
    oof_z_map = np.zeros(len(ep))
    
    # Store standard diagonal projected and strictly out-of-fold GLS projected outputs
    diag_u_sv = np.zeros(len(ep))
    diag_u_svr = np.zeros(len(ep))
    gls_u_sv = np.zeros(len(ep))
    gls_u_svr = np.zeros(len(ep))

    for train_idx, test_idx in gkf.split(ep, groups=groups):
        train_df, test_df = ep.iloc[train_idx], ep.iloc[test_idx]

        # Model with diagonal identity projection
        m_diag = ICHDModel(features=feats, kind="hybrid", enforce_identity=True)
        m_diag.fit(train_df)
        p_diag = m_diag.predict(test_df)
        diag_u_sv[test_idx] = p_diag.dlog_sv
        diag_u_svr[test_idx] = p_diag.dlog_svr

        # Model without identity projection (pure raw heads)
        m_raw = ICHDModel(features=feats, kind="hybrid", enforce_identity=False)
        m_raw.fit(train_df)
        p_raw = m_raw.predict(test_df)
        oof_z_sv[test_idx] = p_raw.dlog_sv
        oof_z_svr[test_idx] = p_raw.dlog_svr
        oof_z_diff[test_idx] = p_raw.vres_minus_vflow
        oof_z_map[test_idx] = p_raw.dlog_map

        # Estimate 4x4 residual covariance strictly inside train_df via internal 3-fold GroupKFold
        inner_gkf = GroupKFold(n_splits=3)
        n_tr = len(train_df)
        in_sv, in_svr, in_diff, in_map = (np.zeros(n_tr) for _ in range(4))
        for itr, ival in inner_gkf.split(train_df, groups=train_df["subject"].values):
            m_in = ICHDModel(features=feats, kind="hybrid", enforce_identity=False).fit(train_df.iloc[itr])
            p_in = m_in.predict(train_df.iloc[ival])
            in_sv[ival] = p_in.dlog_sv
            in_svr[ival] = p_in.dlog_svr
            in_diff[ival] = p_in.vres_minus_vflow
            in_map[ival] = p_in.dlog_map
        y_sv_tr = train_df["dlog_sv"].values
        y_svr_tr = train_df["dlog_svr"].values
        y_co_tr = train_df["dlog_co"].values
        y_map_tr = train_df["dlog_map"].values
        res_tr = np.column_stack([
            in_sv - y_sv_tr,
            in_svr - y_svr_tr,
            in_diff - (y_svr_tr - y_co_tr),
            in_map - y_map_tr,
        ])
        cov_tr = np.cov(res_tr, rowvar=False) + 1e-5 * np.eye(4)
        sigma_inv_tr = np.linalg.inv(cov_tr)
        g_sv_te, g_svr_te = gls_project(
            test_df["dlog_hr"].values, p_raw.dlog_sv, p_raw.dlog_svr,
            p_raw.vres_minus_vflow, p_raw.dlog_map, sigma_inv_tr
        )
        gls_u_sv[test_idx] = g_sv_te
        gls_u_svr[test_idx] = g_svr_te

    # True targets
    y_sv = ep["dlog_sv"].values
    y_svr = ep["dlog_svr"].values
    y_co = ep["dlog_co"].values
    h = ep["dlog_hr"].values
    y_map = ep["dlog_map"].values
    y_diff = y_svr - y_co

    # Pooled OOF residual correlation matrix (for diagnostic visualization)
    e_sv = oof_z_sv - y_sv
    e_svr = oof_z_svr - y_svr
    e_diff = oof_z_diff - y_diff
    e_map = oof_z_map - y_map
    residuals = np.column_stack([e_sv, e_svr, e_diff, e_map])
    cov_res = np.cov(residuals, rowvar=False)
    corr_res = np.corrcoef(residuals, rowvar=False)

    gls_u_co = h + gls_u_sv
    gls_u_map = gls_u_co + gls_u_svr

    diag_u_co = h + diag_u_sv
    diag_u_map = diag_u_co + diag_u_svr
    
    # Compute correlations
    r_diag_sv, _ = stats.pearsonr(y_sv, diag_u_sv)
    r_diag_co, _ = stats.pearsonr(y_co, diag_u_co)
    r_diag_svr, _ = stats.pearsonr(y_svr, diag_u_svr)
    
    r_gls_sv, _ = stats.pearsonr(y_sv, gls_u_sv)
    r_gls_co, _ = stats.pearsonr(y_co, gls_u_co)
    r_gls_svr, _ = stats.pearsonr(y_svr, gls_u_svr)
    
    r_raw_sv, _ = stats.pearsonr(y_sv, oof_z_sv)
    r_raw_co, _ = stats.pearsonr(y_co, h + oof_z_sv)
    r_raw_svr, _ = stats.pearsonr(y_svr, oof_z_svr)
    
    # Closure gaps
    closure_gap_raw = float(np.mean(np.abs(oof_z_map - (h + oof_z_sv + oof_z_svr))))
    closure_gap_diag = float(np.mean(np.abs(diag_u_map - (h + diag_u_sv + diag_u_svr))))
    closure_gap_gls = float(np.mean(np.abs(gls_u_map - (h + gls_u_sv + gls_u_svr))))
    
    off_diag = corr_res[~np.eye(4, dtype=bool)]
    results = {
        "residual_covariance_matrix": cov_res.tolist(),
        "residual_correlation_matrix": corr_res.tolist(),
        "corr_sv_svr": round(float(corr_res[0, 1]), 2),
        "corr_sv_diff": round(float(corr_res[0, 2]), 2),
        "max_offdiag_abs_corr": round(float(np.max(np.abs(off_diag))), 2),
        "raw_unprojected": {
            "r_sv": round(float(r_raw_sv), 3),
            "r_co": round(float(r_raw_co), 3),
            "r_svr": round(float(r_raw_svr), 3),
            "mean_closure_gap_nat": round(closure_gap_raw, 4),
        },
        "diagonal_gauss_markov": {
            "r_sv": round(float(r_diag_sv), 3),
            "r_co": round(float(r_diag_co), 3),
            "r_svr": round(float(r_diag_svr), 3),
            "mean_closure_gap_nat": round(closure_gap_diag, 15),
        },
        "full_covariance_gls": {
            "r_sv": round(float(r_gls_sv), 3),
            "r_co": round(float(r_gls_co), 3),
            "r_svr": round(float(r_gls_svr), 3),
            "mean_closure_gap_nat": round(closure_gap_gls, 15),
        },
    }

    with open(OUT_DIR / "exp5_gls_reconciliation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"exp5_gls_reconciliation: diag r_sv={r_diag_sv:.3f} -> gls r_sv={r_gls_sv:.3f}")


if __name__ == "__main__":
    run_gls_analysis()

"""Statistical bootstrap, partial-correlation, multiplicity, and CV helpers for ICHD."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as _mp
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from .model import ICHDModel, ICHDPrediction


def _r_ci(y: np.ndarray, yhat: np.ndarray, groups: np.ndarray, n_boot: int = 400, seed: int = 42) -> tuple[float, float, float]:
    m = np.isfinite(y) & np.isfinite(yhat)
    y, yhat, groups = y[m], yhat[m], groups[m]
    if len(y) < 5 or np.std(y) < 1e-9 or np.std(yhat) < 1e-9:
        return 0.0, 0.0, 0.0
    r0 = float(pearsonr(y, yhat)[0])
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_g = [np.flatnonzero(groups == g) for g in uniq]
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([idx_by_g[i] for i in pick])
        if np.std(y[idx]) > 1e-9 and np.std(yhat[idx]) > 1e-9:
            boots.append(float(pearsonr(y[idx], yhat[idx])[0]))
    if not boots:
        return r0, r0, r0
    return r0, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _boot_two_sided_p(boots: list[float]) -> float:
    """Two-sided finite-sample bootstrap p-value with plus-one smoothing 2*(k+1)/(B+1)."""
    if not boots:
        return 1.0
    b_arr = np.asarray(boots, dtype=float)
    k_min = int(min(np.sum(b_arr <= 0), np.sum(b_arr >= 0)))
    return float(min(1.0, 2.0 * (k_min + 1.0) / (len(b_arr) + 1.0)))


def _paired_dr_ci(y: np.ndarray, yhat_b: np.ndarray, yhat_a: np.ndarray, groups: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float, float, float]:
    """Paired subject-cluster bootstrap for Delta r = r(y, yhat_b) - r(y, yhat_a) and two-sided p-value."""
    m = np.isfinite(y) & np.isfinite(yhat_b) & np.isfinite(yhat_a)
    y, yhat_b, yhat_a, groups = y[m], yhat_b[m], yhat_a[m], groups[m]
    dr0 = float(pearsonr(y, yhat_b)[0] - pearsonr(y, yhat_a)[0])
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_g = [np.flatnonzero(groups == g) for g in uniq]
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([idx_by_g[i] for i in pick])
        if np.std(y[idx]) > 1e-9 and np.std(yhat_b[idx]) > 1e-9 and np.std(yhat_a[idx]) > 1e-9:
            boots.append(float(pearsonr(y[idx], yhat_b[idx])[0] - pearsonr(y[idx], yhat_a[idx])[0]))
    if not boots:
        return dr0, dr0, dr0, 1.0
    return dr0, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), _boot_two_sided_p(boots)


def _paired_did_dr_ci(
    y: np.ndarray,
    yhat_b1: np.ndarray,
    yhat_a1: np.ndarray,
    yhat_b0: np.ndarray,
    yhat_a0: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Paired subject-cluster bootstrap for interaction DiD = (r_b1 - r_a1) - (r_b0 - r_a0)."""
    m = np.isfinite(y) & np.isfinite(yhat_b1) & np.isfinite(yhat_a1) & np.isfinite(yhat_b0) & np.isfinite(yhat_a0)
    y, yhat_b1, yhat_a1, yhat_b0, yhat_a0, groups = (
        y[m], yhat_b1[m], yhat_a1[m], yhat_b0[m], yhat_a0[m], groups[m]
    )
    did0 = float((pearsonr(y, yhat_b1)[0] - pearsonr(y, yhat_a1)[0]) - (pearsonr(y, yhat_b0)[0] - pearsonr(y, yhat_a0)[0]))
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_g = [np.flatnonzero(groups == g) for g in uniq]
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([idx_by_g[i] for i in pick])
        if (
            np.std(y[idx]) > 1e-9 and np.std(yhat_b1[idx]) > 1e-9 and np.std(yhat_a1[idx]) > 1e-9
            and np.std(yhat_b0[idx]) > 1e-9 and np.std(yhat_a0[idx]) > 1e-9
        ):
            dr1 = pearsonr(y[idx], yhat_b1[idx])[0] - pearsonr(y[idx], yhat_a1[idx])[0]
            dr0 = pearsonr(y[idx], yhat_b0[idx])[0] - pearsonr(y[idx], yhat_a0[idx])[0]
            boots.append(float(dr1 - dr0))
    if not boots:
        return did0, did0, did0, 1.0
    return did0, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), _boot_two_sided_p(boots)


def _paired_within_maneuver_dr_ci(
    y: np.ndarray,
    yhat_b: np.ndarray,
    yhat_a: np.ndarray,
    act: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Paired subject-cluster bootstrap for mean within-maneuver Delta r(SV) and per-maneuver deltas."""
    acts = [1, 2, 3, 4]
    act_names = {1: "breath_hold", 2: "valsalva", 3: "exercise", 4: "cold_pressor"}

    def _eval_deltas(subset_idx: np.ndarray) -> tuple[float, dict[str, float]]:
        per_m = {}
        d_list = []
        for a in acts:
            m_a = subset_idx[act[subset_idx] == a]
            if len(m_a) > 5 and np.std(y[m_a]) > 1e-9 and np.std(yhat_b[m_a]) > 1e-9 and np.std(yhat_a[m_a]) > 1e-9:
                rb = float(pearsonr(y[m_a], yhat_b[m_a])[0])
                ra = float(pearsonr(y[m_a], yhat_a[m_a])[0])
                dr = rb - ra
            else:
                dr = 0.0
            per_m[act_names[a]] = dr
            d_list.append(dr)
        return float(np.mean(d_list)), per_m

    all_idx = np.arange(len(y))
    mean_dr0, per_m0 = _eval_deltas(all_idx)

    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_g = [np.flatnonzero(groups == g) for g in uniq]

    boot_means = []
    boot_pers = {act_names[a]: [] for a in acts}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([idx_by_g[i] for i in pick])
        bm, bp = _eval_deltas(idx)
        boot_means.append(bm)
        for an in bp:
            boot_pers[an].append(bp[an])

    out = {
        "mean_dr": mean_dr0,
        "ci": [float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))],
        "p_val": _boot_two_sided_p(boot_means),
        "per_maneuver": {},
    }
    for an in act_names.values():
        out["per_maneuver"][an] = {
            "dr": per_m0[an],
            "ci": [float(np.percentile(boot_pers[an], 2.5)), float(np.percentile(boot_pers[an], 97.5))],
            "p_val": _boot_two_sided_p(boot_pers[an]),
        }
    return out


def _incremental_r_sv_beyond_oracle(
    df: pd.DataFrame,
    yhat_sv: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Partial correlation of predicted vs true dlog_sv controlling for maneuver dummies and dlog_hr."""
    y_sv = df["dlog_sv"].to_numpy(dtype=float)
    hr = df["dlog_hr"].fillna(0.0).to_numpy(dtype=float)
    groups = df["subject"].to_numpy()

    dummies = pd.get_dummies(df["act"], drop_first=True, dtype=float).to_numpy()
    Z = np.column_stack([np.ones(len(df)), dummies, hr])

    Q, _ = np.linalg.qr(Z)
    res_true = y_sv - Q @ (Q.T @ y_sv)
    res_pred = yhat_sv - Q @ (Q.T @ yhat_sv)

    r_part0, p_part0 = pearsonr(res_true, res_pred)
    r_part0 = float(r_part0)
    p_part0 = float(p_part0)

    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_g = [np.flatnonzero(groups == g) for g in uniq]
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([idx_by_g[i] for i in pick])
        if np.std(res_true[idx]) > 1e-9 and np.std(res_pred[idx]) > 1e-9:
            boots.append(float(pearsonr(res_true[idx], res_pred[idx])[0]))

    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else [r_part0, r_part0]
    return {
        "partial_r": r_part0,
        "partial_r_ci": ci,
        "p_val": p_part0,
        "p_val_clust": _boot_two_sided_p(boots),
    }


def _holm_bonferroni_correction(hypotheses: list[dict]) -> list[dict]:
    """Apply Holm-Bonferroni step-down correction to a list of hypothesis dicts with 'p_val'."""
    m = len(hypotheses)
    indexed = sorted(enumerate(hypotheses), key=lambda x: x[1]["p_val"])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, hyp) in enumerate(indexed):
        multiplier = m - rank
        adj_p = min(1.0, hyp["p_val"] * multiplier)
        running_max = max(running_max, adj_p)
        adjusted[orig_idx] = float(running_max)

    out = []
    for hyp, adj_p in zip(hypotheses, adjusted):
        h_copy = dict(hyp)
        h_copy["holm_p_val"] = adj_p
        out.append(h_copy)
    return out


def _auc_ci(y: np.ndarray, p: np.ndarray, groups: np.ndarray, n_boot: int = 400, seed: int = 42) -> tuple[float, float, float]:
    m = np.isfinite(y) & np.isfinite(p)
    y, p, groups = y[m].astype(int), p[m], groups[m]
    if len(np.unique(y)) < 2:
        return 0.5, 0.5, 0.5
    auc0 = float(roc_auc_score(y, p))
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_g = [np.flatnonzero(groups == g) for g in uniq]
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([idx_by_g[i] for i in pick])
        if len(np.unique(y[idx])) == 2:
            boots.append(float(roc_auc_score(y[idx], p[idx])))
    if not boots:
        return auc0, auc0, auc0
    return auc0, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _eval_predictions(df: pd.DataFrame, pred: ICHDPrediction, n_boot: int = 400) -> dict:
    g = df["subject"].to_numpy()
    h = df["dlog_hr"].to_numpy(dtype=float)
    y_sv = df["dlog_sv"].to_numpy(dtype=float)
    y_co = df["dlog_co"].to_numpy(dtype=float)
    y_svr = df["dlog_svr"].to_numpy(dtype=float)
    y_map = df["dlog_map"].to_numpy(dtype=float)
    y_diff = y_svr - y_co
    y_dom = df["y_res_dom"].to_numpy(dtype=int)

    r_sv, r_sv_lo, r_sv_hi = _r_ci(y_sv, pred.dlog_sv, g, n_boot)
    r_co, r_co_lo, r_co_hi = _r_ci(y_co, pred.dlog_co, g, n_boot)
    r_svr, r_svr_lo, r_svr_hi = _r_ci(y_svr, pred.dlog_svr, g, n_boot)
    r_map, r_map_lo, r_map_hi = _r_ci(y_map, pred.dlog_map, g, n_boot)
    r_diff, r_diff_lo, r_diff_hi = _r_ci(y_diff, pred.vres_minus_vflow, g, n_boot)
    auc, auc_lo, auc_hi = _auc_ci(y_dom, pred.prob_res_dom, g, n_boot)

    yhat_dom = (pred.prob_res_dom >= 0.5).astype(int)
    acc = float(accuracy_score(y_dom, yhat_dom))
    f1 = float(f1_score(y_dom, yhat_dom))

    true_ang = np.arctan2(y_svr, y_co)
    ang_diff = np.abs(np.arctan2(np.sin(pred.hemo_angle_rad - true_ang), np.cos(pred.hemo_angle_rad - true_ang)))
    ang_deg = float(np.degrees(np.median(ang_diff)))

    post_proj_gap = float(np.mean(np.abs(pred.dlog_map - (h + pred.dlog_sv + pred.dlog_svr))))

    return {
        "r_sv": r_sv, "r_sv_ci": [r_sv_lo, r_sv_hi],
        "mae_sv": float(np.mean(np.abs(y_sv - pred.dlog_sv))),
        "rmse_sv": float(np.sqrt(np.mean((y_sv - pred.dlog_sv) ** 2))),
        "r_co": r_co, "r_co_ci": [r_co_lo, r_co_hi],
        "mae_co": float(np.mean(np.abs(y_co - pred.dlog_co))),
        "rmse_co": float(np.sqrt(np.mean((y_co - pred.dlog_co) ** 2))),
        "r_svr": r_svr, "r_svr_ci": [r_svr_lo, r_svr_hi],
        "mae_svr": float(np.mean(np.abs(y_svr - pred.dlog_svr))),
        "rmse_svr": float(np.sqrt(np.mean((y_svr - pred.dlog_svr) ** 2))),
        "r_map": r_map, "r_map_ci": [r_map_lo, r_map_hi],
        "mae_map": float(np.mean(np.abs(y_map - pred.dlog_map))),
        "rmse_map": float(np.sqrt(np.mean((y_map - pred.dlog_map) ** 2))),
        "r_diff": r_diff, "r_diff_ci": [r_diff_lo, r_diff_hi],
        "dom_auc": auc, "dom_auc_ci": [auc_lo, auc_hi],
        "dom_acc": acc,
        "dom_f1": f1,
        "median_angle_err_deg": ang_deg,
        "mean_pre_proj_gap": float(np.mean(pred.inconsistency)),
        "mean_post_proj_gap": post_proj_gap,
    }


def _fit_predict_fold(args):
    df_tr, df_te, te_idx, fcols, kind, enforce_identity = args
    m = ICHDModel(fcols, kind=kind, enforce_identity=enforce_identity).fit(df_tr)
    p = m.predict(df_te)
    return te_idx, p


def _oof_cv(df: pd.DataFrame, fcols: list[str], kind: str = "hybrid",
            enforce_identity: bool = True, n_splits: int = 5) -> ICHDPrediction:
    gkf = GroupKFold(n_splits=n_splits)
    n = len(df)
    sv, co, svr, mp, vd, pr, ang, inc = (np.zeros(n) for _ in range(8))
    tasks = [
        (df.iloc[tr], df.iloc[te], te, fcols, kind, enforce_identity)
        for tr, te in gkf.split(df, groups=df["subject"])
    ]
    n_workers = min(n_splits, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=_mp.get_context("spawn")) as ex:
        for te, p in ex.map(_fit_predict_fold, tasks):
            sv[te], co[te], svr[te], mp[te] = p.dlog_sv, p.dlog_co, p.dlog_svr, p.dlog_map
            vd[te], pr[te], ang[te], inc[te] = p.vres_minus_vflow, p.prob_res_dom, p.hemo_angle_rad, p.inconsistency
    return ICHDPrediction(sv, co, svr, mp, vd, pr, ang, inc)


def _maneuver_oracle_oof(df: pd.DataFrame, include_hr: bool = False) -> ICHDPrediction:
    """Out-of-fold maneuver-mean oracle (or maneuver + HR oracle) to benchmark between-maneuver variance."""
    gkf = GroupKFold(n_splits=5)
    n = len(df)
    sv, co, svr, mp, vd, pr, ang, inc = (np.zeros(n) for _ in range(8))
    for tr, te in gkf.split(df, groups=df["subject"]):
        df_tr = df.iloc[tr]
        df_te = df.iloc[te]
        if not include_hr:
            means = df_tr.groupby("act")[["dlog_sv", "dlog_co", "dlog_svr", "dlog_map", "y_res_dom"]].mean()
            for act_code, row in means.iterrows():
                m_te = (df_te["act"].to_numpy() == act_code)
                idx = te[m_te]
                sv[idx] = float(row["dlog_sv"])
                co[idx] = float(row["dlog_co"])
                svr[idx] = float(row["dlog_svr"])
                mp[idx] = float(row["dlog_map"])
                vd[idx] = svr[idx] - co[idx]
                pr[idx] = float(row["y_res_dom"])
                ang[idx] = np.arctan2(svr[idx], co[idx])
        else:
            x_tr = pd.get_dummies(df_tr["act"], prefix="act", dtype=float)
            x_tr["dlog_hr"] = df_tr["dlog_hr"].to_numpy(dtype=float)
            x_te = pd.get_dummies(df_te["act"], prefix="act", dtype=float).reindex(columns=x_tr.columns, fill_value=0.0)
            x_te["dlog_hr"] = df_te["dlog_hr"].to_numpy(dtype=float)
            X_tr_arr = x_tr.to_numpy(dtype=float)
            X_te_arr = x_te.to_numpy(dtype=float)
            sv[te] = Ridge(alpha=1.0).fit(X_tr_arr, df_tr["dlog_sv"]).predict(X_te_arr)
            co[te] = sv[te] + df_te["dlog_hr"].to_numpy(dtype=float)
            svr[te] = Ridge(alpha=1.0).fit(X_tr_arr, df_tr["dlog_svr"]).predict(X_te_arr)
            mp[te] = co[te] + svr[te]
            vd[te] = svr[te] - co[te]
            pr[te] = LogisticRegression(max_iter=1000).fit(X_tr_arr, df_tr["y_res_dom"].astype(int)).predict_proba(X_te_arr)[:, 1]
            ang[te] = np.arctan2(svr[te], co[te])
    return ICHDPrediction(sv, co, svr, mp, vd, pr, ang, inc)

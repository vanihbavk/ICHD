"""Conformal risk control and selective hemodynamic attribution."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .model import ICHDPrediction


def quality_col_for_config(cfg_name: str, features: list[str] | None = None) -> str | None:
    """Select modality-matched signal quality column without cross-sensor leakage."""
    name = cfg_name.lower()
    if any(k in name for k in ("chest", "scg", "ichd")):
        return "scg_q"
    if "ear" in name:
        return "ppge_q"
    if "ppg" in name or "pat" in name:
        return "ppgf_q"
    if "ecg" in name or "hr" in name:
        return "ecg_q"
    if features is not None:
        if any("pep_lag" in f or "scg" in f for f in features):
            return "scg_q"
        if any("ppge_" in f for f in features):
            return "ppge_q"
        if any("ppgf_" in f for f in features):
            return "ppgf_q"
    return None


def _sensor_quality(df: pd.DataFrame, quality_col: str | None = "scg_q", min_q: float = 0.2) -> np.ndarray:
    if quality_col and quality_col in df.columns:
        return np.clip(df[quality_col].fillna(0.5).to_numpy(dtype=float), min_q, 1.0)
    return np.ones(len(df), dtype=float)


def confidence_score(pred: ICHDPrediction, df: pd.DataFrame,
                     quality_col: str | None = "scg_q") -> np.ndarray:
    """Quality-aware confidence score for flow-vs-resistance attribution."""
    prob_margin = 2.0 * np.abs(pred.prob_res_dom - 0.5)
    vec_margin = np.tanh(np.abs(pred.vres_minus_vflow) / 0.10)
    q = _sensor_quality(df, quality_col=quality_col, min_q=0.1)
    closure_factor = np.exp(-2.0 * pred.inconsistency)
    return prob_margin * (0.5 + 0.5 * vec_margin) * np.sqrt(q) * closure_factor


def risk_coverage_curve(y_true: np.ndarray, y_hat: np.ndarray, conf: np.ndarray,
                        n_grid: int = 100) -> dict[str, np.ndarray | float]:
    """Selective classification risk and accuracy over coverage c in [0.1, 1.0], plus AURC."""
    order = np.argsort(-conf)
    err = (y_true[order] != y_hat[order]).astype(float)
    n = len(err)
    covs = np.linspace(0.10, 1.0, n_grid)
    risks = np.empty_like(covs)
    accs = np.empty_like(covs)
    for i, c in enumerate(covs):
        k = max(1, int(np.round(c * n)))
        risks[i] = float(np.mean(err[:k]))
        accs[i] = 1.0 - risks[i]
    aurc = float(np.trapezoid(risks, covs) / (covs[-1] - covs[0]))
    return {"coverages": covs, "risks": risks, "accuracies": accs, "aurc": aurc}


@dataclass
class Conformal2DRegion:
    q_alpha: float
    sigma_co: float
    sigma_svr: float
    q_diff_alpha: float = 0.0
    sigma_diff: float = 0.1
    quality_col: str | None = "scg_q"

    def half_widths(self, pred: ICHDPrediction, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        q = _sensor_quality(df, quality_col=self.quality_col, min_q=0.2)
        scale = (1.0 + 1.5 * pred.inconsistency) / np.sqrt(q)
        return self.q_alpha * self.sigma_co * scale, self.q_alpha * self.sigma_svr * scale

    def margin_half_width(self, pred: ICHDPrediction, df: pd.DataFrame) -> np.ndarray:
        q = _sensor_quality(df, quality_col=self.quality_col, min_q=0.2)
        scale = (1.0 + 1.5 * pred.inconsistency) / np.sqrt(q)
        return self.q_diff_alpha * self.sigma_diff * scale

    def evaluate(self, pred: ICHDPrediction, df: pd.DataFrame) -> dict[str, float]:
        hw_co, hw_svr = self.half_widths(pred, df)
        hw_diff = self.margin_half_width(pred, df)
        y_co = df["dlog_co"].to_numpy(dtype=float)
        y_svr = df["dlog_svr"].to_numpy(dtype=float)
        y_diff = y_svr - y_co
        cov_co = np.abs(y_co - pred.dlog_co) <= hw_co
        cov_svr = np.abs(y_svr - pred.dlog_svr) <= hw_svr
        cov_diff = np.abs(y_diff - pred.vres_minus_vflow) <= hw_diff
        joint = cov_co & cov_svr

        cert_box = np.abs(pred.vres_minus_vflow) > (hw_co + hw_svr)
        cert_margin = np.abs(pred.vres_minus_vflow) > hw_diff
        y_dom = df["y_res_dom"].to_numpy(dtype=int)
        y_hat_vec = (pred.vres_minus_vflow > 0.0).astype(int)
        cert_acc = float(np.mean(y_dom[cert_box] == y_hat_vec[cert_box])) if cert_box.sum() > 0 else np.nan
        cert_margin_acc = float(np.mean(y_dom[cert_margin] == y_hat_vec[cert_margin])) if cert_margin.sum() > 0 else np.nan
        return {
            "joint_coverage": float(np.mean(joint)),
            "co_coverage": float(np.mean(cov_co)),
            "svr_coverage": float(np.mean(cov_svr)),
            "diff_coverage": float(np.mean(cov_diff)),
            "median_hw_co": float(np.median(hw_co)),
            "median_hw_svr": float(np.median(hw_svr)),
            "median_hw_diff": float(np.median(hw_diff)),
            "certified_fraction": float(np.mean(cert_box)),
            "certified_accuracy": cert_acc,
            "certified_margin_fraction": float(np.mean(cert_margin)),
            "certified_margin_accuracy": cert_margin_acc,
        }


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted quantile for subject-cluster exchangeable conformal calibration."""
    order = np.argsort(values)
    v_sorted = values[order]
    w_sorted = weights[order]
    cdf = np.cumsum(w_sorted) / np.sum(w_sorted)
    idx = int(np.searchsorted(cdf, min(max(q, 0.0), 1.0), side="left"))
    return float(v_sorted[min(idx, len(v_sorted) - 1)])


def calibrate_2d_conformal(pred_cal: ICHDPrediction, df_cal: pd.DataFrame,
                           alpha: float = 0.10, alpha_margin: float = 0.10,
                           quality_col: str | None = "scg_q") -> Conformal2DRegion:
    """Calibrate quality-adaptive 2D conformal box and 1D directional margin interval
    using subject-cluster inverse-size weighting for subject-level exchangeability."""
    y_co = df_cal["dlog_co"].to_numpy(dtype=float)
    y_svr = df_cal["dlog_svr"].to_numpy(dtype=float)
    y_diff = y_svr - y_co

    r_co = np.abs(y_co - pred_cal.dlog_co)
    r_svr = np.abs(y_svr - pred_cal.dlog_svr)
    r_diff = np.abs(y_diff - pred_cal.vres_minus_vflow)

    sigma_co = float(np.median(r_co)) * 1.4826 + 1e-4
    sigma_svr = float(np.median(r_svr)) * 1.4826 + 1e-4
    sigma_diff = float(np.median(r_diff)) * 1.4826 + 1e-4

    q = _sensor_quality(df_cal, quality_col=quality_col, min_q=0.2)
    scale = (1.0 + 1.5 * pred_cal.inconsistency) / np.sqrt(q)

    nonconf_2d = np.maximum(r_co / sigma_co, r_svr / sigma_svr) / scale
    nonconf_diff = (r_diff / sigma_diff) / scale

    if "subject" in df_cal.columns and df_cal["subject"].nunique() >= 5:
        subs = df_cal["subject"].to_numpy()
        _, inv, counts = np.unique(subs, return_inverse=True, return_counts=True)
        weights = 1.0 / counts[inv]
        n_eff = int(len(counts))
    else:
        weights = np.ones(len(nonconf_2d), dtype=float)
        n_eff = len(nonconf_2d)

    level_2d = min(1.0, np.ceil((n_eff + 1) * (1.0 - alpha)) / n_eff)
    level_diff = min(1.0, np.ceil((n_eff + 1) * (1.0 - alpha_margin)) / n_eff)
    return Conformal2DRegion(
        q_alpha=_weighted_quantile(nonconf_2d, weights, level_2d),
        sigma_co=sigma_co,
        sigma_svr=sigma_svr,
        q_diff_alpha=_weighted_quantile(nonconf_diff, weights, level_diff),
        sigma_diff=sigma_diff,
        quality_col=quality_col,
    )

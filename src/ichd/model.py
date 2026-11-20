"""Hemodynamic vector estimators and identity-constrained manifold reconciliation.

Given window-level baseline-normalized delta/log-ratio features and directly measured
ECG log-rate change h = dlog_hr, we estimate u = (dlog_co, dlog_svr)^T on:
    u_co = h + u_sv,  u_map = u_co + u_svr,  u_diff = u_svr - u_co.

Couples decoupled z_sv(x_card), z_svr(x_vasc), and rate-free joint heads z_diff(x_joint),
z_map(x_joint) via out-of-fold precision weights in closed form (2x2 Gauss-Markov).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier


# Dimensionless baseline-normalized log-ratio feature sets across sensor configurations
FEATURES_HR_ONLY = [
    "dlog_hr",
    "d_hr_bpm",
]

FEATURES_ECG_ONLY = FEATURES_HR_ONLY + [
    "d_rr_sd",
    "dlog_rr_sd",
    "d_ecg_rt",
    "dlog_ecg_rt",
    "d_ecg_t_over_r",
    "dlog_ecg_t_over_r",
]

FEATURES_CHEST_SCG = list(dict.fromkeys(FEATURES_ECG_ONLY + [
    "d_pep_lag", "dlog_pep_lag",
    "d_lvet_lag", "dlog_lvet_lag",
    "dlog_weissler",
    "d_scg5_ao_3d", "d_scg5_ao_loc", "dlog_scg5_ao_loc",
    "d_scg5_lvet_loc", "dlog_scg5_lvet_loc", "d_scg5_axis_cos",
    "dlog_scg5_sys_net", "dlog_scg5_dia_net",
    "dlog_scg12_sys_net", "dlog_scg12_dia_net", "d_scg12_s2_s1", "dlog_scg12_s2_s1", "d_scg12_ao_3d",
]))

_SIGNED_PPG_SUFFIXES = {"dia_slope", "apg_ba", "apg_da", "apg_agi"}


def _ppg_feature_cols(pfx: str) -> list[str]:
    sufs = [
        "pat", "inv_pat2", "amp", "slope", "max_dp_norm", "rise",
        "pw25", "pw50", "pw75", "hw", "t_notch", "notch_amp", "ri_dia",
        "t_sys_dia", "ipa", "area_sys_norm", "area_raw", "dia_decay", "dia_slope",
        "apg_ba", "apg_da", "apg_ea", "apg_agi",
    ]
    out = []
    for s in sufs:
        out.append(f"d_{pfx}_{s}")
        if s not in _SIGNED_PPG_SUFFIXES:
            out.append(f"dlog_{pfx}_{s}")
    return out

FEATURES_EAR_PPG = list(dict.fromkeys(FEATURES_ECG_ONLY + _ppg_feature_cols("ppge")))

FEATURES_PPG_FINGER = list(dict.fromkeys(FEATURES_ECG_ONLY + _ppg_feature_cols("ppgf")))

FEATURES_CHEST_EAR_PPG = list(dict.fromkeys(
    FEATURES_CHEST_SCG + _ppg_feature_cols("ppge")
    + ["d_ppge_ptt", "dlog_ppge_ptt", "d_ppge_ptt5", "dlog_ppge_ptt5"]
))

FEATURES_FULL_ICHD = list(dict.fromkeys(
    FEATURES_CHEST_EAR_PPG + _ppg_feature_cols("ppgf")
    + ["d_ppgf_ptt", "dlog_ppgf_ptt", "d_ppgf_ptt5", "dlog_ppgf_ptt5",
       "d_ptt_fe", "dlog_ptt_fe", "d_inv_ptt_fe2", "dlog_inv_ptt_fe2"]
))

FEATURES_PAT_ONLY = [
    "dlog_hr",
    "d_hr_bpm",
    "d_ppgf_pat",
    "dlog_ppgf_pat",
    "d_ppge_pat",
    "dlog_ppge_pat",
]

FEATURES_SCG_ONLY = [c for c in FEATURES_CHEST_SCG if c not in FEATURES_ECG_ONLY]
FEATURES_EAR_OPTICAL_ONLY = _ppg_feature_cols("ppge")
FEATURES_FINGER_OPTICAL_ONLY = _ppg_feature_cols("ppgf")
FEATURES_PPG_ONLY = FEATURES_PPG_FINGER
_RAW_AMP_SUFFIXES = (
    "_ppge_amp",
    "_ppgf_amp",
    "_ppge_slope",
    "_ppgf_slope",
    "_ppge_area_raw",
    "_ppgf_area_raw",
)
FEATURES_EAR_NO_AMP = [c for c in FEATURES_EAR_PPG if not c.endswith(_RAW_AMP_SUFFIXES)]
FEATURES_FINGER_NO_AMP = [c for c in FEATURES_PPG_FINGER if not c.endswith(_RAW_AMP_SUFFIXES)]

CONFIG_FEATURES = {
    "PAT-Only": FEATURES_PAT_ONLY,
    "HR-Only": FEATURES_HR_ONLY,
    "ECG-Only": FEATURES_ECG_ONLY,
    "PPG-Only": FEATURES_PPG_ONLY,
    "Chest-ECG+SCG": FEATURES_CHEST_SCG,
    "Chest+Ear-PPG": FEATURES_CHEST_EAR_PPG,
    "ICHD-Full": FEATURES_FULL_ICHD,
}


def _split_subsystem_cols(features: list[str], available: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Partition features into decoupled cardiogenic (f_card), vascular (f_vasc),
    and rate-free joint (f_joint) subsets.

    HR observables (dlog_hr, d_hr_bpm) are withheld from f_card and f_joint, while
    f_vasc retains hr/rr_sd (and sub-minimal baselines fall back to cols).
    """
    cols = [c for c in features if c in available]
    hr_spanned = ("dlog_hr", "d_hr_bpm")
    card_keys = (
        "pep_lag", "lvet_lag", "weissler",
        "scg5_ao", "scg5_lvet", "scg5_sys_net", "scg5_dia_net", "scg5_axis",
        "scg12_sys", "scg12_dia_net", "scg12_ao",
        "ecg_rt", "ecg_t_over_r",
        "t_notch", "area_sys_norm", "area_raw",
        "ppgf_amp", "ppge_amp",
        "ppgf_slope", "ppge_slope",
        "max_dp_norm", "rise", "pw50",
    )
    vasc_keys = (
        "hr", "rr_sd",
        "pat", "inv_pat2", "ptt", "ptt5", "ptt_fe",
        "pw25", "pw75", "hw",
        "notch_amp", "ri_dia", "t_sys_dia", "ipa",
        "dia_decay", "dia_slope",
        "apg_ba", "apg_da", "apg_ea", "apg_agi",
        "scg12_s2_s1",
    )
    f_card = [c for c in cols if any(k in c for k in card_keys) and c not in hr_spanned]
    f_vasc = [c for c in cols if any(k in c for k in vasc_keys) and c not in f_card]
    f_joint = [c for c in cols if c not in hr_spanned]
    if len(f_joint) < 2:
        f_joint = cols
    if len(f_card) < 2:
        # Sub-minimal baselines (HR-Only, PAT-Only) lack dedicated cardiac timing features
        f_card = f_joint or cols[:1]
    if len(f_vasc) < 2:
        f_vasc = cols[1:] if len(cols) >= 2 else cols
    return f_card, f_vasc, f_joint



def _clip_z(X: np.ndarray) -> np.ndarray:
    """Bound RobustScaler z-scores to +/-12 IQR to guard linear heads against denominator spikes."""
    return np.clip(X, -12.0, 12.0)


def _make_reg(kind: str = "ridge", seed: int = 42):
    if kind == "ridge":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", RobustScaler()),
            ("clip", FunctionTransformer(_clip_z)),
            ("reg", RidgeCV(alphas=np.logspace(-1, 4, 25))),
        ])
    elif kind == "gbm":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("reg", HistGradientBoostingRegressor(
                max_depth=4, max_iter=120, learning_rate=0.05,
                min_samples_leaf=20, l2_regularization=2.0, random_state=seed
            )),
        ])
    raise ValueError(f"Unknown regressor kind: {kind}")


def _make_clf(kind: str = "ridge", seed: int = 42):
    if kind == "ridge":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", RobustScaler()),
            ("clip", FunctionTransformer(_clip_z)),
            ("clf", LogisticRegression(C=0.5, max_iter=500, class_weight="balanced", random_state=seed)),
        ])
    elif kind == "gbm":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(
                max_depth=4, max_iter=120, learning_rate=0.05,
                min_samples_leaf=20, l2_regularization=2.0, class_weight="balanced", random_state=seed
            )),
        ])
    raise ValueError(f"Unknown classifier kind: {kind}")


def reconcile_manifold_2x2(
    h: np.ndarray,
    z_sv: np.ndarray,
    z_svr: np.ndarray,
    z_diff: np.ndarray,
    z_map: np.ndarray,
    var_sv: float,
    var_svr: float,
    var_diff: float,
    var_map: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form 2x2 Gauss-Markov / Lagrange manifold projection solving:
        min_{u_sv, u_svr}  w_sv (u_sv - z_sv)^2
                         + w_svr (u_svr - z_svr)^2
                         + w_diff ((u_svr - h - u_sv) - z_diff)^2
                         + w_map ((h + u_sv + u_svr) - z_map)^2
    subject to exact definitions u_co = h + u_sv and u_map = u_co + u_svr.
    """
    w_sv = 1.0 / max(var_sv, 1e-5)
    w_svr = 1.0 / max(var_svr, 1e-5)
    w_diff = 1.0 / max(var_diff, 1e-5)
    # Downweight noisy scalar MAP channel slightly so it cannot corrupt strong SV/SVR estimates
    w_map = 0.25 / max(var_map, 1e-5)

    a11 = w_sv + w_diff + w_map
    a22 = w_svr + w_diff + w_map
    a12 = w_map - w_diff
    det = a11 * a22 - a12 * a12

    b1 = w_sv * z_sv - w_diff * (h + z_diff) + w_map * (z_map - h)
    b2 = w_svr * z_svr + w_diff * (h + z_diff) + w_map * (z_map - h)

    u_sv = (a22 * b1 - a12 * b2) / det
    u_svr = (a11 * b2 - a12 * b1) / det
    u_co = h + u_sv
    u_map = u_co + u_svr
    return u_sv, u_co, u_svr, u_map


@dataclass
class ICHDPrediction:
    dlog_sv: np.ndarray
    dlog_co: np.ndarray
    dlog_svr: np.ndarray
    dlog_map: np.ndarray
    vres_minus_vflow: np.ndarray
    prob_res_dom: np.ndarray
    hemo_angle_rad: np.ndarray
    inconsistency: np.ndarray  # Pre-projection manifold discrepancy (nat)


class _SingleICHDModel:
    def __init__(self, features: list[str], kind: str = "ridge",
                 enforce_identity: bool = True, seed: int = 42):
        self.features = list(features)
        self.kind = kind
        self.enforce_identity = enforce_identity
        self.seed = seed
        self.reg_sv = _make_reg(kind, seed)
        self.reg_svr = _make_reg(kind, seed + 1)
        self.reg_diff = _make_reg(kind, seed + 2)
        self.reg_map = _make_reg(kind, seed + 3)
        self.clf_dom = _make_clf(kind, seed + 4)
        self.f_card_: list[str] = []
        self.f_vasc_: list[str] = []
        self.f_full_: list[str] = []
        self.var_sv = 1.0
        self.var_svr = 1.0
        self.var_diff = 1.0
        self.var_map = 1.0
        self.v_diff_scale_ = 0.15

    def fit(self, df: pd.DataFrame) -> "_SingleICHDModel":
        self.f_card_, self.f_vasc_, self.f_full_ = _split_subsystem_cols(
            self.features, list(df.columns)
        )
        X_card = df[self.f_card_].to_numpy(dtype=float)
        X_vasc = df[self.f_vasc_].to_numpy(dtype=float)
        X_full = df[self.f_full_].to_numpy(dtype=float)

        h = df["dlog_hr"].to_numpy(dtype=float)
        y_sv = df["dlog_sv"].to_numpy(dtype=float)
        y_svr = df["dlog_svr"].to_numpy(dtype=float)
        y_map = df["dlog_map"].to_numpy(dtype=float)
        y_diff = y_svr - (h + y_sv)
        y_dom = df["y_res_dom"].to_numpy(dtype=int)

        # Estimate out-of-fold residual variances via internal 3-fold subject-disjoint CV
        # to prevent GBM within-subject overfitting bias
        n = len(df)
        has_groups = "subject" in df.columns and df["subject"].nunique() >= 3
        if n >= 15 and has_groups:
            groups = df["subject"].to_numpy()
            splitter = GroupKFold(n_splits=3).split(X_full, groups=groups)
        elif n >= 15:
            splitter = KFold(n_splits=3, shuffle=True, random_state=self.seed).split(X_full)
        else:
            splitter = None

        if splitter is not None:
            oof_sv = np.zeros(n)
            oof_svr = np.zeros(n)
            oof_diff = np.zeros(n)
            oof_map = np.zeros(n)
            for tr, val in splitter:
                r_sv = _make_reg(self.kind, self.seed).fit(X_card[tr], y_sv[tr])
                r_svr = _make_reg(self.kind, self.seed + 1).fit(X_vasc[tr], y_svr[tr])
                r_diff = _make_reg(self.kind, self.seed + 2).fit(X_full[tr], y_diff[tr])
                r_map = _make_reg(self.kind, self.seed + 3).fit(X_full[tr], y_map[tr])
                oof_sv[val] = r_sv.predict(X_card[val])
                oof_svr[val] = r_svr.predict(X_vasc[val])
                oof_diff[val] = r_diff.predict(X_full[val])
                oof_map[val] = r_map.predict(X_full[val])
            self.var_sv = float(np.mean((y_sv - oof_sv) ** 2)) + 1e-4
            self.var_svr = float(np.mean((y_svr - oof_svr) ** 2)) + 1e-4
            self.var_diff = float(np.mean((y_diff - oof_diff) ** 2)) + 1e-4
            self.var_map = float(np.mean((y_map - oof_map) ** 2)) + 1e-4
        else:
            self.var_sv = float(np.var(y_sv)) + 1e-3
            self.var_svr = float(np.var(y_svr)) + 1e-3
            self.var_diff = float(np.var(y_diff)) + 1e-3
            self.var_map = float(np.var(y_map)) + 1e-3

        self.reg_sv.fit(X_card, y_sv)
        self.reg_svr.fit(X_vasc, y_svr)
        self.reg_diff.fit(X_full, y_diff)
        self.reg_map.fit(X_full, y_map)
        self.clf_dom.fit(X_full, y_dom)

        # Fit structural sigmoid scale on training fold only
        z_sv_tr = self.reg_sv.predict(X_card)
        z_svr_tr = self.reg_svr.predict(X_vasc)
        z_diff_tr = self.reg_diff.predict(X_full)
        z_map_tr = self.reg_map.predict(X_full)
        if self.enforce_identity:
            _, u_co_tr, u_svr_tr, _ = reconcile_manifold_2x2(
                h, z_sv_tr, z_svr_tr, z_diff_tr, z_map_tr,
                self.var_sv, self.var_svr, self.var_diff, self.var_map,
            )
            v_diff_tr = u_svr_tr - u_co_tr
        else:
            v_diff_tr = z_svr_tr - (h + z_sv_tr)
        self.v_diff_scale_ = max(float(np.std(v_diff_tr)), 0.05)
        return self

    def predict(self, df: pd.DataFrame) -> ICHDPrediction:
        X_card = df[self.f_card_].to_numpy(dtype=float)
        X_vasc = df[self.f_vasc_].to_numpy(dtype=float)
        X_full = df[self.f_full_].to_numpy(dtype=float)
        h = df["dlog_hr"].to_numpy(dtype=float)

        z_sv = self.reg_sv.predict(X_card)
        z_svr = self.reg_svr.predict(X_vasc)
        z_diff = self.reg_diff.predict(X_full)
        z_map = self.reg_map.predict(X_full)

        gap_map = z_map - (h + z_sv + z_svr)
        gap_diff = z_diff - (z_svr - h - z_sv)
        inconsistency = 0.5 * (np.abs(gap_map) + np.abs(gap_diff))

        if self.enforce_identity:
            u_sv, u_co, u_svr, u_map = reconcile_manifold_2x2(
                h, z_sv, z_svr, z_diff, z_map,
                self.var_sv, self.var_svr, self.var_diff, self.var_map,
            )
            v_diff = u_svr - u_co
        else:
            u_sv = z_sv
            u_svr = z_svr
            u_co = h + u_sv
            u_map = z_map
            v_diff = u_svr - u_co

        p_struct = 1.0 / (1.0 + np.exp(-v_diff / self.v_diff_scale_))
        if self.enforce_identity:
            prob_res_dom = p_struct
        else:
            p_clf = self.clf_dom.predict_proba(X_full)[:, 1]
            prob_res_dom = 0.5 * p_clf + 0.5 * p_struct
        angle = np.arctan2(u_svr, u_co)

        return ICHDPrediction(
            dlog_sv=u_sv,
            dlog_co=u_co,
            dlog_svr=u_svr,
            dlog_map=u_map,
            vres_minus_vflow=v_diff,
            prob_res_dom=prob_res_dom,
            hemo_angle_rad=angle,
            inconsistency=inconsistency,
        )


class ICHDModel:
    """Identity-Constrained Hemodynamic Vector Estimator.

    Supports:
      * kind='ridge':  Pure linear log-ratio estimator
      * kind='gbm':    Pure gradient-boosted tree estimator
      * kind='hybrid': Physics-guided blend (0.5 * Ridge + 0.5 * GBM) combining
                       smooth log-linear extrapolation with nonlinear interactions
    """

    def __init__(self, features: list[str], kind: str = "hybrid",
                 enforce_identity: bool = True, seed: int = 42):
        self.features = list(features)
        self.kind = kind
        self.enforce_identity = enforce_identity
        self.seed = seed
        if kind == "hybrid":
            self.m_ridge = _SingleICHDModel(features, "ridge", enforce_identity, seed)
            self.m_gbm = _SingleICHDModel(features, "gbm", enforce_identity, seed)
        else:
            self.m_single = _SingleICHDModel(features, kind, enforce_identity, seed)

    def fit(self, df: pd.DataFrame) -> "ICHDModel":
        if self.kind == "hybrid":
            self.m_ridge.fit(df)
            self.m_gbm.fit(df)
        else:
            self.m_single.fit(df)
        return self

    def predict(self, df: pd.DataFrame) -> ICHDPrediction:
        if self.kind != "hybrid":
            return self.m_single.predict(df)
        p1 = self.m_ridge.predict(df)
        p2 = self.m_gbm.predict(df)
        u_sv = 0.5 * (p1.dlog_sv + p2.dlog_sv)
        u_co = 0.5 * (p1.dlog_co + p2.dlog_co)
        u_svr = 0.5 * (p1.dlog_svr + p2.dlog_svr)
        u_map = 0.5 * (p1.dlog_map + p2.dlog_map)
        v_diff = u_svr - u_co
        prob = 0.5 * (p1.prob_res_dom + p2.prob_res_dom)
        inc = 0.5 * (p1.inconsistency + p2.inconsistency)
        return ICHDPrediction(
            dlog_sv=u_sv,
            dlog_co=u_co,
            dlog_svr=u_svr,
            dlog_map=u_map,
            vres_minus_vflow=v_diff,
            prob_res_dom=prob,
            hemo_angle_rad=np.arctan2(u_svr, u_co),
            inconsistency=inc,
        )

"""Unit and integration test suite for Identity-Constrained Hemodynamic
Decomposition (ICHD)."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ichd import signals as S
from ichd import episodes as E
from ichd import extract as X
from ichd.model import ICHDModel, CONFIG_FEATURES, reconcile_manifold_2x2, _split_subsystem_cols
from ichd.conformal import (
    confidence_score,
    risk_coverage_curve,
    calibrate_2d_conformal,
    quality_col_for_config,
    _weighted_quantile,
)

ROOT = Path(__file__).resolve().parents[1]


def test_r_peak_detector_synthetic_ecg():
    fs = 1000.0
    t = np.arange(0, 15.0, 1.0 / fs)
    ecg = np.zeros_like(t)
    true_r = np.arange(600, 14500, 800)  # 75 bpm
    for r in true_r:
        idx = np.arange(-25, 26)
        ecg[r + idx] += 1.2 * np.exp(-0.5 * (idx / 6.0) ** 2)
    detected = S.detect_r_peaks(ecg, fs=fs)
    assert len(detected) == len(true_r)
    assert np.max(np.abs(detected - true_r)) <= 2
    mask = S.clean_rr_mask(detected, fs=fs)
    assert mask[1:].all()


def test_ensemble_snr_reduction_law():
    rng = np.random.default_rng(123)
    pre, post = 100, 400
    L = pre + post
    tmpl = np.sin(2.0 * np.pi * 15.0 * np.arange(L) / 1000.0) * 0.010
    n_beats = 16
    sig = np.zeros(n_beats * 800 + 1000)
    r_peaks = np.arange(400, 400 + n_beats * 800, 800)
    for r in r_peaks:
        sig[r - pre: r + post] += tmpl
    noise_sigma = 0.004
    noisy = sig + rng.normal(0.0, noise_sigma, size=len(sig))
    ens, q, nb = S.ensemble(noisy, r_peaks, pre, post)
    assert ens is not None
    assert nb == n_beats
    assert q > 0.70
    err_rms = float(np.sqrt(np.mean((ens - tmpl) ** 2)))
    assert err_rms < noise_sigma / np.sqrt(n_beats) * 1.5


def test_extract_ppg_contour_synthetic_and_edge_cases():
    fs = 1000.0
    pre = 100
    n = pre + 750
    t = (np.arange(n) - pre) / fs
    # Synthesize PPG pulse: foot near 180 ms, systolic peak near 290 ms, dicrotic notch near 420 ms
    pulse = np.zeros(n)
    active = t >= 0.16
    tau = t[active] - 0.16
    sys_wave = 1.0 * np.exp(-((tau - 0.12) / 0.055) ** 2)
    dia_wave = 0.42 * np.exp(-((tau - 0.30) / 0.11) ** 2)
    pulse[active] = sys_wave + dia_wave

    feats = S.extract_ppg_contour(pulse, pre=pre, search=(80, 450), fs=fs)
    for k in ("pat", "inv_pat2", "amp", "slope", "max_dp_norm", "rise", "pw50", "t_notch", "notch_amp", "ipa", "apg_agi"):
        assert k in feats, f"missing {k}"
        assert np.isfinite(feats[k])
    assert 130.0 <= feats["pat"] <= 230.0
    assert feats["amp"] > 0.5

    assert S.extract_ppg_contour(None, pre=pre, search=(80, 450), fs=fs) == {}
    assert S.extract_ppg_contour(np.zeros(200), pre=pre, search=(80, 450), fs=fs) == {}


def test_extract_helpers_onset_and_ref_median():
    act = np.array([0, 0, 0, 3, 3, 2, 0])
    assert X._first_maneuver_onset(act) == 3
    assert X._first_maneuver_onset(np.zeros(10, dtype=int)) == 10

    arr = np.array([70.0, 72.0, 74.0, -5.0, 71.0, 73.0])
    assert X._ref_median(arr, 40.0, 150.0) == pytest.approx(72.0)
    assert np.isnan(X._ref_median(np.array([10.0, 12.0, 75.0]), 40.0, 150.0))


def test_identity_constrained_projection_exact_closure_and_no_leakage():
    rng = np.random.default_rng(42)
    n = 120
    cols = {
        "subject": [f"S_{i // 10}" for i in range(n)],
        "dlog_hr": rng.normal(0.05, 0.10, size=n),
        "dlog_pep_lag": rng.normal(-0.03, 0.08, size=n),
        "dlog_lvet_lag": rng.normal(0.02, 0.06, size=n),
        "d_scg_amp_log": rng.normal(0.10, 0.20, size=n),
        "dlog_ppgf_ptt": rng.normal(-0.04, 0.06, size=n),
        "dlog_ppgf_amp": rng.normal(-0.10, 0.25, size=n),
        "scg_q": rng.uniform(0.5, 0.95, size=n),
    }
    for c in CONFIG_FEATURES["ICHD-Full"]:
        if c not in cols:
            cols[c] = rng.normal(0.0, 0.05, size=n)
    cols["dlog_sv"] = -0.8 * cols["dlog_pep_lag"] + 0.6 * cols["dlog_lvet_lag"] + rng.normal(0, 0.02, n)
    cols["dlog_co"] = cols["dlog_hr"] + cols["dlog_sv"]
    cols["dlog_svr"] = -0.5 * cols["dlog_ppgf_ptt"] - 0.3 * cols["dlog_ppgf_amp"] + rng.normal(0, 0.03, n)
    cols["dlog_map"] = cols["dlog_co"] + cols["dlog_svr"]
    cols["y_res_dom"] = (cols["dlog_svr"] > cols["dlog_co"]).astype(int)
    df = pd.DataFrame(cols)

    for kind in ("ridge", "gbm", "hybrid"):
        m = ICHDModel(CONFIG_FEATURES["ICHD-Full"], kind=kind, enforce_identity=True).fit(df)
        p_batch = m.predict(df)
        p_one = m.predict(df.iloc[[0]])
        assert abs(p_batch.prob_res_dom[0] - p_one.prob_res_dom[0]) < 1e-12
        assert np.max(np.abs((p_batch.dlog_co + p_batch.dlog_svr) - p_batch.dlog_map)) < 1e-12
        assert np.max(np.abs((df["dlog_hr"].to_numpy() + p_batch.dlog_sv) - p_batch.dlog_co)) < 1e-12


def test_conformal_2d_and_risk_coverage():
    rng = np.random.default_rng(99)
    n = 200
    cols = {
        "subject": [f"S_{i // 10}" for i in range(n)],
        "dlog_hr": rng.normal(0.0, 0.1, n),
        "scg_q": rng.uniform(0.4, 0.95, n),
    }
    for c in CONFIG_FEATURES["Chest-ECG+SCG"]:
        if c not in cols:
            cols[c] = rng.normal(0.0, 0.1, n)
    cols["dlog_sv"] = -0.9 * cols["dlog_pep_lag"] + rng.normal(0, 0.03, n)
    cols["dlog_co"] = cols["dlog_hr"] + cols["dlog_sv"]
    cols["dlog_svr"] = -0.7 * cols["dlog_co"] + rng.normal(0, 0.04, n)
    cols["dlog_map"] = cols["dlog_co"] + cols["dlog_svr"]
    cols["y_res_dom"] = (cols["dlog_svr"] > cols["dlog_co"]).astype(int)
    df = pd.DataFrame(cols)

    m = ICHDModel(CONFIG_FEATURES["Chest-ECG+SCG"], kind="ridge").fit(df.iloc[:80])
    p_cal = m.predict(df.iloc[80:140])
    p_test = m.predict(df.iloc[140:])
    reg = calibrate_2d_conformal(p_cal, df.iloc[80:140], alpha=0.10)
    ev = reg.evaluate(p_test, df.iloc[140:])
    assert ev["joint_coverage"] >= 0.80
    assert 0.0 <= ev["certified_fraction"] <= ev["certified_margin_fraction"] <= 1.0
    c_score = confidence_score(p_test, df.iloc[140:])
    rcc = risk_coverage_curve(df.iloc[140:]["y_res_dom"].to_numpy(), (p_test.prob_res_dom >= 0.5).astype(int), c_score)
    assert 0.0 <= rcc["aurc"] <= 0.5


def test_conformal_quality_routing_and_weighted_quantile():
    from ichd.model import FEATURES_EAR_PPG

    assert quality_col_for_config("Ear-PPG-Only", FEATURES_EAR_PPG) == "ppge_q"
    assert quality_col_for_config("PPG-Only", CONFIG_FEATURES["PPG-Only"]) == "ppgf_q"
    assert quality_col_for_config("PAT-Only", CONFIG_FEATURES["PAT-Only"]) == "ppgf_q"
    assert quality_col_for_config("Chest-ECG+SCG", CONFIG_FEATURES["Chest-ECG+SCG"]) == "scg_q"
    assert quality_col_for_config("ECG-Only", CONFIG_FEATURES["ECG-Only"]) == "ecg_q"
    assert quality_col_for_config("HR-Only", CONFIG_FEATURES["HR-Only"]) == "ecg_q"
    assert quality_col_for_config("Unknown") is None

    vals = np.array([1.0, 2.0, 10.0])
    wts = np.array([0.5, 0.4, 0.1])
    assert _weighted_quantile(vals, wts, 0.0) == 1.0
    assert _weighted_quantile(vals, wts, 1.0) == 10.0
    assert _weighted_quantile(vals, wts, 0.55) == 2.0


def test_results_json_invariants():
    """Verify canonical results JSON structure and physical closure invariants."""
    import json

    res_path = ROOT / "results" / "ichd_results.json"
    if not res_path.exists():
        pytest.skip("results/ichd_results.json not present")
    with open(res_path) as f:
        r = json.load(f)

    for key in ("exp1_biophysics", "exp2_cv", "exp3_lomo", "exp4_transfer", "exp5_conformal"):
        assert key in r
    assert "ICHD-Full" in r["exp2_cv"]["hybrid"]
    assert r["exp2_cv"]["hybrid"]["ICHD-Full"]["mean_post_proj_gap"] < 1e-12
    assert r["exp2_cv"]["hybrid"]["ICHD-Full"]["r_sv"] > 0.5


def test_build_deltas_and_episodes_causal_baseline_and_closure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Exercise build_deltas_and_episodes end-to-end on a synthetic session to verify
    causal baseline windowing, second-half aggregation, and exact algebraic closure."""
    raw_dir = tmp_path / "raw"
    in_dir = tmp_path / "interim"
    raw_dir.mkdir()
    in_dir.mkdir()

    pd.DataFrame({
        "ID": ["HC_1"],
        "Age": [28],
        "LV ejection fraction": ["62%"],
        "NYHA Class": [np.nan],
    }).to_csv(raw_dir / "demographics.csv", index=False)

    t0 = np.arange(0.0, 200.0, 5.0)
    act = np.where((t0 >= 100.0) & (t0 < 160.0), 3, 0)
    hr = np.where(act == 3, 110.0, 68.0)
    sv = np.where(act == 3, 85.0, 72.0)
    map_v = np.where(act == 3, 102.0, 86.0)
    co = hr * sv / 1000.0

    sess = pd.DataFrame({
        "record": "HC_1_1",
        "subject": "HC_1",
        "cohort": "HC",
        "t0": t0,
        "act": act,
        "act_frac": np.ones_like(t0),
        "n_beats": np.full(len(t0), 10, dtype=int),
        "hr": hr,
        "rr_sd": np.full(len(t0), 25.0),
        "scg_q": np.full(len(t0), 0.8),
        "scg_ao": np.full(len(t0), 95.0),
        "scg_ac": np.full(len(t0), 340.0),
        "scg_lag_sys": np.where(act == 3, -10.0, 0.0),
        "scg_lag_dia": np.where(act == 3, -20.0, 0.0),
        "scg_rms_sys": np.full(len(t0), 0.01),
        "scg_rms_dia": np.full(len(t0), 0.005),
        "scg_rms_pre": np.full(len(t0), 0.001),
        "scg_amp_sys": np.full(len(t0), 0.02),
        "ppgf_pat": np.where(act == 3, 210.0, 250.0),
        "ppge_pat": np.where(act == 3, 170.0, 205.0),
        "ref_map": map_v,
        "ref_sbp": map_v + 25.0,
        "ref_dbp": map_v - 15.0,
        "ref_sv": sv,
        "ref_co": co,
        "ref_hr": hr,
    })
    sess.to_csv(in_dir / "HC_1_1.csv.gz", index=False, compression="gzip")

    monkeypatch.setattr(E, "RAW", raw_dir)
    win_df, ep_df = E.build_deltas_and_episodes(in_dir=in_dir)

    assert len(ep_df) == 1
    ep_row = ep_df.iloc[0]
    # Exact algebraic closure in episode table
    assert abs(ep_row["dlog_map"] - (ep_row["dlog_co"] + ep_row["dlog_svr"])) < 1e-14
    assert abs(ep_row["d_ppgf_ptt"] - (ep_row["d_ppgf_pat"] - ep_row["d_pep_lag"])) < 1e-14
    assert abs(ep_row["d_ppge_ptt"] - (ep_row["d_ppge_pat"] - ep_row["d_pep_lag"])) < 1e-14
    assert bool(ep_row["is_causal_baseline"]) is True
    # Baseline came strictly from pre-onset rest (hr=68), not steady-state exercise (hr=110)
    assert ep_row["base_hr"] == pytest.approx(68.0)
    assert ep_row["d_hr_bpm"] == pytest.approx(42.0)
    # Active windows start strictly at or after onset (t0 >= 100)
    active_wins = win_df[win_df["act"] == 3]
    assert (active_wins["t0"] >= 100.0).all()


def test_gauss_markov_projection_algebraic_closure():
    """Verify that 2x2 Gauss-Markov projection guarantees exact algebraic closure (<10^-14) for arbitrary inputs."""
    rng = np.random.default_rng(2026)
    n_samples = 100
    z_sv = rng.normal(0, 0.2, n_samples)
    z_svr = rng.normal(0, 0.2, n_samples)
    z_map = rng.normal(0, 0.3, n_samples)
    z_diff = rng.normal(0, 0.3, n_samples)
    h = rng.normal(0.05, 0.1, n_samples)
    var_sv, var_svr, var_diff, var_map = 0.04, 0.04, 0.05, 0.08

    u_sv, u_co, u_svr, u_map = reconcile_manifold_2x2(h, z_sv, z_svr, z_diff, z_map, var_sv, var_svr, var_diff, var_map)

    assert np.max(np.abs(u_co - (h + u_sv))) < 1e-14
    assert np.max(np.abs(u_map - (u_co + u_svr))) < 1e-14
    assert np.max(np.abs(u_map - (h + u_sv + u_svr))) < 1e-14


def test_subsystem_feature_routing_and_optical_sets():
    """Verify HR is withheld from f_card/f_joint, exact subsystem partition holds,
    vascular PPG features route to f_vasc, and optical sets exclude ECG/SCG."""
    from ichd.model import (
        FEATURES_EAR_PPG, FEATURES_CHEST_EAR_PPG,
        FEATURES_EAR_OPTICAL_ONLY, FEATURES_FINGER_OPTICAL_ONLY,
        FEATURES_EAR_NO_AMP, FEATURES_FINGER_NO_AMP,
    )

    assert len(CONFIG_FEATURES["HR-Only"]) == 2
    assert len(CONFIG_FEATURES["PAT-Only"]) == 6
    assert len(CONFIG_FEATURES["ECG-Only"]) == 8
    assert len(CONFIG_FEATURES["Chest-ECG+SCG"]) == 26
    assert len(CONFIG_FEATURES["PPG-Only"]) == 50
    assert len(FEATURES_EAR_PPG) == 50
    assert len(FEATURES_CHEST_EAR_PPG) == 72
    assert len(CONFIG_FEATURES["ICHD-Full"]) == 122

    feats = CONFIG_FEATURES["ICHD-Full"]
    f_card, f_vasc, f_joint = _split_subsystem_cols(feats, feats)
    for hr_col in ("dlog_hr", "d_hr_bpm"):
        assert hr_col not in f_card
        assert hr_col not in f_joint
    assert len(f_card) + len(f_vasc) == len(feats)
    assert len(f_joint) == len(feats) - 2
    for c in ("d_ppge_notch_amp", "dlog_ppge_notch_amp", "d_ppge_dia_slope",
              "d_ppgf_notch_amp", "dlog_ppgf_notch_amp", "d_ppgf_dia_slope"):
        assert c in f_vasc
        assert c not in f_card

    assert len(FEATURES_EAR_OPTICAL_ONLY) == 42
    assert len(FEATURES_FINGER_OPTICAL_ONLY) == 42
    assert "dlog_hr" not in FEATURES_EAR_OPTICAL_ONLY
    assert not any("ecg" in c or "scg" in c or "ptt" in c for c in FEATURES_EAR_OPTICAL_ONLY)
    assert not any("ecg" in c or "scg" in c or "ptt" in c for c in FEATURES_FINGER_OPTICAL_ONLY)

    assert len(FEATURES_EAR_NO_AMP) == 44
    assert len(FEATURES_FINGER_NO_AMP) == 44
    for raw_tag in ("_ppge_amp", "_ppgf_amp", "_ppge_slope", "_ppgf_slope", "_ppge_area_raw", "_ppgf_area_raw"):
        assert not any(c.endswith(raw_tag) for c in FEATURES_EAR_NO_AMP)
        assert not any(c.endswith(raw_tag) for c in FEATURES_FINGER_NO_AMP)
    assert "d_ppge_notch_amp" in FEATURES_EAR_NO_AMP
    assert "d_ppgf_notch_amp" in FEATURES_FINGER_NO_AMP

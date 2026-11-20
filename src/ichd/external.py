"""Shared VitalDB external validation utilities (caching, shifting, windowing, and bootstrap)."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal
from scipy.stats import pearsonr

from . import signals as S
from .model import _make_reg, FEATURES_PPG_FINGER, FEATURES_HR_ONLY

ROOT = Path(__file__).resolve().parents[2]
VITALDB_CACHE_DIR = ROOT / "data" / "interim" / "vitaldb_cache"
TABLES_CACHE = ROOT / "data" / "interim" / "ichd_tables"

# Pre-registered 36 consecutively indexed qualifying VitalDB cases
SELECTED_CASES = [
    17, 125, 137, 138, 164, 448, 544, 631, 779, 781,
    787, 843, 931, 937, 958, 1034, 1078, 1166, 1172, 1194,
    1213, 1323, 1325, 1523, 1642, 1652, 1802, 1852, 1891, 1897,
    1912, 1938, 1973, 1988, 2000, 2010,
]

# Fixed hardware buffer on GE Solar 8000 / Tram-Rac analog pleth output
GE_PLETH_HARDWARE_DELAY_MS = 380


def fetch_vitaldb_signals(
    caseid: int,
    max_duration_min: float = 120.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Download and resample (1000 Hz) ECG, Pleth, and FloTrac SV for one VitalDB case."""
    import vitaldb  # optional dependency

    try:
        vf = vitaldb.VitalFile(caseid, ["SNUADC/ECG_II", "SNUADC/PLETH", "EV1000/SV"])
        max_samples_500 = int(max_duration_min * 60 * 500)
        arr_signals = vf.to_numpy(["SNUADC/ECG_II", "SNUADC/PLETH"], 1 / 500)[:max_samples_500]
        arr_sv = vf.to_numpy("EV1000/SV", 1.0)[:int(max_duration_min * 60)]
    except Exception as e:
        print(f"Case {caseid}: failed to load ({e})")
        return None

    if len(arr_signals) < 30 * 60 * 500:
        return None

    ecg_500 = np.nan_to_num(arr_signals[:, 0], nan=0.0)
    pleth_500 = np.nan_to_num(arr_signals[:, 1], nan=0.0)
    ecg = scipy.signal.resample_poly(ecg_500, 2, 1).astype(np.float32)
    pleth = scipy.signal.resample_poly(pleth_500, 2, 1).astype(np.float32)
    arr_sv = np.nan_to_num(arr_sv, nan=-1.0).astype(np.float32)
    return ecg, pleth, arr_sv


def cache_case_raw(caseid: int, max_duration_min: float = 120.0) -> bool:
    """Download and cache raw 1000 Hz resampled signals for a VitalDB case."""
    VITALDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_file = VITALDB_CACHE_DIR / f"case_{caseid:04d}.npz"
    if out_file.exists():
        return True
    fetched = fetch_vitaldb_signals(caseid, max_duration_min=max_duration_min)
    if fetched is None:
        return False
    ecg, pleth, arr_sv = fetched
    np.savez_compressed(out_file, ecg=ecg, pleth=pleth, sv=arr_sv)
    return True


def load_or_fetch_case(
    caseid: int,
    max_duration_min: float = 120.0,
    allow_fetch: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load cached VitalDB `.npz` signals or optionally fetch from VitalDB API."""
    cache_file = VITALDB_CACHE_DIR / f"case_{caseid:04d}.npz"
    if cache_file.exists():
        d = np.load(cache_file)
        return d["ecg"], d["pleth"], d["sv"]
    if not allow_fetch:
        return None
    return fetch_vitaldb_signals(caseid, max_duration_min=max_duration_min)


def non_circular_shift(arr: np.ndarray, shift_ms: int) -> np.ndarray:
    """Non-circular shift of plethysmogram (clamped boundary padding, no wrap-around)."""
    shifted = np.empty_like(arr)
    if shift_ms > 0:
        shifted[:-shift_ms] = arr[shift_ms:]
        shifted[-shift_ms:] = arr[-1]
    elif shift_ms < 0:
        s = -shift_ms
        shifted[s:] = arr[:-s]
        shifted[:s] = arr[0]
    else:
        shifted[:] = arr
    return shifted


def train_frozen_auricle_models(episodes_path: Path | None = None) -> dict[str, tuple[list[str], object]]:
    """Train and freeze Ridge models on the complete internal Auricle development dataset."""
    ep_path = episodes_path or (TABLES_CACHE / "episodes.pkl")
    df_auricle = pd.read_pickle(ep_path)
    df_auricle = df_auricle[np.isfinite(df_auricle["dlog_sv"])].copy()
    y_sv = df_auricle["dlog_sv"].to_numpy(dtype=float)

    cols_finger = [c for c in FEATURES_PPG_FINGER if c in df_auricle.columns]
    cols_hr = [c for c in FEATURES_HR_ONLY if c in df_auricle.columns]
    cols_pat = [c for c in ["dlog_hr", "d_hr_bpm", "d_ppgf_pat", "dlog_ppgf_pat"] if c in df_auricle.columns]
    cols_optical = [c for c in cols_finger if "ppgf" in c]

    return {
        "PPG_Finger": (cols_finger, _make_reg("ridge", 42).fit(df_auricle[cols_finger].to_numpy(dtype=float), y_sv)),
        "HR_Only": (cols_hr, _make_reg("ridge", 42).fit(df_auricle[cols_hr].to_numpy(dtype=float), y_sv)),
        "PAT_Finger": (cols_pat, _make_reg("ridge", 42).fit(df_auricle[cols_pat].to_numpy(dtype=float), y_sv)),
        "Optical_Only": (cols_optical, _make_reg("ridge", 42).fit(df_auricle[cols_optical].to_numpy(dtype=float), y_sv)),
    }


def extract_vitaldb_case_windows(
    caseid: int,
    shift_ms: int = GE_PLETH_HARDWARE_DELAY_MS,
    max_duration_min: float = 120.0,
    allow_fetch: bool = True,
) -> pd.DataFrame:
    """Extract baseline-referenced 10s evaluation windows for one VitalDB case."""
    loaded = load_or_fetch_case(caseid, max_duration_min=max_duration_min, allow_fetch=allow_fetch)
    if loaded is None:
        return pd.DataFrame()
    ecg, pleth, arr_sv = loaded

    pleth_shifted = non_circular_shift(pleth, shift_ms)
    pleth_f = S.bandpass(pleth_shifted, 0.45, 16.0, fs=1000)

    r = S.detect_r_peaks(ecg, fs=1000)
    ok = S.clean_rr_mask(r)

    win_len = 10000
    hop = 5000
    n_wins = (len(ecg) - win_len) // hop
    rows = []

    for wi in range(n_wins):
        w_t0 = wi * hop
        w_t1 = w_t0 + win_len
        t_s0 = w_t0 // 1000
        t_s1 = w_t1 // 1000

        if t_s1 > len(arr_sv):
            break
        sv_chunk = arr_sv[t_s0:t_s1]
        sv_chunk = sv_chunk[(~np.isnan(sv_chunk)) & (sv_chunk > 0)]
        if len(sv_chunk) == 0:
            continue
        sv_val = float(np.median(sv_chunk))
        if sv_val < 15.0 or sv_val > 200.0:
            continue

        rb = r[ok & (r >= w_t0) & (r < w_t1)]
        if len(rb) < 5:
            continue
        rr = np.diff(rb)
        rr = rr[(rr > 300) & (rr < 2000)]
        if len(rr) < 3:
            continue
        rr_med = float(np.median(rr))
        hr = 60000.0 / rr_med
        rr_sd = float(np.std(rr))

        ens, q, nb = S.ensemble(pleth_f, rb, pre=100, post=750, clip=None, detrend_linear=True)
        if ens is None or q < 0.60:
            continue
        feats = S.extract_ppg_contour(ens, pre=100, search=(100, 500), fs=1000)
        if "pat" not in feats:
            continue

        row = {
            "caseid": caseid,
            "wi": wi,
            "t0_s": t_s0,
            "ref_sv": sv_val,
            "hr": hr,
            "rr_sd": rr_sd,
            "q": q,
            "nb": nb,
        }
        for k, v in feats.items():
            row[f"ppgf_{k}"] = v
        rows.append(row)

    df_raw = pd.DataFrame(rows)
    if len(df_raw) < 30:
        return pd.DataFrame()

    base = df_raw.iloc[:15].median()
    if base["ref_sv"] <= 0 or base["hr"] <= 0:
        return pd.DataFrame()

    df_eval = df_raw.iloc[15:].copy()
    df_eval["dlog_sv"] = np.log(df_eval["ref_sv"] / base["ref_sv"])
    df_eval["dlog_hr"] = np.clip(np.log(df_eval["hr"] / base["hr"]), -2.0, 2.0)
    df_eval["d_hr_bpm"] = df_eval["hr"] - base["hr"]
    df_eval["d_rr_sd"] = df_eval["rr_sd"] - base["rr_sd"]
    df_eval["dlog_rr_sd"] = np.clip(np.log(df_eval["rr_sd"] / (base["rr_sd"] + 1e-4)), -2.0, 2.0)

    for col in FEATURES_PPG_FINGER:
        if col in ["dlog_hr", "d_hr_bpm", "d_rr_sd", "dlog_rr_sd"]:
            continue
        if col.startswith("dlog_ppgf_"):
            raw_k = col[5:]
            if raw_k in base and base[raw_k] > 1e-6 and raw_k in df_eval:
                df_eval[col] = np.clip(np.log(df_eval[raw_k] / base[raw_k]), -2.0, 2.0)
            else:
                df_eval[col] = np.nan
        elif col.startswith("d_ppgf_"):
            raw_k = col[2:]
            if raw_k in base and raw_k in df_eval:
                df_eval[col] = df_eval[raw_k] - base[raw_k]
            else:
                df_eval[col] = np.nan
        else:
            df_eval[col] = np.nan

    return df_eval


def cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray | None,
    clusters: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Patient-clustered bootstrap 95% CI on Pearson r and paired difference."""
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(clusters)
    n_clust = len(unique_clusters)
    cluster_indices = {c: np.flatnonzero(clusters == c) for c in unique_clusters}

    r_a_boots = []
    r_b_boots = []
    delta_boots = []

    r_a_obs = float(pearsonr(y_true, y_pred_a)[0])
    r_b_obs = float(pearsonr(y_true, y_pred_b)[0]) if y_pred_b is not None else np.nan
    delta_obs = r_a_obs - r_b_obs if y_pred_b is not None else np.nan

    for _ in range(n_boot):
        sample_clusters = rng.choice(unique_clusters, size=n_clust, replace=True)
        idx = np.concatenate([cluster_indices[c] for c in sample_clusters])
        y_b = y_true[idx]
        ya_b = y_pred_a[idx]

        if np.std(y_b) < 1e-6 or np.std(ya_b) < 1e-6:
            continue
        r_a_val = float(pearsonr(y_b, ya_b)[0])
        r_a_boots.append(r_a_val)

        if y_pred_b is not None:
            yb_b = y_pred_b[idx]
            if np.std(yb_b) < 1e-6:
                continue
            r_b_val = float(pearsonr(y_b, yb_b)[0])
            r_b_boots.append(r_b_val)
            delta_boots.append(r_a_val - r_b_val)

    out = {
        "r_obs": r_a_obs,
        "ci_low": float(np.percentile(r_a_boots, 2.5)) if r_a_boots else r_a_obs,
        "ci_high": float(np.percentile(r_a_boots, 97.5)) if r_a_boots else r_a_obs,
    }
    if delta_boots:
        d_arr = np.asarray(delta_boots, dtype=float)
        k_min = int(min(np.sum(d_arr <= 0), np.sum(d_arr >= 0)))
        p_val = float(min(1.0, 2.0 * (k_min + 1.0) / (len(d_arr) + 1.0)))
        out.update({
            "delta_obs": delta_obs,
            "delta_ci_low": float(np.percentile(delta_boots, 2.5)),
            "delta_ci_high": float(np.percentile(delta_boots, 97.5)),
            "delta_p_val": p_val,
        })
    return out

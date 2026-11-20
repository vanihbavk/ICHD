"""Low-level 1 kHz DSP primitives: filtering, QRS detection, R-locked ensemble averaging, and PPG/PCG features."""
from __future__ import annotations

import numpy as np
from scipy import signal

FS = 1000


def bandpass(x: np.ndarray, lo: float, hi: float, fs: float = FS, order: int = 3) -> np.ndarray:
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


def lowpass(x: np.ndarray, hi: float, fs: float = FS, order: int = 3) -> np.ndarray:
    sos = signal.butter(order, hi, btype="lowpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)


# ECG QRS detection and R-R interval screening
def detect_r_peaks(ecg: np.ndarray, fs: float = FS) -> np.ndarray:
    """Energy-envelope QRS detector with block-adaptive threshold, prominence
    guard, and polarity-aware refinement onto the R apex."""
    ecg = np.nan_to_num(np.asarray(ecg, dtype=float))
    f = bandpass(ecg, 8.0, 25.0, fs)
    e = np.gradient(f) ** 2
    w = max(1, int(0.10 * fs))
    integ = np.convolve(e, np.ones(w) / w, mode="same")
    blk = int(5 * fs)
    thr = np.empty_like(integ)
    for i in range(0, len(integ), blk):
        seg = integ[max(0, i - blk): i + 2 * blk]
        thr[i: i + blk] = max(0.32 * np.percentile(seg, 98.5), 2.5 * np.median(seg)) if len(seg) else np.inf
    pk, _ = signal.find_peaks(integ, height=thr, distance=int(0.32 * fs), prominence=0.35 * thr)
    if len(pk) < 3:
        return np.array([], dtype=np.int64)
    hp = bandpass(ecg, 5.0, 40.0, fs)
    half = int(0.06 * fs)
    apex = np.array([max(0, p - half) + int(np.argmax(np.abs(hp[max(0, p - half): p + half]))) for p in pk])
    pol = np.sign(np.median(hp[apex])) or 1.0
    r = np.array([max(0, p - half) + int(np.argmax(pol * hp[max(0, p - half): p + half])) for p in pk])
    r = np.unique(r)
    keep = np.r_[True, np.diff(r) >= int(0.32 * fs)]
    return r[keep].astype(np.int64)


def clean_rr_mask(r: np.ndarray, fs: float = FS) -> np.ndarray:
    """Boolean mask over beats (len(r)) marking beats whose preceding RR is
    physiologic (300-2000 ms) and within 25 % of the local median."""
    rr = np.diff(r) / fs
    ok = np.zeros(len(r), dtype=bool)
    if len(rr) < 3:
        return ok
    med = signal.medfilt(rr, 7)
    good = (rr > 0.3) & (rr < 2.0) & (np.abs(rr - med) < 0.25 * med)
    ok[1:] = good
    return ok


# R-locked ensemble averaging and fiducial tracking
def ensemble(x: np.ndarray, r: np.ndarray, pre: int, post: int, reject_k: float = 4.0,
             clip: float | None = None, detrend_linear: bool = False) -> tuple[np.ndarray | None, float, int]:
    """R-aligned robust ensemble average.

    Returns (ensemble, beat consistency q in [-1,1], n beats used).
    Beats with excessive peak amplitude (motion) or clipping are rejected.
    """
    n = len(x)
    segs = [x[i - pre: i + post] for i in r if i - pre >= 0 and i + post <= n]
    if len(segs) < 3:
        return None, 0.0, 0
    S = np.asarray(segs, dtype=float).copy()
    if S.ndim == 3:  # multi-axis (beats, time, axes)
        amp = np.abs(S - S.mean(axis=1, keepdims=True)).max(axis=(1, 2))
    else:
        amp = np.abs(S - S.mean(axis=1, keepdims=True)).max(axis=1)
    good = amp < reject_k * np.median(amp)
    if clip is not None:
        good &= ~(np.abs(S) >= clip).reshape(len(S), -1).any(axis=1)
    S = S[good]
    if len(S) < 3:
        return None, 0.0, int(len(S))
    if detrend_linear and S.ndim == 2:
        k0 = min(30, pre)
        k1 = min(40, post)
        y0 = np.median(S[:, :k0], axis=1, keepdims=True)
        y1 = np.median(S[:, -k1:], axis=1, keepdims=True)
        t_ramp = np.linspace(0.0, 1.0, S.shape[1])[None, :]
        S = S - (y0 + (y1 - y0) * t_ramp)
    ens = np.median(S, axis=0) if len(S) >= 5 else S.mean(axis=0)
    flat = S.reshape(len(S), -1)
    ef = ens.reshape(-1)
    fc = flat - flat.mean(axis=1, keepdims=True)
    ec = ef - ef.mean()
    den = np.linalg.norm(fc, axis=1) * np.linalg.norm(ec) + 1e-12
    q = float(np.median(fc @ ec / den))
    return ens, q, int(len(S))


def xcorr_lag(seg: np.ndarray, tmpl: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Lag (samples) maximising normalised correlation of `seg` against
    `tmpl`; both are the same length; positive lag = seg is delayed."""
    best, bl = -2.0, 0
    L = len(tmpl)
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = seg[lag:], tmpl[: L - lag]
        else:
            a, b = seg[: L + lag], tmpl[-lag:]
        a = a - a.mean()
        b = b - b.mean()
        c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        if c > best:
            best, bl = c, lag
    return bl, best


def tracked_peak(s: np.ndarray, center: int, half: int, sign: float = 1.0) -> int:
    lo, hi = max(0, center - half), min(len(s), center + half + 1)
    return lo + int(np.argmax(sign * s[lo:hi]))


def extract_ppg_contour(ens: np.ndarray | None, pre: int, search: tuple[int, int],
                        fs: float = FS) -> dict[str, float]:
    """Extract Windkessel, dicrotic-notch, Bramwell-Hill, and APG 2nd-derivative features."""
    out: dict[str, float] = {}
    if ens is None or len(ens) < pre + search[1]:
        return out
    p_s = lowpass(ens, 15.0, fs=fs, order=3)
    dp = np.gradient(p_s) * fs  # units / s
    dp_samp = np.gradient(p_s)
    a, b = pre + search[0], min(len(p_s) - 2, pre + search[1])
    if b - a < 20:
        return out
    up_idx = a + int(np.argmax(dp_samp[a:b]))
    slope_samp = float(dp_samp[up_idx])
    if slope_samp <= 1e-9:
        return out
    foot_lo = max(pre + 15, up_idx - int(0.13 * fs))
    foot_min_idx = foot_lo + int(np.argmin(p_s[foot_lo: up_idx + 1]))
    t_tan = float(np.clip(
        up_idx - (p_s[up_idx] - p_s[foot_min_idx]) / slope_samp,
        pre + max(20.0, search[0] - 35.0),
        float(b),
    ))
    pat_ms = (t_tan - pre) * 1000.0 / fs
    pk_hi = min(len(p_s), up_idx + int(0.25 * fs))
    pk_idx = up_idx + int(np.argmax(p_s[up_idx:pk_hi]))
    p_base = float(p_s[foot_min_idx])
    amp = float(p_s[pk_idx] - p_base)
    if amp <= 1e-6 or pat_ms <= 20.0:
        return out

    pn = (p_s - p_base) / amp
    dp_n = np.gradient(pn) * fs       # 1 / s
    ddp_n = np.gradient(dp_n) * fs    # 1 / s^2

    out["pat"] = float(pat_ms)
    out["inv_pat2"] = float(1e6 / (pat_ms ** 2))
    out["amp"] = amp
    out["slope"] = float(dp[up_idx])
    out["max_dp_norm"] = float(dp_n[up_idx])
    out["rise"] = float((pk_idx - t_tan) * 1000.0 / fs)

    # Pulse widths at 25/50/75 % systolic peak amplitude
    seg = pn[foot_min_idx: min(len(pn), foot_min_idx + int(0.55 * fs))]
    pk_loc = pk_idx - foot_min_idx
    if pk_loc >= 0 and pk_loc < len(seg):
        for lvl in (25, 50, 75):
            thr = lvl / 100.0
            l_idx = np.flatnonzero(seg[:pk_loc + 1] >= thr)
            r_idx = np.flatnonzero(seg[pk_loc:] <= thr)
            if len(l_idx) and len(r_idx):
                out[f"pw{lvl}"] = float(((pk_loc + r_idx[0]) - l_idx[0]) * 1000.0 / fs)
        r50 = np.flatnonzero(seg[pk_loc:] <= 0.5)
        if len(r50):
            out["hw"] = float(pk_loc + r50[0]) * 1000.0 / fs

    # Dicrotic notch & Windkessel diastolic decay
    n_lo = min(len(pn) - 20, pk_idx + int(0.04 * fs))
    n_hi = min(len(pn) - 5, pk_idx + int(0.26 * fs))
    if n_hi > n_lo + 15:
        notch_idx = n_lo + int(np.argmax(ddp_n[n_lo:n_hi]))
        dia_lo = min(len(pn) - 5, notch_idx + int(0.01 * fs))
        dia_hi = min(len(pn) - 1, notch_idx + int(0.15 * fs))
        dia_pk_idx = dia_lo + int(np.argmax(pn[dia_lo:dia_hi])) if dia_hi > dia_lo else notch_idx
        out["t_notch"] = float((notch_idx - t_tan) * 1000.0 / fs)
        out["notch_amp"] = float(pn[notch_idx])
        out["ri_dia"] = float(pn[dia_pk_idx])
        out["t_sys_dia"] = float((dia_pk_idx - pk_idx) * 1000.0 / fs)
        end_idx = min(len(pn), foot_min_idx + int(0.52 * fs))
        a1 = float(np.trapezoid(np.maximum(0.0, pn[foot_min_idx:notch_idx]))) / fs
        a2 = float(np.trapezoid(np.maximum(0.0, pn[notch_idx:end_idx]))) / fs
        if a1 > 1e-6:
            out["ipa"] = a2 / a1
            out["area_sys_norm"] = a1
            out["area_raw"] = float(amp * (a1 + a2))
        d_start = min(len(pn) - 30, dia_pk_idx + int(0.015 * fs))
        d_end = min(len(pn), foot_min_idx + int(0.50 * fs))
        if d_end > d_start + int(0.04 * fs):
            y_d = pn[d_start:d_end]
            m_pos = y_d > 0.05
            if m_pos.sum() > int(0.025 * fs):
                t_d = np.arange(len(y_d))[m_pos] / fs
                out["dia_decay"] = float(-np.polyfit(t_d, np.log(y_d[m_pos]), 1)[0])
                out["dia_slope"] = float(np.polyfit(t_d, y_d[m_pos], 1)[0])

    # Second-derivative APG arterial aging/tone waves (a, b, d, e)
    a_lo = max(pre + 15, foot_min_idx - int(0.01 * fs))
    a_hi = min(len(ddp_n) - 1, up_idx + int(0.02 * fs))
    if a_hi > a_lo + 5:
        a_idx = a_lo + int(np.argmax(ddp_n[a_lo:a_hi]))
        a_val = float(ddp_n[a_idx])
        if a_val > 1e-6:
            b_hi = min(len(ddp_n) - 1, a_idx + int(0.09 * fs))
            b_idx = a_idx + int(np.argmin(ddp_n[a_idx:b_hi]))
            apg_ba = float(ddp_n[b_idx] / a_val)
            out["apg_ba"] = apg_ba
            d_lo = min(len(ddp_n) - 10, pk_idx)
            d_hi = min(len(ddp_n) - 1, pk_idx + int(0.14 * fs))
            if d_hi > d_lo + 10:
                apg_da = float(np.min(ddp_n[d_lo:d_hi]) / a_val)
                apg_ea = float(np.max(ddp_n[d_lo:d_hi]) / a_val)
                out["apg_da"] = apg_da
                out["apg_ea"] = apg_ea
                out["apg_agi"] = apg_ba - apg_da - apg_ea
    return out


def pcg_envelope(pcg: np.ndarray, fs: float = FS, lo: float = 25.0, hi: float = 140.0) -> np.ndarray:
    f = bandpass(pcg, lo, hi, fs, order=4)
    env = np.abs(signal.hilbert(f))
    w = max(1, int(0.02 * fs))
    return np.convolve(env, np.ones(w) / w, mode="same")

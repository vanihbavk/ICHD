"""Window-level feature and reference extraction for the Auricle dataset.

For every session we produce a table with one row per 10 s window (5 s hop):
  * chest-patch features (ECG timing, SCG electromechanical fiducials),
  * peripheral features (finger / ear PPG pulse arrival and amplitude),
  * phonocardiogram fiducials (used only as an independent timing check),
  * CNAP references (MAP, SBP, DBP, SV, CO, SVR, HR) and activity label.

Usage:  python -m ichd.extract  [--workers 3]
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from . import signals as S

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "auricle"
OUT = ROOT / "data" / "interim" / "ichd"

FS = 1000
WIN_S, HOP_S = 10.0, 5.0
PRE_SCG, POST_SCG = 100, 600
PRE_PPG, POST_PPG = 100, 750

CHANNELS = ["ECG", "Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z",
            "Digital_Stethoscope", "PPG_Finger", "PPG_Ear",
            "Mean_BP", "Systolic_BP", "Diastolic_BP", "Stroke_Volume",
            "Cardiac_Output", "Heart_Rate", "Activity_Annotations"]


def list_records() -> list[str]:
    return [ln.strip().split("/")[0] for ln in (RAW / "RECORDS").read_text().splitlines() if ln.strip()]


def load_record(rec: str) -> dict[str, np.ndarray]:
    hdr = wfdb.rdheader(str(RAW / "data" / rec / rec))
    names = [c for c in CHANNELS if c in hdr.sig_name]
    r = wfdb.rdrecord(str(RAW / "data" / rec / rec), channel_names=names)
    assert r.fs == FS
    d = {n: r.p_signal[:, i].astype(np.float64) for i, n in enumerate(r.sig_name)}
    if "Activity_Annotations" in d:
        d["Activity_Annotations"] = np.round(d["Activity_Annotations"]).astype(int)
    else:
        d["Activity_Annotations"] = np.zeros(len(d["ECG"]), dtype=int)
    return d


def _ref_median(x: np.ndarray, lo: float, hi: float) -> float:
    v = x[(x > lo) & (x < hi)]
    return float(np.median(v)) if len(v) > 0.5 * len(x) else np.nan


def _first_maneuver_onset(act: np.ndarray) -> int:
    idx = np.flatnonzero(np.isin(act, [1, 2, 3, 4]))
    return int(idx[0]) if len(idx) else len(act)


def build_templates(d: dict, r: np.ndarray, ok: np.ndarray) -> dict:
    """Session calibration from the 150 s of rest preceding the first maneuver."""
    act = d["Activity_Annotations"]
    onset = _first_maneuver_onset(act)
    lo = max(0, onset - 150 * FS)
    sel = r[ok & (r >= lo) & (r < onset) & np.isin(act[np.clip(r, 0, len(act) - 1)], [0, 9])]
    if len(sel) < 40:
        sel = r[ok & np.isin(act[np.clip(r, 0, len(act) - 1)], [0, 9])][:200]
    acc = np.stack([d["Accelerometer_X"], d["Accelerometer_Y"], d["Accelerometer_Z"]], axis=1)
    ens3, q3, _ = S.ensemble(acc, sel, PRE_SCG, POST_SCG)
    rr_med = float(np.median(np.diff(sel))) if len(sel) > 2 else 900.0
    T: dict = {"rr_med": rr_med, "n_beats": int(len(sel))}
    if ens3 is None:
        return T
    seg = ens3[PRE_SCG: PRE_SCG + 600] - ens3[PRE_SCG: PRE_SCG + 600].mean(axis=0)
    _, _, vt = np.linalg.svd(seg, full_matrices=False)
    w = vt[0]
    s = ens3 @ w
    sys = slice(PRE_SCG + 20, PRE_SCG + 200)
    if np.max(s[sys]) < -np.min(s[sys]):
        w, s = -w, -s
    ao = PRE_SCG + 20 + int(np.argmax(s[sys]))
    dia_hi = PRE_SCG + int(min(560, 0.8 * rr_med))
    dlo = PRE_SCG + 260
    ac_rel = int(np.argmax(np.abs(s[dlo:dia_hi]))) if dia_hi > dlo + 20 else 100
    ac = dlo + ac_rel
    T.update(w=w, scg=s, ao=ao, ac=ac, ac_sign=float(np.sign(s[ac]) or 1.0), scg_q=q3)

    # Resting SVD axis for 5-40 Hz motion-rejected SCG
    if "_acc_5_40" in d:
        ens5, _, _ = S.ensemble(d["_acc_5_40"], sel, PRE_SCG, POST_SCG)
        if ens5 is not None:
            seg5 = ens5[PRE_SCG: PRE_SCG + 550] - ens5[PRE_SCG: PRE_SCG + 550].mean(axis=0)
            _, _, vt5 = np.linalg.svd(seg5, full_matrices=False)
            w5 = vt5[0]
            s5 = ens5 @ w5
            if np.max(s5[sys]) < -np.min(s5[sys]):
                w5 = -w5
            T["w5"] = w5

    # PCG template
    env = d.get("_pcg_env")
    if env is not None:
        e_ens, _, _ = S.ensemble(env, sel, PRE_SCG, POST_SCG)
        if e_ens is not None:
            s1 = PRE_SCG + int(np.argmax(e_ens[PRE_SCG: PRE_SCG + 150]))
            s2 = dlo + int(np.argmax(e_ens[dlo:dia_hi])) if dia_hi > dlo + 20 else np.nan
            T.update(pcg=e_ens, s1=s1, s2=s2)
    # ECG T-wave template location
    e_ens, _, _ = S.ensemble(d["_ecg_f"], sel, PRE_SCG, POST_SCG)
    if e_ens is not None:
        tw = slice(PRE_SCG + 150, PRE_SCG + int(min(480, 0.7 * rr_med)))
        seg = e_ens[tw] - np.median(e_ens[PRE_SCG - 100: PRE_SCG - 40])
        T["t_sign"] = float(np.sign(seg[np.argmax(np.abs(seg))]) or 1.0)
        T["rt"] = PRE_SCG + 150 + int(np.argmax(T["t_sign"] * seg))
    return T


def window_features(d: dict, r: np.ndarray, ok: np.ndarray, T: dict) -> pd.DataFrame:
    n = len(d["ECG"])
    act = d["Activity_Annotations"]
    acc = np.stack([d["Accelerometer_X"], d["Accelerometer_Y"], d["Accelerometer_Z"]], axis=1)
    scg_proj = acc @ T["w"] if "w" in T else None
    acc_5_40 = d.get("_acc_5_40")
    acc_12_45 = d.get("_acc_12_45")
    w_rest5 = T.get("w5")

    rows = []
    win, hop = int(WIN_S * FS), int(HOP_S * FS)
    for t0 in range(0, n - win - POST_PPG, hop):
        t1 = t0 + win
        m = (r >= t0) & (r < t1)
        rb = r[m & ok]
        a = act[t0:t1]
        vals, cnts = np.unique(a, return_counts=True)
        row = {"t0": t0 / FS, "act": int(vals[np.argmax(cnts)]), "act_frac": float(cnts.max() / len(a)),
               "n_beats": int(len(rb)), "n_all": int(m.sum())}
        # CNAP references
        row["ref_map"] = _ref_median(d["Mean_BP"][t0:t1], 30, 200)
        row["ref_sbp"] = _ref_median(d["Systolic_BP"][t0:t1], 50, 260)
        row["ref_dbp"] = _ref_median(d["Diastolic_BP"][t0:t1], 20, 160)
        row["ref_sv"] = _ref_median(d["Stroke_Volume"][t0:t1], 10, 149.5)
        row["ref_co"] = _ref_median(d["Cardiac_Output"][t0:t1], 0.5, 19.9)
        row["ref_hr"] = _ref_median(d["Heart_Rate"][t0:t1], 30, 220)
        if len(rb) < 5:
            rows.append(row)
            continue
        rr = np.diff(r[(r >= t0 - 2 * FS) & (r < t1)])
        rr = rr[(rr > 300) & (rr < 2000)]
        if len(rr) < 3:
            rows.append(row)
            continue
        rr_med_w = float(np.median(rr))
        row["hr"] = 60000.0 / rr_med_w
        row["rr_sd"] = float(np.std(rr))

        # ECG morphology & electrical systole
        e_ens, eq, _ = S.ensemble(d["_ecg_f"], rb, PRE_SCG, POST_SCG)
        if e_ens is not None and "rt" in T:
            base = np.median(e_ens[PRE_SCG - 90: PRE_SCG - 40])
            en = e_ens - base
            r_amp = float(abs(en[PRE_SCG]))
            rt_rel = float(S.tracked_peak(en, T["rt"], 80, T["t_sign"]) - PRE_SCG)
            t_seg = en[PRE_SCG + 120: PRE_SCG + int(min(500, 0.72 * rr_med_w))]
            t_amp = float(np.max(np.abs(t_seg))) if len(t_seg) > 20 else np.nan
            row["ecg_rt"] = rt_rel
            row["ecg_t_over_r"] = float(t_amp / (r_amp + 1e-6)) if np.isfinite(t_amp) else np.nan
            row["ecg_q"] = eq

        # 1-40 Hz Fixed-Axis SCG
        if scg_proj is not None:
            s, q, nb = S.ensemble(scg_proj, rb, PRE_SCG, POST_SCG)
            if s is not None:
                tm = T["scg"]
                row["scg_q"] = q
                row["scg_nb"] = nb
                sys_sl = slice(PRE_SCG - 20, PRE_SCG + 220)
                lag_s, _ = S.xcorr_lag(s[sys_sl], tm[sys_sl], 40)
                row["scg_lag_sys"] = lag_s
                ac = T["ac"]
                dsl = slice(max(0, ac - 110), min(len(s), ac + 110))
                lag_d, c_d = S.xcorr_lag(s[dsl], tm[dsl], 80)
                row["scg_lag_dia"], row["scg_tc_dia"] = lag_d, c_d
                row["scg_ao"] = float(S.tracked_peak(s, T["ao"], 45) - PRE_SCG)
                row["scg_ac"] = float(S.tracked_peak(s, ac, 80, T["ac_sign"]) - PRE_SCG)
                sysw = s[PRE_SCG: PRE_SCG + 200]
                row["scg_amp_sys"] = float(sysw.max() - sysw.min())
                row["scg_rms_sys"] = float(np.sqrt(np.mean(sysw ** 2)))
                row["scg_rms_dia"] = float(np.sqrt(np.mean(s[dsl] ** 2)))
                row["scg_rms_pre"] = float(np.sqrt(np.mean(s[: PRE_SCG - 20] ** 2)))

        # Multi-Band Rotation-Invariant & Noise-Subtracted 3D SCG
        if acc_5_40 is not None:
            s5, _, _ = S.ensemble(acc_5_40, rb, PRE_SCG, POST_SCG)
            if s5 is not None:
                m5 = np.linalg.norm(s5, axis=1)
                e_pre5 = float(np.mean(m5[10: PRE_SCG - 20] ** 2))
                e_sys5 = float(np.mean(m5[PRE_SCG + 10: PRE_SCG + 180] ** 2))
                e_dia5 = float(np.mean(m5[PRE_SCG + 200: PRE_SCG + 460] ** 2))
                row["scg5_sys_net"] = float(np.sqrt(max(1e-8, e_sys5 - e_pre5)))
                row["scg5_dia_net"] = float(np.sqrt(max(1e-8, e_dia5 - e_pre5)))
                row["scg5_ao_3d"] = float(15 + np.argmax(m5[PRE_SCG + 15: PRE_SCG + 150]))
                seg_l = s5[PRE_SCG: PRE_SCG + 450] - s5[PRE_SCG: PRE_SCG + 450].mean(axis=0)
                _, _, vtl = np.linalg.svd(seg_l, full_matrices=False)
                wl = vtl[0]
                if w_rest5 is not None:
                    if np.dot(wl, w_rest5) < 0:
                        wl = -wl
                    row["scg5_axis_cos"] = float(np.abs(np.dot(wl, w_rest5)))
                sl = s5 @ wl
                if np.max(sl[PRE_SCG + 15: PRE_SCG + 180]) < -np.min(sl[PRE_SCG + 15: PRE_SCG + 180]):
                    sl = -sl
                ao5 = int(np.argmax(sl[PRE_SCG + 15: PRE_SCG + 160])) + 15
                row["scg5_ao_loc"] = float(ao5)

        if acc_12_45 is not None:
            s12, _, _ = S.ensemble(acc_12_45, rb, PRE_SCG, POST_SCG)
            if s12 is not None:
                m12 = np.linalg.norm(s12, axis=1)
                e_pre12 = float(np.mean(m12[10: PRE_SCG - 20] ** 2))
                e_sys12 = float(np.mean(m12[PRE_SCG + 10: PRE_SCG + 160] ** 2))
                e_dia12 = float(np.mean(m12[PRE_SCG + 200: PRE_SCG + 460] ** 2))
                sys_net12 = float(np.sqrt(max(1e-8, e_sys12 - e_pre12)))
                dia_net12 = float(np.sqrt(max(1e-8, e_dia12 - e_pre12)))
                row["scg12_sys_net"] = sys_net12
                row["scg12_dia_net"] = dia_net12
                row["scg12_s2_s1"] = float(dia_net12 / (sys_net12 + 1e-6))
                row["scg12_ao_3d"] = float(15 + np.argmax(m12[PRE_SCG + 15: PRE_SCG + 140]))

        # Chest PCG (used for independent acoustic S1/S2 validation)
        if "pcg" in T:
            e, _, _ = S.ensemble(d["_pcg_env"], rb, PRE_SCG, POST_SCG)
            if e is not None:
                s1_t = float(S.tracked_peak(e, T["s1"], 40) - PRE_SCG)
                s2_t = float(S.tracked_peak(e, int(T["s2"]), 80) - PRE_SCG) if np.isfinite(T.get("s2", np.nan)) else np.nan
                row["pcg_s1"] = s1_t
                if np.isfinite(s2_t):
                    row["pcg_s2"] = s2_t

        # PPG Finger & Ear contour
        for key, ch, search in (("ppgf", "PPG_Finger", (100, 500)), ("ppge", "PPG_Ear", (50, 420))):
            if "_" + ch not in d:
                continue
            p, q, _ = S.ensemble(d["_" + ch], rb, PRE_PPG, POST_PPG, clip=9.9, detrend_linear=True)
            if p is None:
                continue
            feats = S.extract_ppg_contour(p, PRE_PPG, search, fs=FS)
            for fk, fv in feats.items():
                row[f"{key}_{fk}"] = fv
            row[f"{key}_q"] = q

        # Differential zero-PEP arterial conduit transit time (PAT_finger - PAT_ear)
        if np.isfinite(row.get("ppgf_pat", np.nan)) and np.isfinite(row.get("ppge_pat", np.nan)):
            ptt_fe = row["ppgf_pat"] - row["ppge_pat"]
            if 5.0 <= ptt_fe <= 350.0:
                row["ptt_fe"] = float(ptt_fe)
                row["inv_ptt_fe2"] = float(1e6 / (ptt_fe ** 2))
        rows.append(row)
    return pd.DataFrame(rows)


def process(rec: str, out_dir: Path = OUT) -> str:
    t_start = time.time()
    d = load_record(rec)
    d["_ecg_f"] = S.bandpass(d["ECG"], 0.6, 40.0)
    if "Digital_Stethoscope" in d:
        d["_pcg_env"] = S.pcg_envelope(d["Digital_Stethoscope"], fs=FS, lo=30.0, hi=140.0)
    for ch in ("PPG_Finger", "PPG_Ear"):
        if ch in d:
            x = d[ch]
            if np.nanstd(x) < 0.01:
                continue
            clipped = np.abs(x) >= 9.9
            y = S.bandpass(np.where(clipped, 0.0, x), 0.45, 16.0)
            y[clipped] = 99.0
            d["_" + ch] = y
    acc_raw = np.stack([d["Accelerometer_X"], d["Accelerometer_Y"], d["Accelerometer_Z"]], axis=1)
    d["_acc_5_40"] = np.stack([S.bandpass(acc_raw[:, k], 5.0, 40.0) for k in range(3)], axis=1)
    d["_acc_12_45"] = np.stack([S.bandpass(acc_raw[:, k], 12.0, 45.0) for k in range(3)], axis=1)
    for ax in ("X", "Y", "Z"):
        d[f"Accelerometer_{ax}"] = S.bandpass(d[f"Accelerometer_{ax}"], 1.0, 40.0)
    r = S.detect_r_peaks(d["ECG"])
    ok = S.clean_rr_mask(r)
    T = build_templates(d, r, ok)
    df = window_features(d, r, ok, T)
    df.insert(0, "record", rec)
    df.insert(1, "subject", "_".join(rec.split("_")[:2]))
    df.insert(2, "cohort", rec.split("_")[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{rec}.csv.gz", index=False)
    tmpl = {k: v for k, v in T.items() if isinstance(v, (np.ndarray, float, int))}
    np.savez_compressed(out_dir / f"{rec}_templates.npz", **{k: np.asarray(v) for k, v in tmpl.items()})
    return f"{rec}: {len(df)} windows, {len(r)} beats, tmpl_beats={T.get('n_beats')} ({time.time() - t_start:.1f}s)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--records", nargs="*")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    recs = a.records or list_records()
    todo = recs if a.force else [x for x in recs if not (OUT / f"{x}.csv.gz").exists()]
    print(f"{len(todo)} records to process -> {OUT}", flush=True)
    if a.workers <= 1:
        for x in todo:
            print(process(x, OUT), flush=True)
    else:
        with ProcessPoolExecutor(a.workers) as ex:
            futs = {x: ex.submit(process, x, OUT) for x in todo}
            for x, f in futs.items():
                try:
                    print(f.result(), flush=True)
                except Exception as e:
                    print(f"{x}: FAILED {e!r}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()

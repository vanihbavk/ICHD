"""Construct window-level and episode-level baseline-referenced hemodynamic log-delta tables."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .extract import OUT, RAW

ACT_NAMES = {
    0: "rest",
    1: "breath_hold",
    2: "valsalva",
    3: "exercise",
    4: "cold_pressor",
    9: "not_applicable",
}


def load_demographics() -> pd.DataFrame:
    df = pd.read_csv(RAW / "demographics.csv")
    df.columns = [c.strip() for c in df.columns]
    df["subject"] = df["ID"].str.strip()
    return df


def enrich_session(df: pd.DataFrame, tmpl_path: Path | None = None) -> pd.DataFrame:
    """Add physiological intervals (PEP, LVET, Weissler ratio, PEP-corrected PTT)
    and exact hemodynamic log-quantities to a single session window DataFrame."""
    df = df.copy()
    t_ao = float(df["scg_ao"].median()) if "scg_ao" in df else 100.0
    t_ac = float(df["scg_ac"].median()) if "scg_ac" in df else 310.0
    if tmpl_path is not None and tmpl_path.exists():
        z = np.load(tmpl_path)
        if "ao" in z:
            t_ao = float(z["ao"]) - 100.0
        if "ac" in z:
            t_ac = float(z["ac"]) - 100.0

    # Lag-tracked PEP and LVET (more robust to beat morphology jitter than raw argmax)
    df["pep_lag"] = t_ao + df.get("scg_lag_sys", np.nan)
    df["ac_lag"] = t_ac + df.get("scg_lag_dia", np.nan)
    df["lvet_lag"] = df["ac_lag"] - df["pep_lag"]
    df["weissler"] = df["pep_lag"] / df["lvet_lag"].clip(lower=120.0)

    # True vascular Pulse Transit Time (PTT = PAT - PEP) vs uncorrected PAT
    for k in ("ppgf", "ppge"):
        pat_col = f"{k}_pat"
        if pat_col in df:
            df[f"{k}_ptt"] = df[pat_col] - df["pep_lag"]
        else:
            df[f"{k}_pat"] = np.nan
            df[f"{k}_ptt"] = np.nan
            df[f"{k}_amp"] = np.nan
            df[f"{k}_slope"] = np.nan
            df[f"{k}_q"] = np.nan
            df[f"{k}_rise"] = np.nan
            df[f"{k}_hw"] = np.nan

    # Dynamic 5-40 Hz local-SVD PEP-corrected transit times & LVET
    if "scg5_ao_loc" in df:
        df["scg5_lvet_loc"] = df.get("scg5_ac_loc", df["ac_lag"]) - df["scg5_ao_loc"]
        df["ppgf_ptt5"] = df["ppgf_pat"] - df["scg5_ao_loc"]
        df["ppge_ptt5"] = df["ppge_pat"] - df["scg5_ao_loc"]
    else:
        df["scg5_lvet_loc"] = np.nan
        df["ppgf_ptt5"] = np.nan
        df["ppge_ptt5"] = np.nan

    # SCG log-amplitude and RMS metrics
    rms_sys = df.get("scg_rms_sys", pd.Series(np.nan, index=df.index)).clip(lower=1e-6)
    amp_sys = df.get("scg_amp_sys", pd.Series(np.nan, index=df.index)).clip(lower=1e-5)
    df["scg_amp_log"] = np.log(amp_sys)
    df["scg_rms_sys_log"] = np.log(rms_sys)

    # Reference hemodynamics:
    # Preserve true CNAP Cardiac Output (ref_co) and enforce exact CNAP Ohm's law
    # SVR = 80 * MAP / CO (avoiding ECG HR-error leakage into ground-truth targets).
    valid_ref = (
        df["ref_map"].between(40, 190)
        & df["ref_sv"].between(15, 148)
        & df["ref_co"].between(1.2, 19.5)
        & df["ref_hr"].between(35, 200)
        & df["hr"].between(35, 200)
    )
    df["valid_ref"] = valid_ref
    df["ref_co_cnap"] = df["ref_co"]
    df["ref_co"] = np.where(valid_ref, df["ref_co_cnap"], np.nan)
    df["ref_svr"] = np.where(valid_ref, 80.0 * df["ref_map"] / df["ref_co"], np.nan)
    return df


_PPG_CONTOUR_SUFFIXES = [
    "pat", "ptt", "ptt5", "inv_pat2", "amp", "slope", "max_dp_norm",
    "rise", "pw25", "pw50", "pw75", "hw", "t_notch", "notch_amp",
    "ri_dia", "t_sys_dia", "ipa", "area_sys_norm", "area_raw",
    "dia_decay", "dia_slope", "apg_ba", "apg_da", "apg_ea", "apg_agi", "q",
]

FEATURE_BASE_COLS = [
    "hr",
    "rr_sd",
    "ecg_rt",
    "ecg_t_over_r",
    "pep_lag",
    "scg_ao",
    "ac_lag",
    "scg_ac",
    "lvet_lag",
    "weissler",
    "scg_amp_log",
    "scg_rms_sys_log",
    "scg_tc_dia",
    "scg_q",
    "ecg_q",
    # Multi-band 3D rotation-invariant & noise-subtracted chest electromechanics
    "scg5_sys_net",
    "scg5_dia_net",
    "scg5_ao_3d",
    "scg5_axis_cos",
    "scg5_ao_loc",
    "scg5_lvet_loc",
    "scg12_sys_net",
    "scg12_dia_net",
    "scg12_s2_s1",
    "scg12_ao_3d",
    # PCG acoustic timing (used only for independent biophysical validation)
    "pcg_s1",
    "pcg_s2",
    # Two-site zero-PEP conduit arterial transit
    "ptt_fe",
    "inv_ptt_fe2",
    *[f"ppgf_{s}" for s in _PPG_CONTOUR_SUFFIXES],
    *[f"ppge_{s}" for s in _PPG_CONTOUR_SUFFIXES],
]

REF_COLS = [
    "ref_map",
    "ref_sbp",
    "ref_dbp",
    "ref_sv",
    "ref_co",
    "ref_co_cnap",
    "ref_hr",
    "ref_svr",
]


def _enforce_base_closure(base: pd.Series) -> pd.Series:
    """Enforce exact additive and multiplicative closure on resting baseline."""
    b = base.copy()
    if np.isfinite(b.get("ac_lag", np.nan)) and np.isfinite(b.get("pep_lag", np.nan)):
        b["lvet_lag"] = b["ac_lag"] - b["pep_lag"]
    if np.isfinite(b.get("ppgf_pat", np.nan)) and np.isfinite(b.get("pep_lag", np.nan)):
        b["ppgf_ptt"] = b["ppgf_pat"] - b["pep_lag"]
    if np.isfinite(b.get("ppge_pat", np.nan)) and np.isfinite(b.get("pep_lag", np.nan)):
        b["ppge_ptt"] = b["ppge_pat"] - b["pep_lag"]
    if np.isfinite(b.get("ref_map", np.nan)) and np.isfinite(b.get("ref_co", np.nan)) and b["ref_co"] > 0:
        b["ref_svr"] = 80.0 * b["ref_map"] / b["ref_co"]
    return b


def build_deltas_and_episodes(in_dir: Path = OUT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (1) window-level delta table and (2) contiguous maneuver-episode table
    referenced to the local pre-maneuver resting baseline, preserving exact linear
    additivity across all circulation and timing identities."""
    demo = load_demographics().set_index("subject")
    win_rows = []
    ep_rows = []

    files = sorted(in_dir.glob("*.csv.gz"))
    for fp in files:
        rec = fp.name.replace(".csv.gz", "")
        tmpl_fp = in_dir / f"{rec}_templates.npz"
        df = pd.read_csv(fp)
        if len(df) < 10:
            continue
        df = enrich_session(df, tmpl_fp)
        missing_cols = [col for col in FEATURE_BASE_COLS if col not in df.columns]
        if missing_cols:
            df = pd.concat([df, pd.DataFrame(np.nan, index=df.index, columns=missing_cols)], axis=1)

        # Clean resting windows available as baseline candidates
        rest_mask = (
            (df["act"] == 0)
            & (df["act_frac"] >= 0.8)
            & df["valid_ref"]
            & (df["scg_q"].fillna(0) >= 0.25)
        )
        if rest_mask.sum() < 3:
            rest_mask = df["valid_ref"] & (df["t0"] <= 600) & (df["scg_q"].fillna(0) >= 0.20)
        if rest_mask.sum() < 2:
            continue

        session_base = _enforce_base_closure(
            df.loc[rest_mask, FEATURE_BASE_COLS + REF_COLS].median()
        )

        # Identify contiguous activity bouts
        act_arr = df["act"].to_numpy()
        chg = np.flatnonzero(np.diff(act_arr) != 0)
        starts = np.r_[0, chg + 1]
        ends = np.r_[chg + 1, len(df)]

        bout_id = 0
        for st, en in zip(starts, ends):
            act_code = int(act_arr[st])
            t_onset = float(df["t0"].iloc[st])
            pre_mask = rest_mask & (df["t0"] >= t_onset - 120.0) & (df["t0"] <= t_onset - 5.0)
            is_causal = bool(pre_mask.sum() >= 2)
            if is_causal:
                base = _enforce_base_closure(
                    df.loc[pre_mask, FEATURE_BASE_COLS + REF_COLS].median()
                )
            else:
                base = session_base

            if not (np.isfinite(base["ref_map"]) and np.isfinite(base["ref_co"]) and np.isfinite(base["ref_sv"])):
                continue

            bout_df = df.iloc[st:en].copy()
            bout_valid = bout_df[bout_df["valid_ref"] & (bout_df["n_beats"] >= 5)].copy()
            if len(bout_valid) == 0:
                continue

            if act_code in (1, 2, 3, 4):
                bout_id += 1

            for _, w in bout_valid.iterrows():
                if w["act_frac"] < 0.7:
                    continue
                r = {
                    "record": rec,
                    "subject": w["subject"],
                    "cohort": w["cohort"],
                    "bout_id": f"{rec}_b{bout_id}" if act_code in (1, 2, 3, 4) else f"{rec}_rest",
                    "t0": float(w["t0"]),
                    "act": act_code,
                    "act_name": ACT_NAMES.get(act_code, "other"),
                }
                # Measured ECG log-delta HR (exact observable on chest patch)
                r["dlog_hr"] = float(np.log(w["hr"] / base["hr"]))
                r["d_hr_bpm"] = float(w["hr"] - base["hr"])

                # Exact CNAP log-delta references obeying dlog_map = dlog_co + dlog_svr
                r["dlog_map"] = float(np.log(w["ref_map"] / base["ref_map"]))
                r["dlog_sv"] = float(np.log(w["ref_sv"] / base["ref_sv"]))
                r["dlog_hr_ref"] = float(np.log(w["ref_hr"] / base["ref_hr"]))
                r["dlog_co_cnap"] = float(np.log(w["ref_co_cnap"] / base["ref_co_cnap"]))
                r["dlog_co"] = float(r["dlog_co_cnap"])
                r["dlog_svr"] = float(r["dlog_map"] - r["dlog_co"])

                r["d_map_mmhg"] = float(w["ref_map"] - base["ref_map"])
                r["d_sbp_mmhg"] = float(w["ref_sbp"] - base["ref_sbp"])
                r["d_sv_ml"] = float(w["ref_sv"] - base["ref_sv"])
                r["d_co_lpm"] = float(w["ref_co"] - base["ref_co"])
                r["d_svr_dyn"] = float(w["ref_svr"] - base["ref_svr"])
                r["base_hr"] = float(base["hr"])

                # Feature deltas, levels, and dimensionless baseline-normalized log-ratios
                for col in FEATURE_BASE_COLS:
                    val = float(w[col]) if col in w and pd.notna(w[col]) else np.nan
                    bval = float(base[col]) if col in base and pd.notna(base[col]) else np.nan
                    r[col] = val
                    r[f"d_{col}"] = val - bval if (np.isfinite(val) and np.isfinite(bval)) else np.nan
                    if np.isfinite(val) and np.isfinite(bval) and bval > 1e-4 and val > 1e-4:
                        r[f"dlog_{col}"] = float(np.clip(np.log(val / bval), -2.0, 2.0))
                    else:
                        r[f"dlog_{col}"] = np.nan

                # Enforce exact additive timing closure on deltas: d_ppgf_ptt == d_ppgf_pat - d_pep_lag
                if np.isfinite(r["d_ppgf_pat"]) and np.isfinite(r["d_pep_lag"]):
                    r["d_ppgf_ptt"] = float(r["d_ppgf_pat"] - r["d_pep_lag"])
                if np.isfinite(r["d_ppge_pat"]) and np.isfinite(r["d_pep_lag"]):
                    r["d_ppge_ptt"] = float(r["d_ppge_pat"] - r["d_pep_lag"])

                # Hemodynamic vector angle and mechanism dominance
                r["hemo_angle_rad"] = float(np.arctan2(r["dlog_svr"], r["dlog_co"]))
                r["y_res_dom"] = int(r["dlog_svr"] > r["dlog_co"])

                if w["subject"] in demo.index:
                    drow = demo.loc[w["subject"]]
                    try:
                        r["lvef"] = float(str(drow.get("LV ejection fraction", "")).replace("%", "").strip())
                    except (ValueError, TypeError):
                        r["lvef"] = np.nan
                win_rows.append(r)

            # Summarize steady-state second half of bout via linear mean
            if act_code in (1, 2, 3, 4):
                sub_b = [x for x in win_rows if x["bout_id"] == f"{rec}_b{bout_id}"]
                if len(sub_b) >= 1:
                    bdf = pd.DataFrame(sub_b)
                    half_bdf = bdf.iloc[len(bdf) // 2:] if len(bdf) >= 2 else bdf
                    agg = half_bdf.select_dtypes(include=[np.number]).mean().to_dict()
                    # Re-enforce exact algebraic MAP=CO+SVR closure on CNAP references
                    agg["dlog_svr"] = float(agg["dlog_map"] - agg["dlog_co"])
                    agg["hemo_angle_rad"] = float(np.arctan2(agg["dlog_svr"], agg["dlog_co"]))
                    if np.isfinite(agg.get("d_ppgf_pat", np.nan)) and np.isfinite(agg.get("d_pep_lag", np.nan)):
                        agg["d_ppgf_ptt"] = float(agg["d_ppgf_pat"] - agg["d_pep_lag"])
                    if np.isfinite(agg.get("d_ppge_pat", np.nan)) and np.isfinite(agg.get("d_pep_lag", np.nan)):
                        agg["d_ppge_ptt"] = float(agg["d_ppge_pat"] - agg["d_pep_lag"])
                    agg["record"] = rec
                    agg["subject"] = sub_b[0]["subject"]
                    agg["cohort"] = sub_b[0]["cohort"]
                    agg["bout_id"] = f"{rec}_b{bout_id}"
                    agg["act"] = act_code
                    agg["act_name"] = ACT_NAMES.get(act_code, "other")
                    agg["is_causal_baseline"] = is_causal
                    agg["y_res_dom"] = int(agg["dlog_svr"] > agg["dlog_co"])
                    ep_rows.append(agg)

    win_df = pd.DataFrame(win_rows)
    ep_df = pd.DataFrame(ep_rows)
    return win_df, ep_df


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    TABLES_CACHE = ROOT / "data" / "interim" / "ichd_tables"
    TABLES_CACHE.mkdir(parents=True, exist_ok=True)
    w_df, ep_df = build_deltas_and_episodes()
    w_df.to_pickle(TABLES_CACHE / "windows.pkl")
    ep_df.to_pickle(TABLES_CACHE / "episodes.pkl")
    print(f"Saved {len(w_df)} windows and {len(ep_df)} episodes with exact manifold closure.")

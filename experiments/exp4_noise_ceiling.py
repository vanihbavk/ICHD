"""Experiment 4: CNAP Reference Noise Ceiling & Test-Retest Reliability Analysis."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TABLES_CACHE = ROOT / "data" / "interim" / "ichd_tables"
OUT_DIR = ROOT / "results"


def run_noise_ceiling_analysis():
    ep = pd.read_pickle(TABLES_CACHE / "episodes.pkl")
    rec_per_sub = ep.groupby("subject")["record"].nunique()
    multi_subs = rec_per_sub[rec_per_sub == 2].index

    pairs = []
    for sub in multi_subs:
        sub_df = ep[ep["subject"] == sub]
        recs = sub_df["record"].unique()
        r1_df = sub_df[sub_df["record"] == recs[0]]
        r2_df = sub_df[sub_df["record"] == recs[1]]
        for act in ["exercise", "cold_pressor", "valsalva", "breath_hold"]:
            m1 = r1_df[r1_df["act_name"] == act]
            m2 = r2_df[r2_df["act_name"] == act]
            if len(m1) > 0 and len(m2) > 0:
                pairs.append({
                    "subject": sub,
                    "act": act,
                    "s1_dlog_sv": float(m1["dlog_sv"].mean()),
                    "s2_dlog_sv": float(m2["dlog_sv"].mean()),
                    "s1_dlog_co": float(m1["dlog_co"].mean()),
                    "s2_dlog_co": float(m2["dlog_co"].mean()),
                    "s1_dlog_svr": float(m1["dlog_svr"].mean()),
                    "s2_dlog_svr": float(m2["dlog_svr"].mean()),
                    "s1_dlog_map": float(m1["dlog_map"].mean()),
                    "s2_dlog_map": float(m2["dlog_map"].mean()),
                })

    pdf = pd.DataFrame(pairs)
    results = {
        "n_multi_session_subjects": len(multi_subs),
        "n_matched_maneuver_pairs": len(pdf),
        "variables": {},
    }

    with open(OUT_DIR / "ichd_results.json") as f:
        res_full = json.load(f)["exp2_cv"]["hybrid"]["ICHD-Full"]
    model_r = {
        "dlog_sv": round(float(res_full["r_sv"]), 3),
        "dlog_co": round(float(res_full["r_co"]), 3),
        "dlog_svr": round(float(res_full["r_svr"]), 3),
        "dlog_map": round(float(res_full["r_map"]), 3),
    }

    for var in ["dlog_sv", "dlog_co", "dlog_svr", "dlog_map"]:
        s1 = pdf[f"s1_{var}"].values
        s2 = pdf[f"s2_{var}"].values
        valid = np.isfinite(s1) & np.isfinite(s2)
        s1, s2 = s1[valid], s2[valid]
        r, p = stats.pearsonr(s1, s2)
        r_max = float(np.sqrt(max(r, 0.0)))
        pct_captured = float((model_r[var] / r_max) * 100.0) if r_max > 0 else 0.0
        results["variables"][var] = {
            "test_retest_r": round(float(r), 3),
            "p_val": float(f"{p:.2e}"),
            "theoretical_ceiling_r_max": round(r_max, 3),
            "model_r_achieved": model_r[var],
            "fraction_of_ceiling_pct": round(pct_captured, 1),
        }

    with open(OUT_DIR / "exp4_noise_ceiling_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"exp4_noise_ceiling: n_subs={len(multi_subs)}, n_pairs={len(pdf)}, sv_max={results['variables']['dlog_sv']['theoretical_ceiling_r_max']}")


if __name__ == "__main__":
    run_noise_ceiling_analysis()

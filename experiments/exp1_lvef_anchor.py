"""Experiment 1: Heart-failure exercise inotropic reserve vs. clinical echocardiographic LVEF."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TABLES_CACHE = ROOT / "data" / "interim" / "ichd_tables"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)


def run_lvef_analysis():
    ep = pd.read_pickle(TABLES_CACHE / "episodes.pkl")

    hc_ex = ep[(ep["cohort"] == "HC") & (ep["act_name"] == "exercise")]
    hfc_ex = ep[(ep["cohort"] == "HFC") & (ep["act_name"] == "exercise")]

    hc_d_pep = float(hc_ex["d_pep_lag"].mean())
    hfc_d_pep = float(hfc_ex["d_pep_lag"].mean())
    hc_d_co = float(hc_ex["d_co_lpm"].mean())
    hfc_d_co = float(hfc_ex["d_co_lpm"].mean())

    hfc_lvef_ex = hfc_ex.dropna(subset=["lvef"]).copy()
    sub_ex = hfc_lvef_ex.groupby("subject").mean(numeric_only=True)

    hfref = sub_ex[sub_ex["lvef"] <= 40]
    hfpef = sub_ex[sub_ex["lvef"] > 40]

    hfref_lvef_mean = float(hfref["lvef"].mean())
    hfpef_lvef_mean = float(hfpef["lvef"].mean())

    hfref_d_pep_mean = float(hfref["d_pep_lag"].mean())
    hfpef_d_pep_mean = float(hfpef["d_pep_lag"].mean())

    hfref_d_co_mean = float(hfref["d_co_lpm"].mean())
    hfpef_d_co_mean = float(hfpef["d_co_lpm"].mean())

    t_pep, p_pep = stats.ttest_ind(hfref["d_pep_lag"].dropna(), hfpef["d_pep_lag"].dropna())
    r_pep_lvef, p_r_pep = stats.pearsonr(sub_ex["d_pep_lag"], sub_ex["lvef"])
    rho_pep_lvef, _ = stats.spearmanr(sub_ex["d_pep_lag"], sub_ex["lvef"])

    hfc_all = ep[ep["cohort"] == "HFC"].dropna(subset=["lvef"])
    sub_all = hfc_all.groupby("subject").first(numeric_only=True)
    sub_all["pep_lvet"] = sub_all["pep_lag"] / sub_all["lvet_lag"]
    valid_weissler = sub_all.dropna(subset=["pep_lvet", "lvef"])
    rho_weissler, _ = stats.spearmanr(valid_weissler["pep_lvet"], valid_weissler["lvef"])
    
    n_hc_ex = int(hc_ex["subject"].nunique())
    res = {
        "n_hc_subjects": n_hc_ex,
        "n_hfc_subjects_lvef": len(sub_ex),
        "hc_exercise_d_pep_ms": round(hc_d_pep, 2),
        "hfc_exercise_d_pep_ms": round(hfc_d_pep, 2),
        "hc_exercise_d_co_lpm": round(hc_d_co, 2),
        "hfc_exercise_d_co_lpm": round(hfc_d_co, 2),
        "hfref_n": len(hfref),
        "hfref_lvef_mean": round(hfref_lvef_mean, 1),
        "hfref_d_pep_ms_mean": round(hfref_d_pep_mean, 2),
        "hfref_d_co_lpm_mean": round(hfref_d_co_mean, 2),
        "hfpef_n": len(hfpef),
        "hfpef_lvef_mean": round(hfpef_lvef_mean, 1),
        "hfpef_d_pep_ms_mean": round(hfpef_d_pep_mean, 2),
        "hfpef_d_co_lpm_mean": round(hfpef_d_co_mean, 2),
        "pep_diff_hfref_vs_hfpef_ms": round(hfref_d_pep_mean - hfpef_d_pep_mean, 2),
        "ttest_pep_t": round(float(t_pep), 3),
        "ttest_pep_p": round(float(p_pep), 4),
        "corr_d_pep_vs_lvef_r": round(float(r_pep_lvef), 3),
        "corr_d_pep_vs_lvef_p": round(float(p_r_pep), 4),
        "corr_d_pep_vs_lvef_rho": round(float(rho_pep_lvef), 3),
        "corr_weissler_vs_lvef_rho": round(float(rho_weissler), 3),
    }

    with open(OUT_DIR / "exp1_lvef_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"exp1_lvef_anchor: n_hfc={len(sub_ex)}, r={r_pep_lvef:+.3f} (p={p_r_pep:.3f})")


if __name__ == "__main__":
    run_lvef_analysis()

"""Experiment 8: Isolate the 2x2 identity constraint against unconstrained baselines."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from ichd.model import CONFIG_FEATURES, _split_subsystem_cols, _make_reg
from ichd.stats import _r_ci, _paired_dr_ci

ROOT = Path(__file__).resolve().parents[1]
TABLES_CACHE = ROOT / "data" / "interim" / "ichd_tables"
OUT_DIR = ROOT / "results"


def reconcile_manifold_variant_a(
    h: np.ndarray,
    z_sv: np.ndarray,
    z_svr: np.ndarray,
    z_diff_noh: np.ndarray,
    z_map: np.ndarray,
    var_sv: float,
    var_svr: float,
    var_diff: float,
    var_map: float,
    w_map_scale: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Projection where diff head targets (SVR - SV) directly without subtracting h.
    Loss: min w_sv(u_sv - z_sv)^2 + w_svr(u_svr - z_svr)^2
             + w_diff((u_svr - u_sv) - z_diff_noh)^2
             + w_map((h + u_sv + u_svr) - z_map)^2
    """
    w_sv = 1.0 / max(var_sv, 1e-5)
    w_svr = 1.0 / max(var_svr, 1e-5)
    w_diff = 1.0 / max(var_diff, 1e-5)
    w_map = w_map_scale / max(var_map, 1e-5)

    a11 = w_sv + w_diff + w_map
    a22 = w_svr + w_diff + w_map
    a12 = w_map - w_diff
    det = a11 * a22 - a12 * a12

    b1 = w_sv * z_sv - w_diff * z_diff_noh + w_map * (z_map - h)
    b2 = w_svr * z_svr + w_diff * z_diff_noh + w_map * (z_map - h)

    u_sv = (a22 * b1 - a12 * b2) / det
    u_svr = (a11 * b2 - a12 * b1) / det
    u_co = h + u_sv
    u_map = u_co + u_svr
    return u_sv, u_co, u_svr, u_map


def reconcile_manifold_custom_wmap(
    h: np.ndarray,
    z_sv: np.ndarray,
    z_svr: np.ndarray,
    z_diff: np.ndarray,
    z_map: np.ndarray,
    var_sv: float,
    var_svr: float,
    var_diff: float,
    var_map: float,
    w_map_scale: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standard 2x2 projection with variable w_map scale."""
    w_sv = 1.0 / max(var_sv, 1e-5)
    w_svr = 1.0 / max(var_svr, 1e-5)
    w_diff = 1.0 / max(var_diff, 1e-5)
    w_map = w_map_scale / max(var_map, 1e-5)

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


def run_identity_isolation():
    ep = pd.read_pickle(TABLES_CACHE / "episodes.pkl")
    fcols = CONFIG_FEATURES["ICHD-Full"]

    n = len(ep)
    g_all = ep["subject"].to_numpy()
    y_sv_all = ep["dlog_sv"].to_numpy(dtype=float)

    gkf = GroupKFold(n_splits=5)

    preds = {
        "1_subsystem_unprojected": np.zeros(n),
        "2_direct_full_features": np.zeros(n),
        "3_matched_unconstrained_stacker": np.zeros(n),
        "3b_matched_unconstrained_avg": np.zeros(n),
        "4_current_projected": np.zeros(n),
        "5_variant_a_no_h_diff": np.zeros(n),
    }

    w_map_grid = [0.0, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0]
    preds_grid = {f"wmap_{s}": np.zeros(n) for s in w_map_grid}
    closure_gaps = {k: np.zeros(n) for k in preds.keys()}

    for tr, te in gkf.split(ep, groups=g_all):
        df_tr = ep.iloc[tr]
        df_te = ep.iloc[te]

        f_card, f_vasc, f_full = _split_subsystem_cols(fcols, list(ep.columns))

        X_card_tr = df_tr[f_card].to_numpy(dtype=float)
        X_vasc_tr = df_tr[f_vasc].to_numpy(dtype=float)
        X_full_tr = df_tr[f_full].to_numpy(dtype=float)

        X_card_te = df_te[f_card].to_numpy(dtype=float)
        X_vasc_te = df_te[f_vasc].to_numpy(dtype=float)
        X_full_te = df_te[f_full].to_numpy(dtype=float)

        h_tr = df_tr["dlog_hr"].to_numpy(dtype=float)
        h_te = df_te["dlog_hr"].to_numpy(dtype=float)

        y_sv_tr = df_tr["dlog_sv"].to_numpy(dtype=float)
        y_svr_tr = df_tr["dlog_svr"].to_numpy(dtype=float)
        y_map_tr = df_tr["dlog_map"].to_numpy(dtype=float)
        y_diff_tr = y_svr_tr - (h_tr + y_sv_tr)
        y_diff_noh_tr = y_svr_tr - y_sv_tr

        inner_gkf = GroupKFold(n_splits=3)
        n_tr = len(df_tr)
        oof_sv = np.zeros(n_tr)
        oof_svr = np.zeros(n_tr)
        oof_diff = np.zeros(n_tr)
        oof_diff_noh = np.zeros(n_tr)
        oof_map = np.zeros(n_tr)

        g_tr = df_tr["subject"].to_numpy()
        for in_tr, in_val in inner_gkf.split(df_tr, groups=g_tr):
            r_sv_in = _make_reg("ridge").fit(X_card_tr[in_tr], y_sv_tr[in_tr])
            r_svr_in = _make_reg("ridge").fit(X_vasc_tr[in_tr], y_svr_tr[in_tr])
            r_diff_in = _make_reg("ridge").fit(X_full_tr[in_tr], y_diff_tr[in_tr])
            r_diff_noh_in = _make_reg("ridge").fit(X_full_tr[in_tr], y_diff_noh_tr[in_tr])
            r_map_in = _make_reg("ridge").fit(X_full_tr[in_tr], y_map_tr[in_tr])

            oof_sv[in_val] = r_sv_in.predict(X_card_tr[in_val])
            oof_svr[in_val] = r_svr_in.predict(X_vasc_tr[in_val])
            oof_diff[in_val] = r_diff_in.predict(X_full_tr[in_val])
            oof_diff_noh[in_val] = r_diff_noh_in.predict(X_full_tr[in_val])
            oof_map[in_val] = r_map_in.predict(X_full_tr[in_val])

        var_sv = float(np.var(y_sv_tr - oof_sv))
        var_svr = float(np.var(y_svr_tr - oof_svr))
        var_diff = float(np.var(y_diff_tr - oof_diff))
        var_diff_noh = float(np.var(y_diff_noh_tr - oof_diff_noh))
        var_map = float(np.var(y_map_tr - oof_map))

        reg_sv = _make_reg("ridge").fit(X_card_tr, y_sv_tr)
        reg_svr = _make_reg("ridge").fit(X_vasc_tr, y_svr_tr)
        reg_diff = _make_reg("ridge").fit(X_full_tr, y_diff_tr)
        reg_diff_noh = _make_reg("ridge").fit(X_full_tr, y_diff_noh_tr)
        reg_map = _make_reg("ridge").fit(X_full_tr, y_map_tr)

        z_sv_te = reg_sv.predict(X_card_te)
        z_svr_te = reg_svr.predict(X_vasc_te)
        z_diff_te = reg_diff.predict(X_full_te)
        z_diff_noh_te = reg_diff_noh.predict(X_full_te)
        z_map_te = reg_map.predict(X_full_te)

        preds["1_subsystem_unprojected"][te] = z_sv_te

        reg_full_direct = _make_reg("ridge").fit(X_full_tr, y_sv_tr)
        preds["2_direct_full_features"][te] = reg_full_direct.predict(X_full_te)

        stack_X_tr = np.column_stack([oof_sv, oof_svr, oof_diff, oof_map, h_tr])
        stack_X_te = np.column_stack([z_sv_te, z_svr_te, z_diff_te, z_map_te, h_te])
        stacker = Ridge(alpha=1.0).fit(stack_X_tr, y_sv_tr)
        preds["3_matched_unconstrained_stacker"][te] = stacker.predict(stack_X_te)

        c1 = z_sv_te
        c2 = z_svr_te - z_diff_te - h_te
        c3 = z_map_te - z_svr_te - h_te
        preds["3b_matched_unconstrained_avg"][te] = (c1 + c2 + c3) / 3.0

        u_sv_4, u_co_4, u_svr_4, u_map_4 = reconcile_manifold_custom_wmap(
            h_te, z_sv_te, z_svr_te, z_diff_te, z_map_te,
            var_sv, var_svr, var_diff, var_map, w_map_scale=0.25
        )
        preds["4_current_projected"][te] = u_sv_4
        closure_gaps["4_current_projected"][te] = np.abs(u_map_4 - (h_te + u_sv_4 + u_svr_4))

        u_sv_5, u_co_5, u_svr_5, u_map_5 = reconcile_manifold_variant_a(
            h_te, z_sv_te, z_svr_te, z_diff_noh_te, z_map_te,
            var_sv, var_svr, var_diff_noh, var_map, w_map_scale=0.25
        )
        preds["5_variant_a_no_h_diff"][te] = u_sv_5
        closure_gaps["5_variant_a_no_h_diff"][te] = np.abs(u_map_5 - (h_te + u_sv_5 + u_svr_5))

        for s in w_map_grid:
            u_sv_s, _, _, _ = reconcile_manifold_custom_wmap(
                h_te, z_sv_te, z_svr_te, z_diff_te, z_map_te,
                var_sv, var_svr, var_diff, var_map, w_map_scale=s
            )
            preds_grid[f"wmap_{s}"][te] = u_sv_s

    ref_pred = preds["4_current_projected"]
    r_proj, r_proj_lo, r_proj_hi = _r_ci(y_sv_all, ref_pred, g_all, n_boot=2000, seed=42)

    results = {
        "n_episodes": n,
        "n_subjects": int(len(np.unique(g_all))),
        "baseline_projected": {
            "r_sv": round(r_proj, 3),
            "ci": [round(r_proj_lo, 3), round(r_proj_hi, 3)],
        },
        "variants": {},
        "w_map_grid_sweep": {},
    }

    for name, pred_arr in preds.items():
        r, r_lo, r_hi = _r_ci(y_sv_all, pred_arr, g_all, n_boot=2000, seed=42)
        dr, dr_lo, dr_hi, p_boot = _paired_dr_ci(y_sv_all, pred_arr, ref_pred, g_all, n_boot=2000, seed=42)
        results["variants"][name] = {
            "r_sv": round(r, 3),
            "ci": [round(r_lo, 3), round(r_hi, 3)],
            "delta_r_vs_projected": round(dr, 3),
            "delta_r_ci": [round(dr_lo, 3), round(dr_hi, 3)],
            "p_boot": round(p_boot, 4),
            "closure_gap_mean": float(np.mean(closure_gaps.get(name, 0.0))),
        }

    for s in w_map_grid:
        p_s = preds_grid[f"wmap_{s}"]
        r, r_lo, r_hi = _r_ci(y_sv_all, p_s, g_all, n_boot=2000, seed=42)
        dr, dr_lo, dr_hi, p_boot = _paired_dr_ci(y_sv_all, p_s, ref_pred, g_all, n_boot=2000, seed=42)
        results["w_map_grid_sweep"][f"wmap_{s}"] = {
            "scale": s,
            "r_sv": round(r, 3),
            "ci": [round(r_lo, 3), round(r_hi, 3)],
            "delta_r_vs_default": round(dr, 3),
            "delta_r_ci": [round(dr_lo, 3), round(dr_hi, 3)],
            "p_boot": round(p_boot, 4),
        }

    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "exp8_identity_isolation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"exp8_identity_isolation: projected r_sv={r_proj:+.3f}, stacker r_sv={results['variants']['3_matched_unconstrained_stacker']['r_sv']:+.3f}")


if __name__ == "__main__":
    run_identity_isolation()

"""Generate publication vector figures (muted scientific color palette) for ICHD."""
from __future__ import annotations

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np
import pandas as pd

from .model import CONFIG_FEATURES
from .evaluate import _oof_cv, ROOT, TABLES_CACHE, RESULTS_DIR

FIG_DIR = ROOT / "figures"

MINIMAL_RC = {
    "font.family": "serif",
    "font.size": 11.0,
    "axes.titlesize": 11.8,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.8,
    "xtick.labelsize": 9.8,
    "ytick.labelsize": 9.8,
    "legend.fontsize": 9.2,
    "figure.dpi": 300,
    "axes.linewidth": 0.85,
    "axes.edgecolor": "#222222",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# Muted, publication-grade scientific palette
PAL_NAVY = "#234361"
PAL_GREEN = "#216645"
PAL_CRIMSON = "#A63A33"
PAL_OCHRE = "#C96A2B"
PAL_PLUM = "#5E4B8B"
PAL_STEEL = "#6E8898"
PAL_SLATE_LT = "#8AA4BC"

_MANEUVER_STYLE = {
    "exercise": {"color": PAL_GREEN, "edge": "#111111", "marker": "o", "label": "Exercise (Flow)"},
    "cold_pressor": {"color": PAL_CRIMSON, "edge": "#111111", "marker": "s", "label": "Cold Pressor (SVR)"},
    "valsalva": {"color": PAL_NAVY, "edge": "#111111", "marker": "D", "label": "Valsalva (Preload)"},
    "breath_hold": {"color": PAL_OCHRE, "edge": "#111111", "marker": "^", "label": "Breath-Hold (Apnea)"},
}

_METRIC_BAR_PALETTE = [
    {"facecolor": PAL_SLATE_LT, "edgecolor": "#1a1a1a"},
    {"facecolor": PAL_GREEN, "edgecolor": "#1a1a1a"},
    {"facecolor": PAL_NAVY, "edgecolor": "#1a1a1a"},
]


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _clean_grid(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(axis=axis, ls="-", lw=0.45, color="#e2e2e2", zorder=0)
    ax.set_axisbelow(True)


def _plot_fig1_architecture(res: dict) -> None:
    b = res["exp1_biophysics"]
    with plt.rc_context(MINIMAL_RC):
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(11.8, 5.4), gridspec_kw={"width_ratios": [1.10, 1.0], "wspace": 0.10}
        )
        for ax in (ax1, ax2):
            ax.set_xlim(-0.05, 10.05)
            ax.set_ylim(-0.05, 10.35)
            ax.axis("off")

        ax1.set_title(r"(a) ICHD Signal-Processing, $2\times 2$/GLS & Conformal Pipeline", fontsize=10.5, pad=10)
        boxes_a = [
            (0.10, 7.82, 9.80, 2.08, PAL_NAVY, "#F7F9FB",
             "1. Multimodal Cardiorespiratory Front-End (1 kHz)",
             "ECG (8\u201325 Hz QRS, R\u2013R \u00b125% screen) | Sternal SCG (1\u201340 Hz SVD $\\mathbf{w}_1$)\n"
             "Ear & Finger PPG (tangent $\\mathrm{PAT}, \\mathrm{PTT}, A_{\\mathrm{PPG}}$) | CNAP Ref."),
            (0.10, 5.32, 9.80, 2.08, PAL_NAVY, "#F7F9FB",
             "2. Causal $R$-Locked Ensemble & Baseline Log-Ratios",
             "10 s window (5 s hop), template xcorr lags ($\\Delta\\mathrm{PEP}, \\Delta\\mathrm{LVET}, \\Delta\\mathrm{PTT}$)\n"
             "Gain-invariant $x_k(t) = \\log(f_k(t)/f_k(t_0))$ rel. to 60\u2013150 s rest"),
            (0.10, 2.82, 9.80, 2.08, PAL_NAVY, "#F7F9FB",
             r"3. Chronotropically Decoupled Subspaces ($h \notin \mathcal{F}_{\mathrm{card}}$)",
             r"$\hat{z}_{\mathrm{sv}}(\mathcal{F}_{\mathrm{card}})$: SCG/optical (no HR) | $\hat{z}_{\mathrm{svr}}(\mathcal{F}_{\mathrm{vasc}})$: PTT, PPG" "\n"
             "$\\hat{z}_{\\mathrm{diff}}, \\hat{z}_{\\mathrm{map}}(\\mathcal{F}_{\\mathrm{joint}})$: Total pressure & flow\u2013resistance margin"),
            (0.10, 0.32, 9.80, 2.08, PAL_GREEN, "#F4F9F6",
             r"4. $2\times 2$ WLS / Full-Cov GLS & Conformal Control",
             r"Closure: $u_{\mathrm{co}}^* + u_{\mathrm{svr}}^* = u_{\mathrm{map}}^*$ ($<10^{-15}$ nat)" "\n"
             r"$\to$ Conformal Box $\mathcal{B}_{1-\alpha}(\mathbf{x})$ & Abstention (90% Cov.)"),
        ]
        for x0, y0, wb, hb, ec, fc, title, body in boxes_a:
            patch = FancyBboxPatch(
                (x0, y0), wb, hb, boxstyle="round,pad=0.04,rounding_size=0.05",
                facecolor=fc, edgecolor=ec, linestyle="-", linewidth=1.10
            )
            ax1.add_patch(patch)
            ax1.text(x0 + 0.22, y0 + hb - 0.22, title, fontsize=8.4, fontweight="bold", color="#111111", va="top")
            ax1.text(x0 + 0.22, y0 + 0.72, body, fontsize=7.4, color="#222222", va="center", linespacing=1.38)
        for yt, yb in [(7.76, 7.45), (5.26, 4.95), (2.76, 2.45)]:
            ax1.annotate("", xy=(5.0, yb), xytext=(5.0, yt),
                         arrowprops=dict(arrowstyle="-|>", lw=1.35, color=PAL_NAVY))

        n_excl_sess = b["n_sessions"] - b["n_episode_sessions"]
        n_excl_subj = b["n_subjects"] - b["n_episode_subjects"]
        n_excl_hc = n_excl_sess - n_excl_subj

        ax2.set_title("(b) Auricle Cohort & Episode Attrition Flow", fontsize=10.5, pad=10)
        p_enr = FancyBboxPatch(
            (0.10, 7.65), 9.80, 2.25, boxstyle="round,pad=0.04,rounding_size=0.05",
            facecolor="#F7F9FB", edgecolor=PAL_NAVY, linestyle="-", linewidth=1.10
        )
        ax2.add_patch(p_enr)
        ax2.text(0.32, 9.66, f"Enrolled Cohort: {b['n_subjects']} Subjects, {b['n_sessions']} Sessions",
                 fontsize=8.5, fontweight="bold", color="#111111", va="top")
        ax2.text(
            0.32, 8.42,
            f"{b['n_hc_subjects']} HC + {b['n_hfc_subjects']} HFC  |  {b['n_windows']:,} windows (10 s, 5 s hop)\n"
            f"{b['n_sessions']}/{b['n_sessions']} sessions contain valid CNAP SV reference",
            fontsize=7.5, color="#222222", va="center", linespacing=1.42,
        )

        ax2.annotate("", xy=(0.95, 3.85), xytext=(0.95, 7.58),
                     arrowprops=dict(arrowstyle="-|>", lw=1.5, color=PAL_NAVY))
        ax2.annotate("", xy=(1.82, 5.82), xytext=(0.95, 5.82),
                     arrowprops=dict(arrowstyle="-|>", lw=1.35, color=PAL_CRIMSON, ls="--"))

        p_exc = FancyBboxPatch(
            (1.88, 4.72), 8.02, 2.20, boxstyle="round,pad=0.04,rounding_size=0.05",
            facecolor="#FDF7F7", edgecolor=PAL_CRIMSON, linestyle="--", linewidth=1.10
        )
        ax2.add_patch(p_exc)
        ax2.text(2.10, 6.68, f"Excluded: {n_excl_sess} Rest-Only Sessions ({n_excl_hc} HC, {n_excl_subj} HFC)",
                 fontsize=8.2, fontweight="bold", color=PAL_CRIMSON, va="top")
        ax2.text(
            2.10, 5.45,
            f"{n_excl_sess} sessions contain Rest/NA labels only (0 stressor bouts):\n"
            "HC_1_1, 27_2, 38_2 | HFC_108_1, 124_1, 142_1",
            fontsize=7.4, color="#222222", va="center", linespacing=1.40,
        )

        p_act = FancyBboxPatch(
            (0.10, 0.32), 9.80, 3.45, boxstyle="round,pad=0.04,rounding_size=0.05",
            facecolor="#F4F9F6", edgecolor=PAL_GREEN, linestyle="-", linewidth=1.15
        )
        ax2.add_patch(p_act)
        ax2.text(0.32, 3.52, f"Active Analysis Cohort: {b['n_episode_subjects']} Subjects, {b['n_episode_sessions']} Sessions",
                 fontsize=8.5, fontweight="bold", color="#111111", va="top")
        ax2.text(
            0.32, 1.78,
            f"{b['n_episode_hc_subjects']} HC ({b['n_hc_episodes']} bouts) + {b['n_episode_hfc_subjects']} HFC ({b['n_hfc_episodes']} bouts) = {b['n_episodes']:,} Stressor Episodes\n"
            "Four Stressors: Breath-Hold, Valsalva, Dynamic Exercise, Cold Pressor\n"
            r"Evaluated via 5-Fold CV, SD-LOMO, S1$\to$S2 & $\mathrm{HC}\to\mathrm{HFC}$ Transfer",
            fontsize=7.4, color="#222222", va="center", linespacing=1.46,
        )

        _save(fig, "fig1_architecture")


def _plot_fig2_paradox(ep: pd.DataFrame) -> None:
    with plt.rc_context(MINIMAL_RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 5.0))
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.22, top=0.89, wspace=0.25)

        grid = np.linspace(-0.46, 0.58, 100)
        for c_map in (-0.2, 0.0, 0.2):
            ax1.plot(grid, c_map - grid, ls=":", lw=0.85, color="#b8b8b8", zorder=1)
        ax1.plot(grid, grid, ls="--", lw=1.35, color="#222222", label=r"Equipartition ($u_{\mathrm{svr}}=u_{\mathrm{co}}$)")
        ax1.axhline(0, color="#cccccc", lw=0.75)
        ax1.axvline(0, color="#cccccc", lw=0.75)

        offsets = {
            "exercise": (0.04, -0.09, "Exercise\n(+Flow, -SVR)"),
            "cold_pressor": (0.12, 0.07, "Cold Pressor\n(+SVR)"),
            "valsalva": (-0.36, 0.21, "Valsalva\n(+SVR, -SV)"),
            "breath_hold": (-0.39, -0.14, "Breath-Hold\n(Apnea)"),
        }
        for act_name, grp in ep.groupby("act_name"):
            k = str(act_name)
            s = _MANEUVER_STYLE[k]
            ax1.scatter(grp["dlog_co"], grp["dlog_svr"], s=16, alpha=0.25, color=s["color"],
                        edgecolors="none", marker=s["marker"], zorder=2)
            mx, my = float(grp["dlog_co"].mean()), float(grp["dlog_svr"].mean())
            ax1.scatter([mx], [my], s=125, color=s["color"], edgecolors="#111111", lw=1.35,
                        zorder=6, marker=s["marker"], label=s["label"])
            dx, dy, txt = offsets[k]
            ax1.annotate(
                txt,
                xy=(mx, my),
                xytext=(mx + dx, my + dy),
                fontsize=8.3,
                fontweight="bold",
                color=s["color"],
                bbox=dict(boxstyle="square,pad=0.18", facecolor="#ffffff", edgecolor=s["color"], alpha=0.94, lw=0.75),
                arrowprops=dict(arrowstyle="->", lw=0.95, color=s["color"], shrinkA=2, shrinkB=5),
                zorder=7,
            )

        ax1.set_xlim(-0.44, 0.56)
        ax1.set_ylim(-0.44, 0.42)
        ax1.set_xlabel(r"Flow Shift: $\Delta\log\mathrm{CO}$ (nat)")
        ax1.set_ylabel(r"Resistance Shift: $\Delta\log\mathrm{SVR}$ (nat)")
        ax1.set_title(rf"(a) 2D Hemodynamic State Plane ($N={len(ep):,}$)", pad=9)
        ax1.legend(loc="upper center", bbox_to_anchor=(0.48, -0.17), ncol=2, frameon=False, fontsize=8.6, columnspacing=0.9)

        ex = ep[ep["act_name"] == "exercise"]
        cp = ep[ep["act_name"] == "cold_pressor"]
        metrics = [
            r"$\Delta\log\mathrm{MAP}$",
            r"$\Delta\log\mathrm{CO}$",
            r"$\Delta\log\mathrm{SVR}$",
            r"$-\Delta\mathrm{PEP}/100\,\mathrm{ms}$",
        ]
        ex_m = [ex["dlog_map"].mean(), ex["dlog_co"].mean(), ex["dlog_svr"].mean(), -ex["d_pep_lag"].mean() / 100.0]
        cp_m = [cp["dlog_map"].mean(), cp["dlog_co"].mean(), cp["dlog_svr"].mean(), -cp["d_pep_lag"].mean() / 100.0]
        ex_se = [ex["dlog_map"].sem(), ex["dlog_co"].sem(), ex["dlog_svr"].sem(), ex["d_pep_lag"].sem() / 100.0]
        cp_se = [cp["dlog_map"].sem(), cp["dlog_co"].sem(), cp["dlog_svr"].sem(), cp["d_pep_lag"].sem() / 100.0]
        x = np.arange(len(metrics))
        w = 0.35
        ax2.bar(x - w / 2, ex_m, w, yerr=ex_se, capsize=3.5, facecolor=PAL_GREEN, edgecolor="#111111", lw=0.85,
                label=f"Exercise ($N={len(ex)}$, Flow-Driven)")
        ax2.bar(x + w / 2, cp_m, w, yerr=cp_se, capsize=3.5, facecolor=PAL_CRIMSON, edgecolor="#111111", lw=0.85,
                label=f"Cold Pressor ($N={len(cp)}$, SVR-Driven)")
        ax2.axhline(0, color="#222222", lw=0.85)
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics, fontsize=9.4)
        ax2.set_ylim(-0.25, 0.33)
        ax2.set_ylabel("Mean Log-Shift (nat / scaled ms)")
        ax2.set_title(r"(b) Opposite Pressor Origins ($+\Delta\mathrm{MAP}$)", pad=9)
        ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=8.6)
        _clean_grid(ax2)
        _save(fig, "fig2_pressor_paradox")


def _plot_fig3_pat_confound(ep: pd.DataFrame) -> None:
    with plt.rc_context(MINIMAL_RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 5.0))
        fig.subplots_adjust(left=0.075, right=0.98, bottom=0.21, top=0.89, wspace=0.36)
        acts = ["breath_hold", "valsalva", "cold_pressor", "exercise"]
        act_labels = ["Breath-Hold", "Valsalva", "Cold Pressor", "Exercise"]
        pat_v = [ep[ep["act_name"] == a]["d_ppgf_pat"].mean() for a in acts]
        pep_v = [ep[ep["act_name"] == a]["d_pep_lag"].mean() for a in acts]
        ptt_v = [ep[ep["act_name"] == a]["d_ppgf_ptt"].mean() for a in acts]
        x = np.arange(len(acts))
        w = 0.26
        ax1.bar(x - w, pat_v, w, facecolor=PAL_SLATE_LT, edgecolor="#111111", lw=0.8,
                label=r"$\Delta\mathrm{PAT}_{\mathrm{f}}$ ($\mathrm{R}\!\to\!\mathrm{PPG}$)")
        ax1.bar(x, pep_v, w, facecolor=PAL_OCHRE, edgecolor="#111111", lw=0.8,
                label=r"$\Delta\mathrm{PEP}_{\mathrm{SCG}}$ ($\mathrm{R}\!\to\!\mathrm{AO}$)")
        ax1.bar(x + w, ptt_v, w, facecolor=PAL_NAVY, edgecolor="#111111", lw=0.8,
                label=r"$\Delta\mathrm{PTT}_{\mathrm{f}}$ ($\Delta\mathrm{PAT}-\Delta\mathrm{PEP}$)")
        ax1.axhline(0, color="#222222", lw=0.85)
        ax1.text(2.0, 1.35, "+0.4 / +0.4 / 0.0 ms", ha="center", va="bottom", fontsize=7.8, color="#333333")
        ax1.set_xticks(x)
        ax1.set_xticklabels(act_labels, fontsize=9.3)
        ax1.set_ylim(-23.0, 12.5)
        ax1.set_ylabel("Timing Shift vs. Rest (ms)")
        ax1.set_title(r"(a) Cardiogenic $\mathrm{PAT}$ Confounding", pad=9)
        ax1.legend(loc="upper center", bbox_to_anchor=(0.48, -0.17), ncol=2, frameon=False, fontsize=8.4, columnspacing=0.8)
        _clean_grid(ax1)

        ep_tmp = ep.copy()
        ep_tmp["neg_pep"] = -ep_tmp["d_pep_lag"]
        ep_tmp["neg_weissler"] = -ep_tmp["d_weissler"]
        feats = [
            (r"$\Delta\log\mathrm{HR}$ (ECG)", "dlog_hr"),
            (r"$-\Delta(\mathrm{PEP}/\mathrm{LVET})$", "neg_weissler"),
            (r"$\Delta\mathrm{LVET}_{\mathrm{SCG}}$", "d_lvet_lag"),
            (r"$\Delta\log A_{\mathrm{SCG}}$", "d_scg_amp_log"),
            (r"$-\Delta\mathrm{PEP}_{\mathrm{SCG}}$", "neg_pep"),
            (r"$\Delta\log A_{\mathrm{PPG,f}}$", "dlog_ppgf_amp"),
            (r"$\Delta\log A_{\mathrm{PPG,ear}}$", "dlog_ppge_amp"),
        ]
        rs = [float(ep_tmp[c].corr(ep_tmp["dlog_sv"])) for _, c in feats]
        order = np.argsort(rs)
        feats_sorted = [feats[i][0] for i in order]
        rs_sorted = [rs[i] for i in order]
        y = np.arange(len(feats_sorted))
        cols = [PAL_CRIMSON if v < 0 else PAL_GREEN for v in rs_sorted]
        ax2.barh(y, rs_sorted, color=cols, edgecolor="#111111", lw=0.8, height=0.58)
        for yi, rv in zip(y, rs_sorted):
            offset = 0.012 if rv >= 0 else -0.012
            ha = "left" if rv >= 0 else "right"
            ax2.text(rv + offset, yi, f"{rv:+.2f}", va="center", ha=ha, fontsize=8.4, fontweight="bold", color="#222222")
        ax2.axvline(0, color="#222222", lw=0.85)
        ax2.set_yticks(y)
        ax2.set_yticklabels(feats_sorted, fontsize=9.4)
        ax2.set_xlim(-0.32, 0.50)
        ax2.set_xlabel(rf"Pearson $r$ with $\Delta\log\mathrm{{SV}}$ ($N={len(ep):,}$)")
        ax2.set_title(r"(b) Sorted Univariate $\Delta\log\mathrm{SV}$ Correlates", pad=9)
        _clean_grid(ax2, axis="x")
        _save(fig, "fig3_pat_confounding")


def _plot_fig4_benchmarks(res: dict) -> None:
    cfgs = ["PAT-Only", "ECG-Only", "PPG-Only", "Chest-ECG+SCG", "Ear-PPG-Only", "ICHD-Full"]
    cfg_labels = ["PAT", "ECG", "Finger\nPPG", "Chest\nSCG", "Ear\nPPG", "ICHD\nFull"]

    def _merged(base_d: dict, ctrl_d: dict) -> dict:
        out = dict(base_d)
        out["Ear-PPG-Only"] = ctrl_d["Ear-PPG-Only"]
        return out

    panels = [
        (
            "(a) 5-Fold Subject CV",
            _merged(res["exp2_cv"]["hybrid"], res["exp2_cv"]["controls"]),
            res["exp2_cv"]["hybrid"]["HR-Only"]["dom_auc"],
        ),
        (
            "(b) Subject-Disjoint LOMO",
            _merged(res["exp3_lomo"]["lomo"], res["exp3_lomo"]["controls"]),
            res["exp3_lomo"]["lomo"]["HR-Only"]["dom_auc"],
        ),
        (
            r"(c) Zero-Shot $\mathrm{HC}\to\mathrm{HFC}$",
            _merged(res["exp4_transfer"]["hybrid"], res["exp4_transfer"]["controls"]),
            res["exp4_transfer"]["hybrid"]["HR-Only"]["dom_auc"],
        ),
    ]
    with plt.rc_context(MINIMAL_RC):
        fig, axes = plt.subplots(1, 3, figsize=(10.8, 5.1), sharey=True)
        fig.subplots_adjust(left=0.062, right=0.99, bottom=0.15, top=0.81, wspace=0.10)
        x = np.arange(len(cfgs)) * 1.18
        w = 0.26
        bars_leg = []
        for idx_ax, (ax, (title, d_p, hr_auc)) in enumerate(zip(axes, panels)):
            r_sv = [d_p[c]["r_sv"] for c in cfgs]
            r_co = [d_p[c]["r_co"] for c in cfgs]
            auc = [d_p[c]["dom_auc"] for c in cfgs]
            b1 = ax.bar(x - w, r_sv, w, **_METRIC_BAR_PALETTE[0], lw=0.75, label=r"SV: $r(\Delta\log\mathrm{SV})$")
            b2 = ax.bar(x, r_co, w, **_METRIC_BAR_PALETTE[1], lw=0.75, label=r"CO: $r(\Delta\log\mathrm{CO})$")
            b3 = ax.bar(x + w, auc, w, **_METRIC_BAR_PALETTE[2], lw=0.75, label="Attribution ROC-AUC")
            l_hr = ax.axhline(hr_auc, color=PAL_CRIMSON, ls="--", lw=1.45, label="HR-Only AUC (dashed)")
            if idx_ax == 0:
                bars_leg = [b1, b2, b3, l_hr]
            ax.axhline(0, color="#222222", lw=0.75)
            ax.set_xticks(x)
            ax.set_xticklabels(cfg_labels, fontsize=8.8)
            ax.set_xlim(-0.58, x[-1] + 0.58)
            ax.set_ylim(-0.20, 1.04)
            ax.set_title(title, pad=9)
            _clean_grid(ax)
        axes[0].set_ylabel("Pearson $r$ / ROC-AUC")
        fig.legend(bars_leg, [b.get_label() for b in bars_leg],
                   loc="upper center", bbox_to_anchor=(0.50, 0.985), ncol=4, frameon=False, fontsize=9.2)
        _save(fig, "fig4_decomposition_benchmarks")


def _plot_fig5_conformal(ep: pd.DataFrame, res: dict) -> None:
    with plt.rc_context(MINIMAL_RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 5.1))
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.24, top=0.89, wspace=0.26)
        styles = [
            ("ICHD-Full", "ICHD-Full", "-", PAL_GREEN, 2.3),
            ("Ear-PPG-Only", "Ear-PPG", "--", PAL_PLUM, 2.0),
            ("PPG-Only", "Finger-PPG", (0, (4, 2)), PAL_OCHRE, 1.8),
            ("Chest-ECG+SCG", "Chest-SCG", "-.", PAL_NAVY, 1.8),
            ("PAT-Only", "PAT-Only", (0, (2, 2)), PAL_STEEL, 1.6),
            ("ECG-Only", "ECG-Only", ":", PAL_CRIMSON, 1.6),
        ]
        for cfg, disp, ls, col, lw in styles:
            c_d = res["exp5_conformal"][cfg]
            ax1.plot(np.array(c_d["coverages"]) * 100.0, np.array(c_d["accuracies"]) * 100.0,
                     ls=ls, color=col, lw=lw, label=f"{disp} (AURC {c_d['aurc']:.3f})")
        ax1.set_xlabel("Coverage (% Episodes Retained)")
        ax1.set_ylabel("Selective Attribution Accuracy (%)")
        ax1.set_xlim(100, 20)
        ax1.set_ylim(75, 98.0)
        ax1.set_title("(a) Conformal Risk-Coverage (AURC)", pad=9)
        ax1.legend(loc="upper center", bbox_to_anchor=(0.48, -0.17), ncol=2, frameon=False, fontsize=8.4)
        _clean_grid(ax1)

        p_oof = _oof_cv(ep, CONFIG_FEATURES["ICHD-Full"], kind="hybrid", enforce_identity=True)
        c_full = res["exp5_conformal"]["ICHD-Full"]
        hw_co, hw_svr, hw_diff = c_full["median_hw_co"], c_full["median_hw_svr"], c_full["median_hw_diff"]
        grid = np.linspace(-0.45, 0.55, 100)
        ax2.plot(grid, grid, "k--", lw=1.2, zorder=3)
        ax2.fill_between(grid, grid - (hw_co + hw_svr), grid + (hw_co + hw_svr),
                         color="#F2D894", alpha=0.30, zorder=1)
        ax2.fill_between(grid, grid - hw_diff, grid + hw_diff,
                         color="#B5C7D8", alpha=0.42, zorder=2)
        rng = np.random.default_rng(7)
        idx_sub = rng.choice(len(ep), size=180, replace=False)
        for i in idx_sub:
            cert_margin = abs(p_oof.vres_minus_vflow[i]) > hw_diff
            cert_box = abs(p_oof.vres_minus_vflow[i]) > (hw_co + hw_svr)
            is_flow = (ep["y_res_dom"].iloc[i] == 0)
            col = PAL_GREEN if is_flow else PAL_CRIMSON
            if cert_box:
                ax2.scatter(p_oof.dlog_co[i], p_oof.dlog_svr[i], s=50, alpha=0.95,
                            color=col, edgecolors="#111111", lw=1.0, marker="o", zorder=5)
            elif cert_margin:
                ax2.scatter(p_oof.dlog_co[i], p_oof.dlog_svr[i], s=30, alpha=0.88,
                            color=col, edgecolors="none", marker="o", zorder=4)
            else:
                ax2.scatter(p_oof.dlog_co[i], p_oof.dlog_svr[i], s=18, alpha=0.42,
                            color="#666666", marker="x", zorder=3)
        ax2.set_xlim(-0.35, 0.50)
        ax2.set_ylim(-0.38, 0.35)
        ax2.set_xlabel(r"Predicted $\Delta\log\mathrm{CO}$ (nat)")
        ax2.set_ylabel(r"Predicted $\Delta\log\mathrm{SVR}$ (nat)")
        ax2.set_title(
            f"(b) 1D Margin Cert. ({c_full['certified_margin_acc']*100:.1f}% Acc @ {c_full['certified_margin_frac']*100:.1f}% Cov)",
            pad=9,
        )
        leg_handles = [
            Line2D([0], [0], color="k", ls="--", lw=1.2, label=r"Equipartition ($u_{\mathrm{svr}}^*=u_{\mathrm{co}}^*$)"),
            Patch(facecolor="#B5C7D8", alpha=0.55, label="1D Margin Ambiguity"),
            Patch(facecolor="#F2D894", alpha=0.45, label="2D Box Ambiguity"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=PAL_GREEN, markeredgecolor="#111111", markersize=7, label="Certified Flow"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=PAL_CRIMSON, markeredgecolor="#111111", markersize=7, label="Certified SVR"),
            Line2D([0], [0], marker="x", color="w", markeredgecolor="#666666", markersize=6, label="Abstained"),
        ]
        ax2.legend(handles=leg_handles, loc="upper center", bbox_to_anchor=(0.48, -0.17), ncol=2, frameon=False, fontsize=8.2, columnspacing=0.9)
        _clean_grid(ax2)
        _save(fig, "fig5_conformal_risk_control")


def _plot_fig6_audit(res: dict) -> None:
    wm = res["exp2_cv"]["controls"]["cv_within_maneuver"]
    with open(RESULTS_DIR / "exp4_noise_ceiling_results.json") as f:
        nc = json.load(f)["variables"]

    with plt.rc_context(MINIMAL_RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 5.0))
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.22, top=0.89, wspace=0.25)

        man_keys = ["breath_hold", "valsalva", "exercise", "cold_pressor"]
        man_names = ["Breath-Hold", "Valsalva", "Exercise", "Cold Pressor"]
        man_lbls = [f"{lbl}\n($N={wm[k]['n_episodes']}$)" for k, lbl in zip(man_keys, man_names)]
        cfgs = ["ECG-Only", "PPG-Only", "Chest-ECG+SCG", "Ear-PPG-Only", "ICHD-Full"]
        c_disp = ["ECG", "Finger PPG", "Chest-SCG", "Ear-PPG", "ICHD-Full"]
        cols = [PAL_CRIMSON, PAL_OCHRE, PAL_STEEL, PAL_PLUM, PAL_GREEN]
        x = np.arange(len(man_keys)) * 1.25
        w = 0.17
        for j, (cfg, disp, col) in enumerate(zip(cfgs, c_disp, cols)):
            vals = [wm[m][cfg]["r_sv"] for m in man_keys]
            ax1.bar(x + (j - 2.0) * w, vals, w, color=col, edgecolor="#111111", lw=0.70, label=disp)
        ax1.axhline(0, color="#222222", lw=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(man_lbls, fontsize=8.9)
        ax1.set_xlim(-0.58, x[-1] + 0.58)
        ax1.set_ylim(-0.06, 0.76)
        ax1.set_ylabel(r"Within-Maneuver CV $r(\Delta\log\mathrm{SV})$")
        ax1.set_title("(a) Within-Maneuver SV Heterogeneity", pad=9)
        ax1.legend(loc="upper center", bbox_to_anchor=(0.48, -0.17), ncol=3, frameon=False, fontsize=8.4)
        _clean_grid(ax1)

        v_keys = ["dlog_co", "dlog_svr", "dlog_sv", "dlog_map"]
        v_lbls = [r"$\Delta\log\mathrm{CO}$", r"$\Delta\log\mathrm{SVR}$", r"$\Delta\log\mathrm{SV}$", r"$\Delta\log\mathrm{MAP}$"]
        ceil = [nc[v]["theoretical_ceiling_r_max"] for v in v_keys]
        ach = [nc[v]["model_r_achieved"] for v in v_keys]
        pcts = [nc[v]["fraction_of_ceiling_pct"] for v in v_keys]
        x2 = np.arange(len(v_keys))
        w2 = 0.35
        ax2.bar(x2 - w2 / 2, ceil, w2, facecolor=PAL_SLATE_LT, edgecolor="#111111", lw=0.85, label=r"CNAP Ceiling ($r_{\max}$)")
        ax2.bar(x2 + w2 / 2, ach, w2, facecolor=PAL_NAVY, edgecolor="#111111", lw=0.85, label=r"ICHD-Full ($r$)")
        for xi, av, pv in zip(x2, ach, pcts):
            ax2.text(
                xi + w2 * 0.56, av + 0.022, f"{pv:.1f}%",
                ha="center", va="bottom", fontsize=8.0, fontweight="bold", color=PAL_NAVY,
                bbox=dict(boxstyle="round,pad=0.10", facecolor="#ffffff", edgecolor="none", alpha=0.88),
            )
        ax2.set_xticks(x2)
        ax2.set_xticklabels(v_lbls, fontsize=9.4)
        ax2.set_ylim(0.0, 1.06)
        ax2.set_ylabel("Pearson Correlation ($r$)")
        ax2.set_title("(b) CNAP Retest Ceiling vs. Achieved Skill", pad=9)
        ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=8.8)
        _clean_grid(ax2)
        _save(fig, "fig6_within_maneuver_ceilings")


def plot_all() -> None:
    ep = pd.read_pickle(TABLES_CACHE / "episodes.pkl")
    with open(RESULTS_DIR / "ichd_results.json") as f:
        res = json.load(f)

    _plot_fig1_architecture(res)
    _plot_fig2_paradox(ep)
    _plot_fig3_pat_confound(ep)
    _plot_fig4_benchmarks(res)
    _plot_fig5_conformal(ep, res)
    _plot_fig6_audit(res)
    print("Generated color figures (fig1..fig6) in figures/.")


if __name__ == "__main__":
    plot_all()

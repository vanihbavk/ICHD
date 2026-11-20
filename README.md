# Identity-Constrained Hemodynamic Decomposition (ICHD)

**Preprint (Zenodo):** [https://doi.org/10.5281/zenodo.24342344](https://doi.org/10.5281/zenodo.24342344)

> [!WARNING]
> **Large Dataset Notice (~18 GB on Disk):**
> Raw physiological waveforms are **not** stored in this GitHub repository. Downloading the complete raw datasets (PhysioNet Auricle v1.0.0: **7.3 GB ZIP / 16.0 GB unpacked** + VitalDB 30-case cache: **~791 MB**) requires **~17–18 GB of disk space** and significant download time (20–60+ minutes). All precomputed benchmark JSONs, VitalDB evaluation windows, and figures are included directly in `results/` and `figures/`; see [Data Access & Reproducibility](#data-access--reproducibility) below.

---

## Summary

Conventional cuffless blood-pressure (BP) monitors treat arterial pressure as a one-dimensional scalar inferred from Pulse Arrival Time ($\mathrm{PAT}$). By Ohm's law of the systemic circulation ($\mathrm{MAP} \approx \mathrm{CO} \times \mathrm{SVR} = \mathrm{HR} \times \mathrm{SV} \times \mathrm{SVR}$), identical rises in Mean Arterial Pressure ($\mathrm{MAP}$) can originate from opposite mechanisms: a cardiac-output ($\mathrm{CO}$) surge during exercise versus a systemic vascular resistance ($\mathrm{SVR}$) surge during cold pressor or Valsalva strain.

**Identity-Constrained Hemodynamic Decomposition (ICHD)** focuses on what wearable waveforms genuinely add beyond heart rate alone:
- **Stroke-Volume Recovery Beyond Heart Rate:** Multimodal **ICHD-Full** tracks baseline-referenced stroke-volume changes ($\Delta\log\mathrm{SV}$) at $r = 0.637$ in 5-fold subject-disjoint CV (vs. $r = 0.262$ for `HR-Only`), reaching **98.6% of the CNAP cross-session retest ceiling** ($r_{\max} = 0.646$).
- **Auricular (Ear) PPG Performance:** **Ear-PPG-Only** ($r = 0.629$ CV; $r = 0.452$ LOMO vs. $0.349$ finger PPG, $p = 0.002$; $r = 0.607$ in strictly subject- and session-disjoint Session 1 $\to$ Session 2 temporal evaluation) matches multimodal performance without co-location with the CNAP finger cuff.
- **Honest Boundary Conditions:** Under CNAP's $\mathrm{CVP}\approx 0$ convention, true $\mathrm{SVR}$ and $-\mathrm{CO}$ are strongly collinear ($r = 0.834$), leaving partial $\mathrm{SVR}$ skill given true $\mathrm{CO}$ near zero ($r_{\mathrm{partial}} = -0.002$) and $\Delta\log\mathrm{MAP}$ at 36.8% of ceiling ($r = 0.257$). Binary attribution is largely driven by heart rate (`HR-Only` AUC $= 0.840$; 1-feature `HR-Only` AUC $= 0.841$).
- **External Surgical Generalization Limits (VitalDB):** Across $N = 35$ surgical cases ($42{,}650$ windows, $N_{\mathrm{eff}} \approx 359$), frozen 50-feature finger PPG collapses ($r = -0.035$ vs. $r = +0.226$ `HR-Only`, $p = 0.050$) due to a $11.1\text{--}14.7\times$ raw optical gain shift and within-case sign inversion of PPG amplitude/slope features ($r_{\mathrm{Auricle}} = +0.356 \to r_{\mathrm{VitalDB,mean}} = -0.097$), while `PAT-Finger` ($r = +0.302$, median $r = +0.414$) tracks preserved HR--SV coupling without significantly beating heart rate alone ($p = 0.216$).

---

## Repository Structure

- `src/ichd/`: Core DSP (`signals.py`), Auricle feature extraction (`extract.py`), episode/baseline builder (`episodes.py`), decoupled estimators & $2\times 2$ reconciliation (`model.py`), statistical bootstraps (`stats.py`), master evaluation (`evaluate.py`), conformal calibration (`conformal.py`), shared VitalDB utilities (`external.py`), and plotting (`plots.py`).
- `experiments/`: Hypothesis-driven diagnostic, SAGER sex-stratified, and external validation scripts (`exp1_lvef_anchor.py` through `exp11_sager_sex_audit.py`).
- `results/`: Canonical machine-readable JSON outputs and cached VitalDB window evaluations (`vitaldb_windows.csv.gz`).
- `scripts/`: `download_data.py` auto-fetches Open Access PhysioNet Auricle and VitalDB datasets and verifies `data/CHECKSUMS.txt`.
- `tests/`: Unit and integration test suite (`tests/test_ichd.py`).
- `figures/`: Vector PDF figures (`fig1_architecture.pdf` through `fig7_temporal_vitaldb.pdf`).

---

## Datasets & Storage Breakdown

| Dataset / Artifact | Source & License | Contents & Channels | Transfer Size | Disk Footprint | Local Path |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PhysioNet Auricle v1.0.0** | [PhysioNet](https://physionet.org/content/auricle/1.0.0/) (`10.13026/swme-vp69`, Open Access ODC-By) | 70 subjects (42 HC, 28 HFC), 107 WFDB sessions @ 1 kHz (ECG, 3-axis SCG, Ear/Finger PPG, PCG, CNAP BP/SV/CO/SVR) | **7.3 GB** (ZIP) | **16.0 GB** | `data/raw/auricle/` |
| **VitalDB Surgical Cache** | [VitalDB API](https://vitaldb.net/) (`vitaldb` Python package, Open Access) | 36 pre-registered surgical cases (35 valid $\ge 30$ min) resampled to 1 kHz (`SNUADC/ECG_II`, `SNUADC/PLETH`, `EV1000/SV`) | **~950 MB** | **950 MB** | `data/interim/vitaldb_cache/` |
| **ICHD Derived Tables** | Generated locally by `make extract episodes` | `episodes.pkl` ($N=1{,}018$ bouts across 67 active subjects, 2.1 MB) + `windows.pkl` ($N=33{,}182$ 10 s windows, 68 MB) | **—** (local) | **70.1 MB** | `data/interim/ichd_tables/` |
| **VitalDB Extracted Windows** | Included in repository (`make external`) | $42{,}650$ baseline-referenced 10 s surgical evaluation windows + frozen predictions | **25 MB** | **25 MB** | `results/vitaldb_windows.csv.gz` |

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# Optional (only needed to re-download raw VitalDB surgical cases):
pip install -e ".[external]"
```

---

## Data Access & Reproducibility

1. **Zero-Data Unit & Invariant Tests (~5 s):**
   ```bash
   make test
   ```
   Verifies DSP, $2\times 2$ Gauss–Markov closure, causal baseline construction, conformal calibration, and `results/ichd_results.json` invariants using synthetic waveforms and the precomputed JSONs in `results/`.

2. **Full Raw-to-Benchmark Reproduction (~15–20 min + Download Time):**
   Both [PhysioNet Auricle v1.0.0](https://physionet.org/content/auricle/1.0.0/) (Open Access, ODC-By) and [VitalDB](https://vitaldb.net/) are publicly accessible without credentialed approval. Auto-download and verify all raw datasets (~8 GB transfer, **~17–18 GB unpacked**), then run the full extraction, evaluation, experiment, and plotting pipeline:
   ```bash
   make fetch-data
   make pipeline
   ```
   Once `make extract episodes` has built `data/interim/ichd_tables/`, individual benchmark stages can be re-run in ~60–90 s via `make evaluate experiments plots`. Manual downloads into `data/raw/auricle/` can also be verified directly via:
   ```bash
   cd data && sha256sum -c CHECKSUMS.txt
   ```

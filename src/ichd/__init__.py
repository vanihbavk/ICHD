"""ICHD: Identity-Constrained Hemodynamic Decomposition for attributing
blood-pressure perturbations to cardiac-output vs. vascular-resistance origin.

Pipeline
--------
signals.py   low-level DSP: R-peak detection, ensemble averaging, fiducials
extract.py   per-session window-level feature + reference extraction (Auricle)
episodes.py  maneuver episodes, local baselines and log-delta targets
model.py     hemodynamic-vector estimators (identity-constrained ridge, GBM)
conformal.py risk-controlled attribution with abstention
stats.py     subject-clustered bootstrap and partial-correlation utilities
evaluate.py  benchmark suite (CV, SD-LOMO, HC->HFC, conformal, controls)
external.py  shared VitalDB surgical validation utilities
plots.py     publication vector figure generation
"""

__version__ = "1.0.0"

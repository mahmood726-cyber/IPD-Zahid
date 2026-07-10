# IPD-Zahid — IPD Quantile Meta-Analysis (IPD-QMA)

A two-stage **Individual Participant Data Quantile Meta-Analysis** for detecting
treatment effects that vary across the outcome distribution (not just the mean).

- **Stage 1** — quantile regression (`statsmodels.QuantReg`) estimates quantile
  treatment effects (QTEs) at each τ within every study, with bootstrap SEs that
  preserve cross-quantile covariance.
- **Stage 2** — DerSimonian–Laird random-effects pooling across studies, with
  optional Hartung–Knapp–Sidik–Jonkman (HKSJ) small-*k* correction, Cochran's
  *Q* / *I²* / τ² heterogeneity diagnostics, and prediction intervals.
- A **slope test** (Q90 − Q10) formally tests whether the treatment effect
  changes across quantiles, alongside a log-variability-ratio (lnVR) contrast and
  the standard mean-difference meta-analysis for comparison.

The worked application is an NHANES systolic blood-pressure (SBP) analysis by
antihypertensive medication use, plus a Monte-Carlo simulation study
(Type-I-error and power).

## Layout

```
ipd_qma_v2/
  ipd_qma.py            # core: IPDQMA class + simulate_location_scale()
  run_analysis.py       # real-data (NHANES) + simulation driver -> figures/tables
  run_simulation.py     # full simulation study
  run_simulation_fast.py# reduced-B quick simulation
  fetch_nhanes.py       # NHANES download / IPD preparation
  build_docx.py         # renders the corrected markdown draft to a .docx
  data/                 # nhanes_combined.csv
  output/               # generated figures (fig1..fig7) and tables
  requirements.txt      # pinned runtime dependencies
tests/                  # pytest suite (smoke + statistical-core regressions)
```

## Install

```bash
pip install -r ipd_qma_v2/requirements.txt
```

Requires Python 3.13. Core dependencies: numpy, pandas, scipy, statsmodels,
matplotlib (and python-docx only for `build_docx.py`).

## Run

```bash
# Full analysis: NHANES real-data + simulation, writing to ipd_qma_v2/output/
python ipd_qma_v2/run_analysis.py

# Build the submission Word document from the corrected markdown draft
python ipd_qma_v2/build_docx.py
```

## Test

```bash
python -m pytest -q
```

## Note on the E156 submission

`E156-PROTOCOL.md` and `e156-submission/` hold the micro-paper submission
metadata for this project. The `e156-submission/paper.json` body is still a
**DRAFT** (`submitted: false`); the canonical abstract lives in
`E156-PROTOCOL.md`.

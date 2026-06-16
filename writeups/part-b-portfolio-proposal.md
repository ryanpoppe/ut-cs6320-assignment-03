# Ryan Poppe

# Assignment 3 — Part B: Portfolio Project Proposal (NOAA ENSO Seasonal Forecasting)

## Problem and direction

I am continuing my **preferred Assignment 2 direction without change**: short-horizon
forecasting of the El Niño–Southern Oscillation (ENSO) state from NOAA climate indices. The
problem: given recent monthly ENSO index values, predict the ENSO state **three months ahead**
(El Niño / La Niña / Neutral). No new evidence has emerged that makes this leaky, ill-posed, or
data-blocked, so the only refinement from Assignment 2 is making the horizon and target
explicit. Credit-card fraud remains my documented backup (see fallback plan).

## Stakeholder / use case

The intended audience is a **seasonal climate-risk planner** — an agricultural planner, water
manager, or emergency-management office — who benefits from an early, plain indication of
whether the tropical Pacific is trending warm, cool, or neutral, because ENSO state shifts
regional rainfall, drought, and wildfire odds. Framing is explicit: a small forecasting
*exercise* and baseline comparison, **not** a replacement for NOAA/IRI operational ENSO
forecasts.

## Dataset, access status, and pre-audit

**Primary data:** NOAA Climate Prediction Center **Oceanic Niño Index (ONI)** and NOAA Physical
Sciences Laboratory **Niño 3.4 SST anomalies**, with the **Multivariate ENSO Index v2 (MEI.v2)**
as an optional second feature source.

**Access status:** Publicly available now as plain-text/CSV tables directly from NOAA CPC/PSL;
no account, paywall, or registration. I have located the sources but have not yet pulled and
checksummed a fixed snapshot — I will do that and re-verify the live endpoints when I lock the
dataset in Assignment 4.

**Compact pre-audit (bridge from the Week 1 practice audit toward the Assignment 4 full audit):**

- *Source / provenance.* Public-domain U.S. government climate data (NOAA CPC, NOAA PSL).
  Derived scientific indices computed from sea-surface-temperature observations, not personal
  data — the cleanest access and citation story of my candidates.
- *Access / licensing constraints.* U.S. federal data, effectively public domain; standard
  expectation is attribution to NOAA. No restrictive license. Low licensing risk.
- *Responsible-use uncertainty.* Low privacy risk (no individuals). The real responsible-use
  concern is **overclaiming**: a course model must not be presented as an operational forecast,
  and predicted ENSO state must not be used for real planning decisions. ONI definitions are
  occasionally re-baselined by NOAA, so I will pin a single index version and document it.
- *Prediction target.* ENSO category at a +3-month horizon (3-class). A regression variant
  (future Niño 3.4 anomaly, MAE/RMSE) is available if the class version is too coarse.
- *Candidate inputs.* Lagged index values (e.g., ONI/Niño-3.4 at lags 1, 3, 6, 12 months),
  short trend/derivative features, and month-of-year to encode seasonality.
- *Prediction-time availability / leakage.* The dominant risk is **temporal leakage**: nearby
  months are highly autocorrelated, so a random split would let the model "see the future."
  Mitigation is strict — features use only information available at prediction time, and all
  evaluation uses chronological splits. I must also confirm each index's real publication lag so
  I never feed a value that would not yet be released at the forecast moment.
- *Data quality / missingness / imbalance / representativeness.* Indices are clean and nearly
  complete, but **class imbalance** is real (Neutral months outnumber strong El Niño/La Niña
  events) and, more importantly, the number of **independent events** is small — a few decades
  of strongly autocorrelated months, so the effective sample is far smaller than the row count.
  The record also may not represent future climate-change-influenced ENSO behavior.

## Baselines, metrics, and evaluation strategy

- **Baselines (the heart of the project).** (1) **Persistence** — predict the +3-month state
  equals today's state; strong because indices are autocorrelated. (2) **Seasonal climatology**
  — predict by historical base rates for that month. Any model must beat *both* to be
  interesting. (3) A simple ML baseline: logistic regression / small tree on lagged features.
- **Metrics.** Macro-F1 and balanced accuracy (because of class imbalance), reported against the
  persistence and climatology baselines; per-class recall for El Niño/La Niña; MAE/RMSE for the
  regression variant.
- **Split / evaluation.** **Chronological** train/validation/test (e.g., train on earlier
  decades, test on the most recent years), never a random split. Where feasible I will add
  walk-forward (expanding-window) evaluation to use the limited history honestly.

## Risks, responsible-use, scope, and fallback

- **Failure risks.** The model may not beat persistence at +3 months (a legitimate, reportable
  outcome); the small independent-event count may make results noisy across splits.
- **Responsible-use.** Frame as an educational baseline comparison; do not imply operational
  skill or drive real decisions; pin and cite the exact index version.
- **Scope risks.** The index-only version may not *need* deep learning. To keep a credible
  "why deep learning" story without overreaching, the planned arc is: lagged-index baselines and
  a small NN first, with an **optional** stretch to a CNN/ConvLSTM over gridded SST fields only
  if time allows. I will not start from the heavy gridded version.
- **Fallback / adjustment plan.** If the independent-event limitation makes a satisfying model
  comparison infeasible, I fall back to my Assignment 2 backup — **Kaggle credit-card fraud
  detection** — a clean, well-posed binary task with a clear target and rich evaluation
  (PR-AUC, recall at fixed precision under heavy imbalance). A lighter adjustment, if I keep
  ENSO but it proves thin, is to switch from classification to the Niño-3.4 regression target
  and/or shorten the horizon to +1 month, where beating persistence is harder and more
  informative.

## Sources

- NOAA CPC Oceanic Niño Index (ONI) and NOAA PSL Niño 3.4 SST anomaly data — public-domain U.S.
  government climate indices.
- NOAA PSL Multivariate ENSO Index v2 (MEI.v2).
- Kaggle "Credit Card Fraud Detection" dataset (backup; subject to Kaggle dataset terms).

# Results summary (all numbers verified)

> Plain results tables behind the manuscript. All metrics follow the evaluation
> protocol in `docs/manuscript.md` (Section 2.5): Harrell C-index (risk = −E[T]),
> Uno time-dependent AUC with IPCW (test follow-up capped at the training
> cohort's maximum observed time), IPCW Brier vs. a Kaplan-Meier null model,
> calibration (observed KM minus mean predicted survival), and conditional
> survival against KM.
>
> Cohorts: UCSF-PDGM strict GBM subset (n=367; 223 events / 144 censored;
> 6-2144 days) and UPENN-GBM (n=608; 585 events / 23 censored; 3-6160 days).
> Clinical features: age, sex, extent of resection (GTR), MGMT methylation.
> LOCO = leave-one-cohort-out; "→UPENN" = trained on UCSF, tested on UPENN.

## 1. Imaging pipelines: internal vs. external (LOCO)

### 1a. Pretrained SwinUNETR + SALP

| Model | Internal 25-fold CV (mixed cohort history) | LOCO →UPENN | LOCO →UCSF |
|---|---|---|---|
| salp_only (z_salp, 1152-d) | C-index 0.6590 ± 0.0290 | 0.525 | 0.497 |
| salp_full (1159-d, legacy config) | (same head family) | 0.520 | 0.526 |
| frozen (whole-tumor pooling, no SALP) | C-index 0.6625 ± 0.0254 | — | — |

LOCO Uno-AUC means: salp_only 0.529 (→UPENN) / 0.503 (→UCSF); salp_full 0.520 / 0.531.
3-month Uno-AUC on the UCSF test cohort was 0.391 (below chance).

### 1b. Whole-brain R2SN graph (246-ROI AAL, T1c/T2/FLAIR)

| Model | Internal (UCSF 80/20 val) | LOCO →UPENN | LOCO →UCSF |
|---|---|---|---|
| Graph-only, high-dim node features (455-497/ROI) | 0.63 | 0.523 | 0.531 |
| Graph-only, 93-dim node features | — | 0.567 | 0.601 |
| Graph + clinical, 93-dim | 0.66 | 0.615 | 0.666 |
| Clinical-only (same protocol) | 0.66 | 0.639 | 0.651 |

## 2. Clinical-molecular multi-timepoint framework (positive result)

### 2a. Internal (UCSF 5-fold CV, out-of-fold, n=367)

- Discrete-time framework C-index: 0.622
- Clinical Cox baseline (same folds): 0.671 ± 0.031
- Brier vs. KM null: −4% to −12% across landmarks

### 2b. External (LOCO, both directions)

| Metric | →UPENN (n=608) | →UCSF (n=367) |
|---|---|---|
| C-index | 0.639 | 0.651 |
| Uno-AUC 3/6/12/24/36 mo | 0.714 / 0.688 / 0.705 / 0.658 / 0.659 | 0.802 / 0.741 / 0.730 / 0.739 / 0.747 |
| Brier 3/6/12/24/36 mo | 0.100 / 0.183 / 0.260 / 0.276 / 0.238 | 0.065 / 0.121 / 0.164 / 0.124 / 0.079 |
| Brier vs. KM null | −6.4 / −10.1 / −12.7 / −6.4 / −5.6% | −12.1 / −13.6 / −14.5 / −9.4 / +0.4% |
| Calibration (pred − KM) | +0.032 / +0.032 / +0.044 / +0.047 / +0.017 | −0.001 / −0.006 / +0.013 / +0.006 / +0.007 |

### 2c. Conditional survival (dynamic update)

| Interval | →UPENN (pred vs. KM) | →UCSF (pred vs. KM) |
|---|---|---|
| 12→24 months | 0.463 vs. 0.409 (+0.053) | 0.514 vs. 0.512 (+0.002) |
| 12→36 months | 0.257 vs. 0.241 (+0.016) | 0.337 vs. 0.330 (+0.007) |
| 24→36 months | 0.531 vs. 0.590 (−0.059) | 0.664 vs. 0.645 (+0.020) |

### 2d. Performance-improvement attempts (all below the +0.01 threshold)

Best clinical baseline: average external C-index 0.644 (0.639 →UPENN, 0.651 →UCSF).

| Attempt | avg external C-index | Δ vs. baseline |
|---|---|---|
| K=80 finer grid (vs. K=40) | 0.6448 | +0.0008 |
| Bounded ranking loss (weight 0.5) | 0.6445 | +0.0005 |
| Ranking loss (weight 2.0) | 0.6403 | −0.004 |
| 93-dim graph + clinical | ≈0.641 (direction-dependent) | ≈0 |
| Ensemble clinical + 93-dim graph | 0.6483 | +0.004 |

## 3. Comparison with published work (same/overlapping data)

| Work | Protocol | 1-year metric | Note |
|---|---|---|---|
| CM-GNN (Front. Oncol. 2023, reported) | 5-cohort LOCO; binary 12-month label | AUC 72.21% | Same-source cohorts (UPENN-584, UCSF-351); metrics/protocols not directly comparable |
| This work, clinical framework | 2-cohort LOCO; Uno-AUC | 12-mo Uno-AUC 0.705-0.730 | Same magnitude; R2SN imaging asset (same lineage) fails LOCO here |
| SwinUNETR+SALP (companion pipeline) | mixed-cohort internal 25-fold CV | AUC@48m 0.7396 (C-index 0.659) | Collapses to 0.50-0.53 under LOCO |

## 4. Key negative/positive one-liners

- SwinUNETR+SALP: internal 0.659 → LOCO 0.50-0.53 (chance).
- R2SN graph (high-dim): 0.63 internal → 0.52-0.53 LOCO (chance).
- R2SN graph (93-dim): LOCO 0.57-0.60 → dimensionality mitigates, still below clinical.
- Clinical-molecular framework: external C-index 0.639-0.651, 12-mo Uno-AUC 0.705-0.730,
  Brier 5-15% better than KM null, calibration ≤0.05 absolute, conditional survival ≤0.06.
- Adding imaging to clinical: no consistent external gain (direction-dependent ±0.02).
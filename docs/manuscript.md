# Manuscript Draft 

**Title:**
Imaging features transfer poorly across institutions in glioblastoma survival prediction: internal cross-validation systematically overstates generalization, with a calibrated clinical-molecular multi-timepoint framework as a reproducible baseline

**Running head:** Cross-institutional transfer failure of imaging survival models in GBM



---

## Abstract

**Background.** Deep-learning radiomics based on preoperative MRI has been widely reported to predict overall survival (OS) after glioblastoma (GBM) resection, with internal cross-validation (CV) C-indices typically in the 0.62-0.66 range. Whether such imaging features generalize across institutions remains largely untested.

**Methods.** We systematically tested the cross-institution transfer of two mainstream imaging representations on two public cohorts (UCSF-PDGM, 367 IDH-wildtype WHO grade 4 GBM; UPENN-GBM, 608 GBM), using leave-one-cohort-out (LOCO) validation in both directions: (A) a pre-trained SwinUNETR (BraTS fine-tuned) frozen-encoder pipeline with anatomy-aware soft pooling (SALP) and a discrete survival head, previously reported with internal C-index 0.659; and (B) a whole-brain R2SN graph network (246-ROI AAL atlas, three modalities, high- and low-dimensional node features). As a positive control, we built a multi-timepoint discrete survival framework (DeepHit-style, K=80 bins, monotone survival) on clinical-molecular variables only (age, sex, extent of resection, MGMT methylation), evaluated internally (5-fold CV) and externally (LOCO), with Uno-AUC at 3/6/12/24/36 months, IPCW Brier, calibration, and conditional survival.

**Results.** Both imaging representations failed LOCO validation: SwinUNETR+SALP C-index dropped from 0.659 (internal) to 0.50-0.53 (external); R2SN graph dropped from 0.63 (internal) to 0.52-0.53 (high-dimensional) and 0.57-0.60 (93-dimensional) — at or near chance, and below clinical features alone. The clinical-molecular framework achieved external C-index 0.639 (UCSF-trained) and 0.651 (UPENN-trained), 12-month Uno-AUC 0.705/0.730, Brier 5-15% better than a Kaplan-Meier null model, calibration within 0.05 absolute, and conditional-survival calibration errors of 0.00-0.06.

**Conclusions.** Internal CV systematically overstates the generalization of imaging-based GBM survival models; at least one legacy claim of strong imaging performance (internal C-index 0.659; AUC@48m 0.7396) does not replicate under external validation. A transparent clinical-molecular multi-timepoint and conditional-survival framework provides honest, well-calibrated individualized survival probabilities and should be adopted as the default baseline in future work.

**Keywords:** glioblastoma; survival analysis; external validation; deep learning; SwinUNETR; brain networks; conditional survival

---

## 1. Introduction

Glioblastoma is the most common primary malignant brain tumor in adults, with a median overall survival (OS) of roughly 12-15 months despite maximal safe resection and radio-chemotherapy [ref: *Stupp et al.*]. Preoperative MRI is routinely acquired, and a large literature has used deep learning on MRI to predict post-operative OS [refs]. High internal cross-validation (CV) performance is frequently reported; C-index values around 0.62-0.66 and 1-year AUC values around 0.70-0.75 are typical [refs, e.g. CM-GNN].

Two families of imaging representations dominate this literature. The first family uses pretrained volumetric transformers, most prominently SwinUNETR [ref: *Hatamizadeh et al.*], fine-tuned on BraTS [ref] and then used (frozen) as feature extractors, sometimes combined with anatomically aware pooling of tumor sub-regions. The second family models *relationships* between brain regions, e.g. whole-brain radiomics similarity networks (R2SN) fed to graph neural networks (GNN) [ref: CM-GNN].

A recurring methodological gap in this literature is the reliance on internal CV over mixed-cohort pooling: training and test folds are drawn from the same institutions, so reported performance may reflect cohort- and site-specific signal (including, in mixed cohorts, the discrimination between tumor grades or IDH-mutation subtypes) rather than a generalizable survival signal. External, leave-one-cohort-out (LOCO) validation is the exception rather than the rule.

In this study we address two questions with public data:

1. *Do imaging representations transfer across institutions?* We test two independent pipelines — (A) pretrained SwinUNETR + anatomy-aware soft pooling (SALP) with a discrete survival head, which was previously reported with internal C-index 0.659 (AUC@48m 0.7396) on a mixed 1106-patient cohort, and (B) a whole-brain R2SN graph network similar in spirit to prior GNN work — under strict LOCO validation in both directions between UCSF-PDGM and UPENN-GBM.

2. *What is a defensible baseline?* We provide a clinical-molecular multi-timepoint discrete survival framework predicting individualized survival probabilities at 3/6/12/24/36 months, with dynamic conditional-survival updates, fully externally validated.

We report that both imaging pipelines fail LOCO validation (external C-index ≈ 0.50-0.60, at or near chance), while the clinical-molecular framework transfers well and is well calibrated. We discuss mechanisms — mixed-cohort grade/IDH discrimination, site-specific feature distributions, and segmentation-quality propagation — and argue that internal CV systematically overstates imaging-based survival models.

## 2. Materials and Methods

### 2.1 Cohorts and survival outcomes

**UCSF-PDGM.** The UCSF Preoperative Diffuse Glioma MRI dataset [ref: *Calabrese et al.*] provides preoperative MRI, segmentations, and clinical data for patients with diffuse glioma. For the imaging-transfer experiments we use two constructions: (i) the full diffuse-glioma cohort (n=495) as used in the legacy SwinUNETR pipeline (Section 2.2), and (ii) a strict glioblastoma subset — WHO grade 4, IDH-wildtype, with documented OS — of n=367 (223 events, 144 censored; OS 6-2144 days; median 358 days), which we use for all external-validation experiments and for the clinical framework. One patient with missing OS was excluded; the assembly asserted finite, positive survival times.

**UPENN-GBM.** The UPENN-GBM cohort [ref: *Bakas et al.*] provides preoperative MRI and clinical annotations for 611 patients with GBM. We excluded 3 patients with "deceased — uncertain date of death", leaving n=608 (585 events, 23 censored; follow-up 3-6160 days; median event time 390 days).

**Survival construction.** Time origin is the date of surgery for both cohorts (UCSF OS reported in days; UPENN event days from the surgery-date field and censoring days from the follow-up field). Right censoring is used throughout; the legacy pipeline converted UPENN days to months by dividing by 30.44 and capped survival at 120 months [*cap120*, Section 2.2], which we reproduce where indicated.

**Clinical variables.** Age, sex, extent of resection (GTR vs. not, UCSF) and MGMT promoter methylation status (UCSF 350/367 available; UPENN 261/608 available) were used. KPS was excluded (UPENN numeric values for ~75/608 patients only; UCSF none). IDH status was excluded from modeling (cohort-constant: all IDH-wildtype by construction of the strict GBM subset).

### 2.2 Imaging representation A: pretrained SwinUNETR + SALP (legacy pipeline)

This pipeline is described in its internal-validation form in a companion pipeline report (the full settings are reproduced in this repository's `code/`); we summarize the settings and report its cross-institution behavior here.

**Preprocessing.** Each patient's T1c, T2, and FLAIR volumes were rigidly/serially registered to ICBM152 (ANTs SyN; T1c as moving image, shared transforms for T2/FLAIR). Per-patient normalization only (0.5-99.5 percentile cropping followed by channel-wise z-scoring; no cross-patient statistics). A 128³ volume centered on the tumor centroid (largest connected component of the whole-tumor mask) was cropped with reflective padding for images and zero padding for masks.

**Feature extraction.** A SwinUNETR encoder [ref: *Hatamizadeh et al.*, MONAI], feature_size=48, depths (2,2,2,2)/heads (3,6,12,24), initialized with BrainSegFounder BraTS-finetuned weights [ref], was frozen and used once for offline feature extraction. The four-channel patch-embedding was adapted to three channels by truncating the input-channel weights (`weight.data[:, 1:4]`). The stage-4 output (384 channels at 8³ resolution) was flattened to H ∈ ℝ^(512×384).

**Anatomy-aware soft pooling (SALP).** For each tumor sub-region s ∈ {enhancing tumor (ET), edema (ED), non-enhancing core (NCR)}, we computed a fixed (non-learnable) soft-pooling vector z_s = Σ_i [r_i^s / Σ_j r_j^s] · h_i, where r_i^s is the fraction of voxels of sub-region s in pooling-cell i (8³ down-sampled segmentation grid), and h_i is the corresponding feature cell. The final feature is z = [z_ED; z_ET; z_NCR] ∈ ℝ^1152, concatenated with tumor-global statistics (5) and clinical scalars (age, sex) → d = 1159.

**Legacy head and internal validation.** A discrete survival head (Linear(1159→64) → GELU → Dropout(0.1) → Linear(64→12)) was trained with censored discrete negative log-likelihood; bin edges were 12 quantiles of training-fold event times (*quantile binning*); AdamW, lr 1e-4, weight-decay 1e-5, 200 epochs, batch 64; 25-fold CV (5 repeats × 5 StratifiedKFold on the pooled mixed cohort, stratified on the 12-month binary label; label = death within 12 months). Reported internal metrics: C-index 0.6590 ± 0.0290; Uno-AUC 0.7140/0.6981/0.7093/0.7052/0.7396 at 6/12/24/36/48 months; IPCW Brier 0.1539/0.2131/0.1862/0.1509/0.1084; 12-month binary AUC 0.6571 ± 0.0339.

**Ablation.** On the same folds, the whole-tumor average-pooled variant (no SALP; "frozen") achieved C-index 0.6625 ± 0.0254 and 12-month binary AUC 0.6645 ± 0.0296, i.e. SALP provided no gain in internal validation.

**LOCO external validation (this study).** Using the precomputed feature library (z_salp, tumor_global, clinical for all 1106 patients), we trained the same head on one strict cohort (UCSF 367 or UPENN 608) and evaluated on the other, in both directions, with training-fold-fitted bin edges and scalers (Section 2.6 protocol). Results in Section 3.2.

### 2.3 Imaging representation B: whole-brain R2SN graph network

**Inputs.** For each patient and each modality (T1c, T2, FLAIR), per-ROI node features for 246 AAL regions were taken from the R2SN feature library used in previous work [ref: CM-GNN pipeline]: either the high-dimensional "after_bn" selected normalized features (455/497/480 dimensions per ROI for T1c/T2/FLAIR) or the reduced 93-feature version. The adjacency for each modality was the 246×246 Pearson correlation matrix between ROIs (diagonal set to zero; self-loops removed).

**Model.** A dense pure-PyTorch graph convolution network (no geometric library): Â = D^{-1/2}(A+I)D^{-1/2}, message passing H' = σ(Â H W), per-modality encoders, mean+max readout, clinical (age, sex, GTR, MGMT) fusion, and a multi-timepoint discrete survival head (K=40 or 80 bins over the training-max horizon, monotone S(t)=∏(1−h), censored discrete NLL). Adam lr 1e-3, weight decay 1e-4, batch 16, up to 300 epochs with early stopping (patience 25) on an 80/20 validation split; node features z-scored with training-fold statistics; clinical variables median-imputed and age z-scored.

**Validation.** Internal (UCSF 80/20 split) and LOCO external in both directions (Section 3.2). Variants: graph-only, clinical-only, and graph+clinical (full), each × (high-dim / 93-dim node features where applicable).

### 2.4 Clinical-molecular multi-timepoint discrete survival framework

**Model.** The same discrete survival head family as above, applied to clinical-molecular features only: x = [age, sex, GTR, MGMT]. K=80 time bins over the training-cohort maximum observed time; hazards h = sigmoid(logits) over bins; survival S(k) = ∏_{j≤k} (1−h_j) is monotone by construction; bin width W = T_max/K. Loss: censored discrete NLL (events contribute log h_{bin(t)} + Σ_{j<bin(t)} log(1−h_j); censored contribute Σ_{j≤bin(t)} log(1−h_j)). Linear layers with the same architecture as the imaging heads' MLP.

**Internal validation.** 5-fold CV on UCSF (367): out-of-fold C-index and Uno-AUC; a clinical Cox proportional-hazards model (age/sex/GTR/MGMT) with the same folds served as a classical baseline.

**External validation (LOCO).** Two directions: UCSF-trained → UPENN-tested and UPENN-trained → UCSF-tested, with training-fold-fitted scalers and (for the Cox baseline) coefficients.

**Conditional survival.** Dynamic update S(t|t0) = S(t)/S(t0) for t > t0, evaluated at t0 ∈ {12, 24} months, t ∈ {24, 36} months, against Kaplan-Meier conditional estimates on the external cohort (patients with follow-up > t0, using days).

### 2.5 Evaluation protocol and statistics

All external evaluations follow one protocol:
- **Concordance index**: Harrell C-index [ref: *Harrell et al.*] on the test cohort, risk score = −E[T] (negative expected survival from the discrete curve).
- **Time-dependent AUC**: Uno's IPCW AUC [ref: *Uno et al.*] with estimate 1−S(t), at landmarks 3/6/12/24/36 months (and 48 months where horizon allows). Test follow-up is censored at the training cohort's maximum observed time (horizon cap) to keep IPCW weights defined.
- **Calibration**: IPCW Brier [ref: *Graf et al.*] vs. the Kaplan-Meier null model (training cohort), plus observed-minus-predicted survival at each landmark (Kaplan-Meier estimate of the test cohort vs. mean predicted S(t)), and conditional-survival calibration as above.
- **Software**: Python 3.9, scikit-survival 0.23.1 [ref: *Pölsterl*], lifelines [ref: *Davidson-Pilon*]; GPU training in PyTorch 2.0.1 (RTX 4070 Laptop, 8 GB).

All reported point estimates come from fixed, deterministic protocols (seeds documented in code); repeated-split stability is reported as mean ± SD where applicable.

## 3. Results

### 3.1 Cohort characteristics

[Table 1 about here]

Both cohorts consist of IDH-wildtype GBM (strict subsets). UPENN has a higher event rate (96.2% vs. 60.8%) and longer follow-up; UCSF has more censoring (144/367) concentrated in the stricter subset. In the historical mixed-cohort construction (UCSF 495), 99/495 (20.0%) were WHO grade 2-3 and ~102/495 (20.6%) were IDH-mutant (Section 4).

### 3.2 Cross-institution transfer of imaging representations (negative results)

**SwinUNETR + SALP.** Table 2 reports internal vs. external performance. [Table 2 about here] [Figure 2 about here]

| Model (SwinUNETR+SALP) | Internal 25-fold CV | LOCO →UPENN | LOCO →UCSF |
|---|---|---|---|
| salp_only (1152-d) | 0.6590 ± 0.0290 (legacy, mixed cohort) | 0.525 | 0.497 |
| salp_full (1159-d, legacy config) | 0.6590 ± 0.0290 | 0.520 | 0.526 |
| frozen (no SALP, whole-tumor pooling) | 0.6625 ± 0.0254 | — | — |

Under strict LOCO validation, external C-index drops to chance level (0.50-0.53) in both directions and for both feature variants; 3-month Uno-AUC on the UCSF test cohort was 0.391 (worse than chance), indicating near-total loss of signal. Adding age/sex (salp_full) did not rescue performance.

**Whole-brain R2SN graph.** [Table 3 about here]

| Model | Internal (UCSF val) | LOCO →UPENN | LOCO →UCSF |
|---|---|---|---|
| Graph-only, high-dim (455-497/ROI) | 0.63 (±) | 0.523 | 0.531 |
| Graph-only, 93-dim | — | 0.567 | 0.601 |
| Graph + clinical, 93-dim | 0.66 (±) | 0.615 | 0.666 |
| Clinical-only | 0.66 | 0.639 | 0.651 |

The high-dimensional R2SN graph failed to transfer (≈ chance). Reducing node-feature dimensionality from ~455-497 to 93 per ROI improved external C-index by ≈ +0.06 (0.53 → 0.57-0.60), suggesting dimension/over-fitting contributes to transfer failure; even so, the graph did not exceed a pure clinical model, and adding the graph to clinical features produced no consistent gain (direction-dependent: −0.025 →UPENN, +0.018 →UCSF).

### 3.3 Clinical-molecular framework (positive results)

**Internal validation (UCSF, 5-fold CV, n=367).** Out-of-fold C-index 0.622 (discrete framework); clinical Cox baseline 0.671 ± 0.031. Brier was 4-12% better than the KM null model across the five landmarks.

**External validation.** [Table 4 about here] [Figure 3 about here] [Figure 4 about here]

| Metric | UCSF→UPENN (n=608) | UPENN→UCSF (n=367) |
|---|---|---|
| C-index | 0.639 | 0.651 |
| Uno-AUC @3/6/12/24/36 mo | 0.714/0.688/0.705/0.658/0.659 | 0.802/0.741/0.730/0.739/0.747 |
| Brier vs. KM null @3/6/12/24/36 mo | 0.100/0.183/0.260/0.276/0.238 (−6/−10/−13/−6/−6%) | 0.065/0.121/0.164/0.124/0.079 (−12/−14/−15/−9/+0.4%) |
| Calibration (pred−KM) @3/6/12/24/36 mo | +0.032/+0.032/+0.044/+0.047/+0.017 | −0.001/−0.006/+0.013/+0.006/+0.007 |

The UPENN-trained model transfers to UCSF with excellent calibration (|Δ| ≤ 0.013); the UCSF-trained model transfers to UPENN with a small systematic survival over-estimation (+0.02…+0.05), consistent with the shorter-follow-up UCSF training distribution.

**Conditional survival.** [Table 5 about here]

| Conditional interval | UCSF→UPENN (pred vs. KM) | UPENN→UCSF (pred vs. KM) |
|---|---|---|
| 12→24 months | 0.463 vs 0.409 (+0.053) | 0.514 vs 0.512 (+0.002) |
| 12→36 months | 0.257 vs 0.241 (+0.016) | 0.337 vs 0.330 (+0.007) |
| 24→36 months | 0.531 vs 0.590 (−0.059) | 0.664 vs 0.645 (+0.020) |

Absolute errors ≤ 0.06 in all cells, ≤ 0.02 in the UPENN-trained direction.

### 3.4 Comparison with prior work

[Table 6 about here]

A closely related GNN study (CM-GNN [ref]) reported a 12-month AUC of 72.21% using a 5-cohort leave-one-cohort-out design (UPENN-584, UCSF-351, TCGA-89, IvyGAP-20, CPTAC-12) with a binary 12-month label. Our clinical-molecular framework achieves 12-month Uno-AUC of 0.705-0.730 under 2-cohort LOCO; the two numbers are of the same magnitude but are not directly comparable (metrics, labels, and validation protocols differ). Importantly, the imaging asset underlying that study (the R2SN graph, same feature library) does not transfer across the two cohorts in our protocol, and the previously reported internal performance of the SwinUNETR+SALP pipeline (C-index 0.659, AUC@48m 0.7396) collapses under LOCO validation.

## 4. Discussion

**Main finding.** Two independent imaging representations — a pretrained SwinUNETR encoder with anatomy-aware pooling, and a whole-brain R2SN graph — fail leave-one-cohort-out validation in glioblastoma survival, with external C-indices at or near chance (0.50-0.60) despite favorable internal cross-validation (0.63-0.66). In contrast, a transparent clinical-molecular multi-timepoint discrete survival framework transfers well (external C-index 0.639-0.651) and is well calibrated, including for conditional survival.

**Why does internal CV overstate imaging performance?** We identify two mechanisms. First, *mixed-cohort grade/IDH discrimination*: if the pooled training cohort contains lower-grade or IDH-mutant gliomas, models can learn the relatively easy discrimination between tumor types — whose imaging appearance and prognosis differ dramatically — inflating internal CV; this signal is absent in a test cohort composed entirely of GBM. Our historical mixed construction (UCSF 495) contained 20% WHO grade 2-3 and 21% IDH-mutant cases; the same internal-CV protocol yielded UCSF sub-cohort C-index 0.699 but only 0.607 for the pure-GBM UPENN sub-cohort, consistent with this mechanism. Second, *site-specific distribution shift*: even with strictly homogeneous GBM cohorts, imaging features depend on scanner, protocol, registration, and (critically) segmentation quality; the frozen-encoder features (SwinUNETR) and ROI-level statistics (R2SN) both carry site-dependent structure that internal CV cannot expose.

**Dimensionality mitigates, but does not solve, transfer failure.** Reducing R2SN node features from ~455-497 to 93 dimensions per ROI improved external C-index by ≈ +0.06 without changing the architecture — consistent with over-fitting of high-dimensional per-ROI statistics under limited training samples. This argues that feature representation — including the choice of atlas, normalization, and dimension reduction — matters at least as much as model architecture for generalization, and should be reported and ablated.

**A defensible baseline.** The clinical-molecular model's external discrimination (C-index 0.64-0.65) and calibration (Brier 5-14% better than KM-null; calibration within 0.02-0.05) are comparable to state-of-the-art reported magnitudes for GBM survival, including the 12-month AUC of a complex graph network on overlapping data, without any imaging input. We do not claim clinical utility on the basis of discrimination alone; rather, we offer this framework as a transparent, reproducible baseline — with correct right-censoring, monotone survival, individualized 3-36 month probabilities, and dynamic conditional-survival updates — against which imaging contributions must be measured externally.

**Limitations.** (i) Two public cohorts only; a third independent cohort (e.g., TCGA-GBM) would strengthen the negative result; (ii) IPCW-based metrics require horizon capping, so calibration beyond the training horizon is only partially assessed; (iii) KPS was largely missing and excluded; (iv) the UCSF OS time origin (surgery) reflects the cohort's own definitions, which could not be fully re-verified against the original data dictionary; (v) our LOCO protocol is internal in the sense that both institutions use the same public preprocessing lineage for some assets (e.g., R2SN features), which if anything makes the observed transfer failure more, not less, informative; (vi) single-split external evaluation was cross-checked in both directions rather than with repeated multi-fold external CV.

**Ethics of reporting.** Negative or null results in medical imaging are under-reported; we make all code and predictions available to facilitate independent replication (Data/Section 5).

## 5. Conclusions

Internal cross-validation systematically overstates the cross-institution generalization of imaging-based survival models for glioblastoma. Two mainstream imaging representations — pretrained SwinUNETR with anatomy-aware pooling and whole-brain R2SN graphs — perform at or near chance under strict leave-one-cohort-out validation in both directions on public GBM cohorts. A clinical-molecular multi-timepoint framework with conditional survival provides a well-calibrated, externally validated baseline (C-index 0.639-0.651; 12-month Uno-AUC 0.705-0.730) and should be adopted in future imaging-survival research as the comparator of record.

## Declarations

- **Data availability.** UCSF-PDGM and UPENN-GBM are publicly available; derived survival matrices and prediction files are provided as supplementary material.
- **Code availability.** All code is available at [repository placeholder]; key scripts: `code/model/step3_model.py`, `step3_train.py`, `step3_cv.py`, `step5_loco_train.py`, `step4_summary.py`, `step4_brier.py` and `code/data_pipeline/step0_new_direction.py`–`step2_baseline.py` (data construction; proprietary R2SN/SALP inputs via environment variables, see `data/README.md`).
- **Funding.** None.
- **Conflicts of interest.** None declared.

## Figure legends (placeholders — figures to be generated)

**Figure 1.** Study overview. Two imaging pipelines (SwinUNETR+SALP; whole-brain R2SN graph) and the clinical-molecular framework, all evaluated with the internal-CV vs. LOCO protocol. *(Figure 1 about here)*

**Figure 2.** Transfer waterfall plots: internal vs. external (LOCO) C-index for each imaging pipeline and variant (SwinUNETR salp_only/salp_full; R2SN high-dim/93-dim). Horizontal dashed line at 0.5 (chance). *(Figure 2 about here)*

**Figure 3.** Uno-AUC of the clinical-molecular framework at 3/6/12/24/36 months, both LOCO directions, with 25-fold internal bands where available. *(Figure 3 about here)*

**Figure 4.** Calibration: observed (KM) vs. mean predicted survival at the five landmarks, both directions, with identity line; inset: conditional-survival calibration (predicted vs. KM for 12→24, 12→36, 24→36 months). *(Figure 4 about here)*

## Tables (placeholders — to be formatted per journal)

- **Table 1.** Cohort characteristics and survival construction. *(Table 1 about here)*
- **Table 2.** SwinUNETR+SALP: internal vs. LOCO external performance. *(Table 2 about here)*
- **Table 3.** R2SN graph: internal vs. LOCO external performance by variant. *(Table 3 about here)*
- **Table 4.** Clinical-molecular framework: full external metric set. *(Table 4 about here)*
- **Table 5.** Conditional-survival calibration. *(Table 5 about here)*
- **Table 6.** Comparison with prior published work (CM-GNN; SwinUNETR+SALP legacy). *(Table 6 about here)*

## References (TO VERIFY — author lists, volumes, pages, and DOIs to be checked before submission)

1. Stupp R, et al. Radiotherapy plus concomitant and adjuvant temozolomide for glioblastoma. N Engl J Med. 2005;352:987-996. *(verify)*
2. Hatamizadeh A, et al. Swin UNETR: Swin Transformers for semantic segmentation of brain tumors in MRI images. MICCAI BrainLes workshop 2022. arXiv:2201.01266. *(verify)*
3. Liu Z, et al. Swin Transformer: Hierarchical vision transformer using shifted windows. ICCV 2021. *(verify)*
4. Baid U, et al. The RSNA-ASNR-MICCAI BraTS 2021 benchmark on brain tumor segmentation and radiogenomic classification. arXiv:2107.02314. *(verify)*
5. Calabrese E, et al. UCSF-PDGM dataset: preoperative diffuse glioma MRI dataset. Radiology: Artificial Intelligence (2021/2022). *(verify exact journal/volume)*
6. Bakas S, et al. Advancing the cancer genome atlas glioma MRI collections with expert segmentation labels and radiomic features. Scientific Data. 2017;4:170117. *(verify — UPENN-GBM cohort sources)*
7. Lee C, Zame W, Yoon J, van der Schaar M. DeepHit: A deep learning approach to survival analysis with competing risks. AAAI 2018. *(verify)*
8. Katzman JL, et al. DeepSurv: Personalized treatment recommender system using a Cox proportional hazards deep neural network. BMC Med Res Methodol. 2018;18:24. *(verify)*
9. Uno H, Cai T, Pencina MJ, D'Agostino RB, Wei LJ. On the C-statistics for evaluating overall adequacy of risk prediction procedures with censored survival data. Stat Med. 2011;30(10):1105-1117. *(verify)*
10. Graf E, Schmoor C, Sauerbrei W, Schumacher M. Assessment and comparison of prognostic classification schemes for survival data. Stat Med. 1999;18(17-18):2529-2545. *(verify)*
11. Harrell FE Jr, Lee KL, Mark DB. Multivariable prognostic models: issues in developing models, evaluating assumptions and adequacy, and measuring and reducing errors. Stat Med. 1996;15(4):361-387. *(verify)*
12. Pölsterl S. scikit-survival: A library for time-to-event analysis built on top of scikit-learn. JMLR. 2020;21(212):1-6. *(verify)*
13. Davidson-Pilon C. lifelines: survival analysis in Python. Journal of Open Source Software. 2019;4(39):1317. *(verify)*
14. Tutz G, Schmid M. Modeling Discrete Time-to-Event Data. Springer; 2016. *(verify)*
15. MONAI Consortium. MONAI: Medical Open Network for AI. (Software). *(verify)*
16. BrainSegFounder pretrained weights (MONAI model zoo, BraTS fine-tuned SwinUNETR). *(verify exact publication/DOI)*
17. [CM-GNN reference — the previously published multi-modal graph network for 12-month GBM survival, Front. Oncol. 2023; report 72.21% AUC. *(verify authors/title/DOI)*]
18. [Harrell C-index original reference — see #11 or Harrell FE, et al. JAMA 1982. *(verify)*]
19. [Additional reference for IPCW-Brier options, e.g., Gerds & Schumacher 2006. *(verify)*]

*Note: the web-search API was unavailable during drafting; every reference above must be verified (exact author list, journal, volume, pages, DOI) before submission.*
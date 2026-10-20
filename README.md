# Imaging features transfer poorly across institutions in glioblastoma survival prediction

A transparent, reproducible companion to the accompanying manuscript:

> **Main finding.** Two mainstream MRI-based imaging representations — (A) a pretrained SwinUNETR
> encoder with anatomy-aware soft pooling (SALP) and a discrete survival head, and (B) a whole-brain
> R2SN graph network — perform well in *internal* cross-validation (C-index 0.63-0.66) but **collapse
> to chance (C-index 0.50-0.60) under strict leave-one-cohort-out (LOCO) external validation**
> between UCSF-PDGM and UPENN-GBM, in both directions.
>
> **Positive result.** A clinical-molecular multi-timepoint discrete survival framework
> (age / sex / extent of resection / MGMT methylation) transfers well: external C-index 0.639-0.651,
> 12-month Uno-AUC 0.705-0.730, IPCW Brier 5-15% better than a Kaplan-Meier null model,
> calibration within 0.05 absolute, and well-calibrated conditional survival (dynamic update
> S(t|t0) = S(t)/S(t0)) — provided as the default baseline for future imaging-survival work.

## Repository layout

```
.
├── README.md
├── .gitignore
├── data/
│   └── README.md            # npz schemas, environment variables, demo data
├── docs/
│   ├── manuscript.md        # (near-)final manuscript draft, figures/tables placeholders
│   └── results_tables.md    # all verified numbers behind the manuscript
├── code/
│   ├── model/               # multi-timepoint discrete survival model + evaluation
│   │   ├── step3_model.py       # model + sanity gate (pure PyTorch, no geometric libs)
│   │   ├── step3_train.py       # training (--smoke / --train, modes, reverse)
│   │   ├── step3_cv.py          # internal 5-fold CV (clinical model)
│   │   ├── step5_loco_train.py  # LOCO transfer tests of SwinUNETR/SALP features
│   │   ├── step4_summary.py     # decision table (external C-index both directions)
│   │   └── step4_brier.py       # Brier + calibration + conditional survival tables
│   └── data_pipeline/       # data construction (proprietary inputs via env vars)
│       ├── step0_new_direction.py   # survival construction + R2SN coverage checks
│       ├── step1_sanity.py          # gate before dataset build
│       ├── step1_build_dataset.py   # build UCSF_dev.npz / UPENN_ext.npz
│       ├── step1b_build_v2.py       # 93-feature node-feature variant
│       └── step2_baseline.py        # clinical Cox + whole-brain-pooled Cox baselines
└── tools/
    └── make_demo_data.py    # synthetic demo cohorts for pipeline smoke tests (NOT real data)
```

## Quick start (pipeline smoke test, no real data needed)

```bash
pip install numpy torch  # GPU strongly recommended for training

# 1. generate synthetic demo cohorts (24 patients each, random numbers)
python tools/make_demo_data.py

# 2. model sanity gate (forward pass, monotone survival, finite loss)
python code/model/step3_model.py --sanity

# 3. train smoke (8 real-format patients, 1 forward/backward pass)
python code/model/step3_train.py --smoke

# 4. full training on demo data (UCSF-trained -> UPENN-tested)
python code/model/step3_train.py --train --mode clin_only
```

Metrics scripts (`step4_summary.py`, `step4_brier.py`) require `scikit-survival` and `lifelines`.

## Reproducing the real results

1. Download the public cohorts: **UCSF-PDGM** (preoperative diffuse-glioma MRI dataset;
   see manuscript references) and the **UPENN-GBM** cohort (TCIA-available public GBM collection;
   see manuscript references).
2. Build the R2SN whole-brain feature library and (for the SwinUNETR pipeline) the frozen
   encoder + SALP feature library. These inputs are **proprietary to the authors' environment**;
   the code reads them only through environment variables (see `data/README.md`). Construct the
   survival matrices with `code/data_pipeline/`.
3. Run `code/model/` in order (train → LOCO → metrics). All evaluation uses days as the time unit,
   correct right-censoring, IPCW horizon capping at the training cohort's maximum observed time,
   and both LOCO directions.

Environment variables used by the code (all optional; missing inputs raise clear errors):

| Variable | Used by | Meaning |
|---|---|---|
| `GBM_DATA` | model + pipeline | dir containing `new_direction/*.npz` (defaults to `<repo>/data`) |
| `SALP_FEATURES` | `step5_loco_train.py` | dir with `{pid}.npz` SwinUNETR/SALP feature library |
| `R2SN_NODE` / `R2SN_EDGE` | data_pipeline | dirs with per-patient node-feature / edge-matrix `.npy` files |
| `R2SN_V2_NODE` | `step1b_build_v2.py` | dir with 93-feature node-feature files |
| `CLIN_CSV_DIR` | data_pipeline | dir with clinical CSVs |
| `MANIFEST_DIR` | data_pipeline | dir with patient manifests |

## Licensing

License: **TBD by the authors** — do not reuse the manuscript text or derived clinical
predictions without permission. Code is shared for transparency and reproducibility of the
reported analysis.

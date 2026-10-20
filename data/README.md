# data/

No patient-level data is shipped in this repository. This directory documents the
`.npz` schemas produced by `code/data_pipeline/` and consumed by `code/model/`,
and how to obtain demo data.

## npz schemas (`new_direction/`)

### `UCSF_dev.npz` / `UPENN_ext.npz` (built by `step1_build_dataset.py`)

| key | shape | dtype | meaning |
|---|---|---|---|
| `node_t1c` / `node_t2` / `node_flair` | (N, 246, 455/497/480) | float32 | per-ROI node features (246 AAL ROIs) per modality |
| `adj_t1c` / `adj_t2` / `adj_flair` | (N, 246, 246) | float32 | Pearson-correlation adjacency, zero diagonal |
| `time` | (N,) | float64 | survival days (events) or follow-up days (censored) |
| `event` | (N,) | bool | 1 = death, 0 = right-censored |
| `clin` | (N, 6) | float32 | [age, sex_bin, gtr_bin, mgmt_bin, idh_wt, kps] |
| `clin_cols` | (6,) | — | column names |
| `pids` | (N,) | str | public patient identifiers (same as dataset publications) |

Optional variant `*_v2.npz` (built by `step1b_build_v2.py`): node features are the
93-feature reduced representation; adjacency identical.

### Prediction files (written by `code/model/`)

`pred_gnn_*.npz`, `pred_loco_*.npz`, `pred_clin_cv5.npz` contain per-patient risk
(negative expected survival, higher = worse), survival `S` evaluated at the manuscript
landmarks, `time`/`event`, and cohort labels — see the writing scripts for exact keys.

## Demo data (synthetic, for pipeline smoke tests only)

```bash
python tools/make_demo_data.py
```

writes `new_direction/UCSF_dev.npz` and `new_direction/UPENN_ext.npz` with 24 fully
random patients each, matching the exact schemas above. Numbers are meaningless —
they exist only so the model/training/metrics code can be exercised end-to-end.

## Reproducing the real cohorts

1. **UCSF-PDGM** — public preoperative diffuse-glioma MRI dataset (see manuscript
   references for the data paper). The strict GBM subset used in the manuscript is:
   WHO grade 4, IDH-wildtype, documented OS (n=367; one patient without OS excluded).
2. **UPENN-GBM** — public GBM cohort (TCIA-available; see manuscript references).
   The manuscript excludes 3 patients with "deceased — uncertain date of death"
   (n=608).
3. R2SN node/edge features and the SwinUNETR/SALP feature library are **not publicly
   redistributed here**; they are read from the directories given by the environment
   variables in the root `README.md`.
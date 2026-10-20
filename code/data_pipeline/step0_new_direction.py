"""New direction Step 0: verify the whole-brain R2SN asset + correct survival.

Checks:
  A. UPENN censoring-complete survival (fix: death-days vs censor-days)
  B. UCSF OS time-origin (days vs months)
  C. R2SN coverage: (node_features + edge_matrices) x3 modalities per patient
"""
import os
import glob
import numpy as np
import pandas as pd

# ---- proprietary inputs; set these via environment variables ----
CLIN_DIR = os.environ.get('CLIN_CSV_DIR', '')  # dir with UCSF-PDGM-metadata_v5.csv, UPENN-GBM_clinical_info_v2.1.csv
MAN_DIR = os.environ.get('MANIFEST_DIR', '')   # dir with manifest_UCSF.csv, manifest_UPENN.csv
R2SN_NODE = os.environ.get('R2SN_NODE', '')    # dir with per-patient node-feature .npy files
R2SN_EDGE = os.environ.get('R2SN_EDGE', '')    # dir with per-patient edge-matrix .npy files

# ---------- A. UPENN correct survival ----------
c = pd.read_csv(os.path.join(CLIN_DIR, 'UPENN-GBM_clinical_info_v2.1.csv'), encoding='utf-8-sig')
c['session'] = c.ID.str[-2:]
c['base'] = c.ID.str[:-3]
event = (c.Survival_Status.str.startswith('Deceased', na=False)
         & (c.Survival_Status != 'Deceased - uncertain date of death'))
time = np.where(event,
                pd.to_numeric(c['Survival_from_surgery_days_UPDATED'], errors='coerce'),
                pd.to_numeric(c['Survival_Censor'], errors='coerce'))
c['time_days'] = time
c['event'] = event.astype(int)
c['uncertain'] = (c.Survival_Status == 'Deceased - uncertain date of death').astype(int)
pre = c[c.session == '11'].copy()
valid = pre[pre.time_days.notna() & (pre.time_days > 0)]
print(f'[A] UPENN preop n={len(pre)}; valid-time n={len(valid)}; '
      f'events={int(valid.event.sum())}; censored={int((1 - valid.event).sum())}; '
      f'uncertain-excluded={int(pre.uncertain.sum())}')
print(f'    median: event {valid[valid.event==1].time_days.median():.0f}d | '
      f'censor {valid[valid.event==0].time_days.median():.0f}d')
print(f'    censored follow-up days: {sorted(valid[valid.event==0].time_days.astype(int).tolist())}')

# ---------- B. UCSF OS time origin ----------
u = pd.read_csv(os.path.join(CLIN_DIR, 'UCSF-PDGM-metadata_v5.csv'), encoding='utf-8-sig')
u['OS'] = pd.to_numeric(u['OS'], errors='coerce')
u['OSm'] = pd.to_numeric(u['OS_months'], errors='coerce')
rat = (u['OS'] / u['OSm']).dropna()
print(f'[B] UCSF OS/OS_months ratio median={rat.median():.2f} (≈30.4 => OS is days)')

# ---------- C. R2SN coverage ----------
NF = R2SN_NODE
EM = R2SN_EDGE
nf_files = set(os.path.basename(f) for f in glob.glob(os.path.join(NF, '*.npy')))
em_files = set(os.path.basename(f) for f in glob.glob(os.path.join(EM, '*.npy')))


def key_from_nf(fname):
    # UCSF-PDGM-0004_after_bn_flair_selected_normed_rfeature.npy
    return fname.split('_after_bn_')[0]


def key_from_em(fname):
    # UCSF-PDGM-0004_after_bn_flair_pc_matrix.npy
    return fname.split('_after_bn_')[0]


nf_keys = set(key_from_nf(f) for f in nf_files)
em_keys = set(key_from_em(f) for f in em_files)
common = nf_keys & em_keys
print(f'[C] node_features files={len(nf_files)}, edge_files={len(em_files)}, '
      f'unique patients node={len(nf_keys)}, edge={len(em_keys)}, both={len(common)}')
# sample UPENN keys
up_keys = sorted([k for k in common if 'UPENN' in k])[:5]
uc_keys = sorted([k for k in common if 'UCSF' in k])[:3]
print('    UPENN keys sample:', up_keys)
print('    UCSF keys sample:', uc_keys)

# coverage vs my cohorts
def have_3mod(key):
    return all(any(f.startswith(key + '_after_bn_' + m) for f in nf_files) for m in ['t1c', 't2', 'flair'])


man_u = pd.read_csv(os.path.join(MAN_DIR, 'manifest_UCSF.csv'))
man_p = pd.read_csv(os.path.join(MAN_DIR, 'manifest_UPENN.csv'))
ucsf_ids = man_u[(man_u.grade == 4) & (man_u.idh == 'wildtype') & (man_u.is_followup == 0)].folder_id
upenn_ids = man_p[(man_p.is_followup == 0)].patient_id
u_cov = sum(1 for k in ucsf_ids if k in common and have_3mod(k))
p_cov = sum(1 for k in upenn_ids if k in common and have_3mod(k))
print(f'    UCSF g4wt baseline with full 3-mod R2SN: {u_cov}/{len(ucsf_ids)}')
print(f'    UPENN preop with full 3-mod R2SN: {p_cov}/{len(upenn_ids)}')
missing_u = [k for k in ucsf_ids if k not in common][:5]
missing_p = [k for k in upenn_ids if k not in common][:5]
print('    UCSF missing sample:', missing_u)
print('    UPENN missing sample:', missing_p)
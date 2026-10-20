"""New direction Step 1 sanity checks (run BEFORE building the dataset).

Correctness checks the user asked to gate every run:
  1. node/edge integrity: NaN/Inf, adjacency symmetry, value ranges
  2. modality availability (t1c/t2/flair) per patient
  3. survival <-> 12month label consistency (cross-check)
"""
import os
import numpy as np
import pandas as pd

NF = os.environ.get('R2SN_NODE', '')
EM = os.environ.get('R2SN_EDGE', '')
MODS = ['t1c', 't2', 'flair']


def check_patient(key):
    for m in MODS:
        xf = os.path.join(NF, f'{key}_after_bn_{m}_selected_normed_rfeature.npy')
        af = os.path.join(EM, f'{key}_after_bn_{m}_pc_matrix.npy')
        X = np.load(xf)
        A = np.load(af)
        nanx = int(np.isnan(X).sum())
        infx = int(np.isinf(X).sum())
        nana = int(np.isnan(A).sum())
        sym = bool(np.allclose(A, A.T, atol=1e-5))
        diag = np.diag(A)
        print(f'  {m:5s} X{X.shape} nan={nanx} inf={infx} '
              f'| A{A.shape} nan={nana} sym={sym} diag∈[{diag.min():.3f},{diag.max():.3f}]')
    return True


if __name__ == '__main__':
    for key in ['UCSF-PDGM-0004', 'UPENN-GBM-00001', 'UPENN-GBM-00002']:
        print('patient', key)
        check_patient(key)

    # survival <-> 12month cross-check
    c = pd.read_csv(os.path.join(os.environ.get('CLIN_CSV_DIR', ''), 'UPENN-GBM_clinical_info_v2.1.csv'),
                    encoding='utf-8-sig')
    event = (c.Survival_Status.str.startswith('Deceased', na=False)
             & (c.Survival_Status != 'Deceased - uncertain date of death'))
    time = np.where(event,
                    pd.to_numeric(c['Survival_from_surgery_days_UPDATED'], errors='coerce'),
                    pd.to_numeric(c['Survival_Censor'], errors='coerce'))
    c['time_days'] = time
    c['event'] = event.astype(int)
    dec = c[c.event == 1]
    mismatch = (dec['12month'].astype(int) != (dec.time_days > 365).astype(int)).sum()
    print(f'\nUPENN deceased: "12month" vs (time>365) mismatch = {mismatch}/{len(dec)}')
    cens = c[(c.event == 0) & c.time_days.notna()]
    short = cens[cens.time_days < 365]
    print(f'censored with followup<365d: n={len(short)}  '
          f'times={short.time_days.astype(int).tolist()}  m12={short["12month"].astype(int).tolist()}')
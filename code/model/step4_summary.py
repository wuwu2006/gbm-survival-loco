"""Step 4: summary decision table from all pred_gnn*.npz files.

For each file: external C-index (test cohort) + mean Uno-AUC@5 landmarks,
in BOTH directions (fwd: train UCSF->test UPENN; rev: train UPENN->test UCSF).
Decision metric = avg external C-index across the two directions.
"""
import glob
import os
import numpy as np
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.util import Surv

OUT = os.path.join(os.environ.get('GBM_DATA',
                   os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')),
                   'new_direction')


def eval_file(path):
    z = np.load(path, allow_pickle=True)
    lm = z['landmarks'].astype(float)
    is_rev = 'rev' in os.path.basename(path)
    if is_rev:
        tr_t, tr_e = z['upenn_time'].astype(float), z['upenn_event'].astype(bool)
        te_t, te_e = z['ucsf_time'].astype(float), z['ucsf_event'].astype(bool)
        teS, teR = z['ucsf_S'], z['ucsf_risk']
    else:
        tr_t, tr_e = z['ucsf_time'].astype(float), z['ucsf_event'].astype(bool)
        te_t, te_e = z['upenn_time'].astype(float), z['upenn_event'].astype(bool)
        teS, teR = z['upenn_S'], z['upenn_risk']
    ci = concordance_index_censored(te_e, te_t, teR)[0]
    H = float(tr_t.max())
    tec_t = np.minimum(te_t, H)
    tec_e = te_e & (te_t < H)
    ytr = Surv.from_arrays(tr_e, tr_t)
    yte = Surv.from_arrays(tec_e, tec_t)
    aucs = []
    for i, t in enumerate(lm):
        r = cumulative_dynamic_auc(ytr, yte, 1.0 - teS[:, i], np.array([float(t)]))
        aucs.append(r[0][0] if isinstance(r, tuple) else r[:, 0][0])
    return ci, float(np.mean(aucs)), is_rev


rows = {}
for f in sorted(glob.glob(rf'{OUT}\pred_gnn*.npz')):
    name = os.path.basename(f).replace('pred_gnn_', '').replace('.npz', '').replace('pred_gnn', 'base')
    ci, aucm, rev = eval_file(f)
    key = name.replace('rev_', '') if name.startswith('rev_') else name
    rows.setdefault(key, {})['rev' if rev else 'fwd'] = (ci, aucm)

print(f'{"model":28s} {"extC->UPENN":>11s} {"extC->UCSF":>11s} {"avgExtC":>8s} {"UnoAUC^":>8s} {"UnoAUC>":>8s}')
for k in sorted(rows):
    fwd = rows[k].get('fwd', (np.nan, np.nan))
    rev = rows[k].get('rev', (np.nan, np.nan))
    avg = np.nanmean([fwd[0], rev[0]])
    print(f'{k:28s} {fwd[0]:>11.4f} {rev[0]:>11.4f} {avg:>8.4f} {fwd[1]:>8.3f} {rev[1]:>8.3f}')
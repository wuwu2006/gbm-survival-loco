"""Step B: Brier scores + calibration tables for the final clinical model.

External (fwd: train UCSF->test UPENN; rev: train UPENN->test UCSF),
internal (5-fold CV out-of-fold). Null model = train-KM as reference.
"""
import os
import numpy as np
from sksurv.metrics import brier_score, cumulative_dynamic_auc
from sksurv.util import Surv
from lifelines import KaplanMeierFitter

OUT = os.path.join(os.environ.get('GBM_DATA',
                   os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')),
                   'new_direction')
LM = np.array([91.0, 182.0, 365.0, 730.0, 1095.0]).astype(float)
kmf = KaplanMeierFitter()


def brier_and_cal(tr_t, tr_e, te_t, te_e, teS, tag):
    """teS: (n_te, L) survival at LM. IPCW-cap test at train horizon."""
    H = float(tr_t.max())
    te_t2 = np.minimum(te_t, H)
    te_e2 = te_e & (te_t < H)
    ytr = Surv.from_arrays(tr_e, tr_t)
    yte = Surv.from_arrays(te_e2, te_t2)
    br, _ = brier_score(ytr, yte, teS, LM)  # 0.23.1 returns (times, scores)? detect order
    times_out, scores = (br, _) if np.array_equal(br, LM) else (_, br)
    br = scores if np.array_equal(br, LM) else br
    kmf.fit(tr_t, tr_e)
    null = np.tile(np.asarray([float(kmf.predict(t)) for t in LM]), (len(te_t), 1))
    br_null, _ = brier_score(ytr, yte, null, LM)
    times_out2, scores2 = (br_null, _) if np.array_equal(br_null, LM) else (_, br_null)
    br_null = scores2 if np.array_equal(br_null, LM) else br_null
    print(f'--- {tag}: Brier@3/6/12/24/36mo = '
          + ' / '.join(f'{x:.3f}' for x in br))
    print(f'    null(KM)                  = '
          + ' / '.join(f'{x:.3f}' for x in br_null))
    print(f'    rel-err vs null:          = '
          + ' / '.join(f'{(x - y) / y * 100:+.1f}%' for x, y in zip(br, br_null)))
    # calibration table: KM of test vs mean predicted S(t)
    print(f'    calibration KM vs mean pred @ {LM.astype(int).tolist()}:')
    for j, t in enumerate(LM):
        kmf.fit(te_t2, te_e2)
        emp = float(kmf.predict(t))
        print(f'      t={int(t):4d}d  testKM={emp:.3f}  meanPred={float(teS[:, j].mean()):.3f}  '
              f'delta={float(teS[:, j].mean()) - emp:+.3f}')
    return br, br_null


# --- external forward: train UCSF -> test UPENN (clin K80) ---
z = np.load(rf'{OUT}\pred_gnn_clin_only_k80.npz', allow_pickle=True)
tr_t, tr_e = z['ucsf_time'].astype(float), z['ucsf_event'].astype(bool)
te_t, te_e = z['upenn_time'].astype(float), z['upenn_event'].astype(bool)
teS = z['upenn_S'].astype(float)
brier_and_cal(tr_t, tr_e, te_t, te_e, teS, 'external-forward (UCSF train -> UPENN test)')

# --- external reverse: train UPENN -> test UCSF (clin K80) ---
z = np.load(rf'{OUT}\pred_gnn_rev_clin_only_k80.npz', allow_pickle=True)
tr_t, tr_e = z['upenn_time'].astype(float), z['upenn_event'].astype(bool)
te_t, te_e = z['ucsf_time'].astype(float), z['ucsf_event'].astype(bool)
teS = z['ucsf_S'].astype(float)
brier_and_cal(tr_t, tr_e, te_t, te_e, teS, 'external-reverse (UPENN train -> UCSF test)')

# --- internal 5-fold CV (out-of-fold) ---
z = np.load(rf'{OUT}\pred_clin_cv5.npz', allow_pickle=True)
t5, e5 = z['time'].astype(float), z['event'].astype(bool)
S5 = z['oob_S'].astype(float)
brier_and_cal(t5, e5, t5, e5, S5, 'internal 5-fold CV (UCSF oob)')
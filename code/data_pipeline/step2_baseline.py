"""New direction Step 2: clinical + whole-brain-pooled Cox baselines.

Before GNN training, establish the ladder with correct multi-timepoint evaluation:
  - clinical-only Cox (age/sex/GTR/MGMT)      [KPS dropped: 0%/12%; IDH dropped: cohort-constant]
  - whole-brain-pooled Cox (mean over 246 ROIs per modality, concat)

Eval: UCSF 5-fold CV C-index + external UPENN C-index + time-dependent (Uno) AUC
at {3,6,12,24,36} months.
"""
import os
import numpy as np
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.util import Surv
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold

OUT = os.path.join(os.environ.get('GBM_DATA',
                   os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')),
                   'new_direction')
TP = np.array([91.0, 182.0, 365.0, 730.0, 1095.0])
ALPHAS = np.logspace(-2.0, 0.5, 40)  # kept for reference; fit() uses adaptive 'auto' grid below


def load(name):
    z = np.load(rf'{OUT}\{name}.npz')
    return {k: z[k] for k in z.files}


def clin_X(z):
    return z['clin'].astype(float)[:, [0, 1, 2, 3]]  # age, sex_bin, gtr_bin, mgmt_bin


def wb_pooled_X(z):
    p1 = z['node_t1c'].mean(axis=1)  # (N,455)
    p2 = z['node_t2'].mean(axis=1)   # (N,497)
    p3 = z['node_flair'].mean(axis=1)  # (N,480)
    return np.concatenate([p1, p2, p3], axis=1)


def fit(X, t, e):
    imp = SimpleImputer(strategy='median').fit(X)
    sc = StandardScaler().fit(imp.transform(X))
    Xs = sc.transform(imp.transform(X))
    y = Surv.from_arrays(e, t)
    m = CoxnetSurvivalAnalysis(l1_ratio=0.9, n_alphas=40, alpha_min_ratio='auto')
    m.fit(Xs, y)
    return imp, sc, m, y


def apply_fit(fitobj, X):
    imp, sc, m, _ = fitobj
    return sc.transform(imp.transform(X)), m


def uno_auc(y_train, y_test, risk_test):
    res = cumulative_dynamic_auc(y_train, y_test, risk_test, TP)
    if isinstance(res, tuple):
        return res[0], None, None   # sksurv 0.23 returns (auc_per_t, integrated)
    if res.ndim == 2:
        return res[:, 0], res[:, 1], res[:, 2]
    return res, None, None


def run(name, X_builder, XUP):
    # ---- UCSF internal CV ----
    zu = load('UCSF_dev')
    Xi = X_builder(zu)
    t = zu['time'].astype(float)
    e = zu['event'].astype(bool)
    kf = KFold(n_splits=5, shuffle=True, random_state=20240813)
    civ = []
    for tr, va in kf.split(Xi):
        imp, sc, m, y = fit(Xi[tr], t[tr], e[tr])
        risk = m.predict(sc.transform(imp.transform(Xi[va])))
        civ.append(concordance_index_censored(e[va], t[va], risk)[0])
    print(f'[{name}] UCSF 5-fold CV C-index = {np.mean(civ):.4f} +- {np.std(civ):.4f}')

    # ---- external UCSF -> UPENN ----
    zup = load('UPENN_ext')
    f = fit(Xi, t, e)
    XUPt, m = apply_fit(f, XUP)
    risk_ext = m.predict(XUPt)
    eup = zup['event'].astype(bool)
    tup = zup['time'].astype(float)
    ci = concordance_index_censored(eup, tup, risk_ext)[0]
    # Uno AUC uses IPCW -> cap test follow-up at training horizon so censoring KM is estimable
    H = float(np.max(zu['time']))
    tup_c = np.minimum(tup, H)
    eup_c = eup & (tup < H)
    auc, lo, hi = uno_auc(f[3], Surv.from_arrays(eup_c, tup_c), risk_ext)
    print(f'[{name}] external UPENN C-index = {ci:.4f}')
    print(f'    Uno-AUC (3/6/12/24/36mo) = ' + ' | '.join(f'{a:.3f}' for a in auc))


if __name__ == '__main__':
    zu = load('UCSF_dev')
    zup = load('UPENN_ext')
    run('clinical-Cox', clin_X, clin_X(zup))
    run('wholebrain-pooled-Cox', wb_pooled_X, wb_pooled_X(zup))
"""New direction Step 1: build the R2SN + survival + clinical dataset.

Gate: before building, the run asserts key correctness invariants and prints
verification summaries, so we never waste downstream compute on bad data.

Outputs (saved as .npz so the GPU training env can load without pandas):
  data/new_direction/UCSF_dev.npz   (development, whole-brain R2SN + OS)
  data/new_direction/UPENN_ext.npz  (external, whole-brain R2SN + corrected censoring)
"""
import os
import numpy as np
import pandas as pd

NF = os.environ.get('R2SN_NODE', '')
EM = os.environ.get('R2SN_EDGE', '')
OUT = os.path.join(os.environ.get('GBM_DATA',
                   os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')),
                   'new_direction')
MODS = ['t1c', 't2', 'flair']
os.makedirs(OUT, exist_ok=True)
CLIN_COLS = ['age', 'sex_bin', 'gtr_bin', 'mgmt_bin', 'idh_wt', 'kps']


def load_r2sn(key):
    Xs, As = [], []
    for m in MODS:
        xf = os.path.join(NF, f'{key}_after_bn_{m}_selected_normed_rfeature.npy')
        af = os.path.join(EM, f'{key}_after_bn_{m}_pc_matrix.npy')
        assert os.path.exists(xf), f'missing node feature {xf}'
        assert os.path.exists(af), f'missing edge matrix {af}'
        X = np.load(xf).astype(np.float32)
        A = np.load(af).astype(np.float32)
        assert X.shape[0] == 246 and X.ndim == 2, f'bad X shape {X.shape} in {xf}'
        assert A.shape == (246, 246), f'bad A shape {A.shape} in {af}'
        assert np.isfinite(X).all() and np.isfinite(A).all(), f'non-finite in {key} {m}'
        Xs.append(X)
        As.append(A)
    return Xs, As


def build_ucsf():
    man = pd.read_csv(os.path.join(os.environ.get('MANIFEST_DIR', ''), 'manifest_UCSF.csv'))
    df = man[(man.grade == 4) & (man.idh == 'wildtype') & (man.is_followup == 0)].copy()
    df = df.sort_values('folder_id').reset_index(drop=True)
    time = pd.to_numeric(df.os_days, errors='coerce').values
    event = (df.status_dead == 1).astype(int).values
    # clinical
    age = pd.to_numeric(df.age, errors='coerce').values
    sex = df.sex.map({'M': 1, 'F': 0}).values
    gtr = df.eor.map({'GTR': 1, 'STR': 0, 'biopsy': 0}).values
    mgmt = df.mgmt_status.map({'positive': 1, 'negative': 0}).values
    idh_wt = np.ones(len(df))  # cohort is all wildtype
    kps = np.full(len(df), np.nan)
    clin = np.stack([age, sex, gtr, mgmt, idh_wt, kps], axis=1).astype(np.float32)
    kept = np.isfinite(time)
    return df.folder_id.values[kept], time[kept], event[kept], clin[kept]


def build_upenn():
    raw = pd.read_csv(os.path.join(os.environ.get('CLIN_CSV_DIR', ''), 'UPENN-GBM_clinical_info_v2.1.csv'),
                      encoding='utf-8-sig')
    man = pd.read_csv(os.path.join(os.environ.get('MANIFEST_DIR', ''), 'manifest_UPENN.csv'))
    raw['session'] = raw.ID.str[-2:]
    raw['base'] = raw.ID.str[:-3]
    event = (raw.Survival_Status.str.startswith('Deceased', na=False)
             & (raw.Survival_Status != 'Deceased - uncertain date of death'))
    uncertain = raw.Survival_Status == 'Deceased - uncertain date of death'
    time = np.where(event,
                    pd.to_numeric(raw['Survival_from_surgery_days_UPDATED'], errors='coerce'),
                    pd.to_numeric(raw['Survival_Censor'], errors='coerce'))
    raw['time_days'] = time
    raw['event'] = event.astype(int)
    raw['uncertain'] = uncertain.astype(int)
    pre = raw[(raw.session == '11') & (~uncertain)].copy()
    # merge clinical from manifest (sex/age/mgmt/gtr90/kps/idh1)
    man2 = man[['patient_id', 'sex', 'age', 'mgmt', 'gtr90', 'kps', 'idh1']].copy()
    pre = pre.merge(man2, left_on='ID', right_on='patient_id', how='left')
    pre = pre.sort_values('base').reset_index(drop=True)
    age = pd.to_numeric(pre.age, errors='coerce').values
    sex = pre.sex.map({'M': 1, 'F': 0}).values
    gtr = pre.gtr90.map({'Y': 1, 'N': 0}).values
    mgmt = pre.mgmt.map({'Methylated': 1, 'Unmethylated': 0}).values
    idh = pre.idh1.map({'Wildtype': 1, 'Mutated': 0, 'NOS/NEC': np.nan}).values
    kps = pd.to_numeric(pre.kps, errors='coerce').values
    clin = np.stack([age, sex, gtr, mgmt, idh, kps], axis=1).astype(np.float32)
    return pre.base.values, pre.time_days.values, pre.event.values, clin


def assemble(keys, time, event, clin, name):
    Xs, As = [], []
    for k in keys:
        x, a = load_r2sn(k)
        Xs.append(x)
        As.append(a)
    Xs = [np.stack([xs[i] for xs in Xs]) for i in range(3)]  # per modality (N,246,d)
    As = [np.stack([as_[i] for as_ in As]) for i in range(3)]
    assert len(time) == len(keys) == len(event) == len(clin)
    assert np.isfinite(time).all() and (time > 0).all(), 'time must be finite & positive'
    assert set(np.unique(event)) <= {0, 1}
    np.savez_compressed(os.path.join(OUT, name + '.npz'),
                        node_t1c=Xs[0], node_t2=Xs[1], node_flair=Xs[2],
                        adj_t1c=As[0], adj_t2=As[1], adj_flair=As[2],
                        time=time.astype(np.float32), event=event.astype(np.int8),
                        clin=clin, pids=keys.astype(str),
                        clin_cols=np.array(CLIN_COLS))
    print(f'[{name}] n={len(keys)} events={int(event.sum())} censored={int((1-event).sum())} '
          f'| node feat dims t1c/t2/flair = {Xs[0].shape[2]}/{Xs[1].shape[2]}/{Xs[2].shape[2]} (x{Xs[0].shape[1]} ROIs)')
    # clinical missing report
    for i, c in enumerate(CLIN_COLS):
        print(f'    {c}: non-missing {int((~np.isnan(clin[:, i])).sum())}/{len(keys)}')
    print(f'    saved {name}.npz with keys {list(keys[:2])}...')


if __name__ == '__main__':
    uk, ut, ue, uc = build_ucsf()
    assemble(uk, ut, ue, uc, 'UCSF_dev')
    pk, pt, pe, pc = build_upenn()
    assemble(pk, pt, pe, pc, 'UPENN_ext')
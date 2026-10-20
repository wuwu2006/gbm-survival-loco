"""Generate small SYNTHETIC cohorts matching the real npz schemas (pipeline smoke tests only).

The numbers are random and meaningless; the files exist only so that
code/model (training, LOCO, metrics) can be exercised end-to-end without
any real patient data.

Usage:
    python tools/make_demo_data.py

Writes (overwrites):
    <GBM_DATA | <repo>/data>/new_direction/UCSF_dev.npz
    <GBM_DATA | <repo>/data>/new_direction/UPENN_ext.npz
"""
import os
import numpy as np

OUT = os.path.join(os.environ.get(
    'GBM_DATA',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')),
    'new_direction')

N = 24
RNG = np.random.RandomState(20240813)
NODE_DIMS = {'UCSF_dev': [455, 497, 480], 'UPENN_ext': [455, 497, 480]}
CLIN_COLS = ['age', 'sex_bin', 'gtr_bin', 'mgmt_bin', 'idh_wt', 'kps']


def make_adj(n):
    A = RNG.randn(n, 246, 246)
    A = (A + A.transpose(0, 2, 1)) / 2.0
    idx = np.arange(246)
    A[:, idx, idx] = 0.0
    return A.astype(np.float32)


def make_cohort(name):
    n = N
    pids = np.array([f'{name}-{i + 1:05d}' for i in range(n)])
    d1, d2, d3 = NODE_DIMS[name]
    node_t1c = RNG.randn(n, 246, d1).astype(np.float32)
    node_t2 = RNG.randn(n, 246, d2).astype(np.float32)
    node_flair = RNG.randn(n, 246, d3).astype(np.float32)
    clin = np.zeros((n, 6), np.float32)
    clin[:, 0] = RNG.uniform(20, 85, n)           # age
    clin[:, 1] = RNG.randint(0, 2, n)             # sex
    clin[:, 2] = RNG.randint(0, 2, n)             # GTR
    clin[:, 3] = RNG.randint(0, 2, n)             # MGMT
    clin[:, 4] = 1.0                              # IDH wildtype
    clin[:, 5] = np.nan                           # KPS missing (like the real cohorts)
    time = RNG.uniform(30, 3600, n)
    event = (RNG.rand(n) < 0.6).astype(bool)
    np.savez_compressed(
        os.path.join(OUT, f'{name}.npz'),
        node_t1c=node_t1c, node_t2=node_t2, node_flair=node_flair,
        adj_t1c=make_adj(n), adj_t2=make_adj(n), adj_flair=make_adj(n),
        time=time.astype(np.float64), event=event,
        clin=clin, clin_cols=np.array(CLIN_COLS), pids=pids)
    print(f'wrote {name}.npz  n={n}  events={int(event.sum())}  '
          f'node dims {d1}/{d2}/{d3}')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    make_cohort('UCSF_dev')
    make_cohort('UPENN_ext')
    print('Demo data written. Numbers are random; do NOT use for any scientific claim.')
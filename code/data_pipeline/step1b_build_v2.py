"""New direction: build 93-feature node-feature dataset variant.

Node features from the 93-feature variant library ({pid}_{mod}_radiomics.npy, 246x93 per modality),
adjacency reused from the standard edge matrices (same 246 ROI parcellation).
Survival/clinical identical to the base npz (same cohorts).
"""
import os
import numpy as np

NF2 = os.environ.get('R2SN_V2_NODE', '')
EM = os.environ.get('R2SN_EDGE', '')
OUT = os.path.join(os.environ.get('GBM_DATA',
                   os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')),
                   'new_direction')
MODS = ['t1c', 't2', 'flair']

for name in ['UCSF_dev', 'UPENN_ext']:
    z = np.load(rf'{OUT}\{name}.npz', allow_pickle=True)
    pids = z['pids']
    N = len(pids)
    node = {m: np.zeros((N, 246, 93), np.float32) for m in MODS}
    adj = {m: np.zeros((N, 246, 246), np.float32) for m in MODS}
    dims_seen = set()
    for i, pid in enumerate(pids):
        for m in MODS:
            nf = os.path.join(NF2, f'{pid}_{m}_radiomics.npy')
            assert os.path.exists(nf), f'missing v2 node feature {nf}'
            X = np.load(nf).astype(np.float32)
            dims_seen.add(X.shape[1])
            assert X.shape[0] == 246 and X.ndim == 2, f'bad v2 shape {X.shape} in {nf}'
            node[m][i] = X
            af = os.path.join(EM, f'{pid}_after_bn_{m}_pc_matrix.npy')
            adj[m][i] = np.load(af).astype(np.float32)
    print(f'[{name}_v2] n={N} dims_seen={sorted(dims_seen)}')
    np.savez_compressed(rf'{OUT}\{name}_v2.npz',
                        node_t1c=node['t1c'], node_t2=node['t2'], node_flair=node['flair'],
                        adj_t1c=adj['t1c'], adj_t2=adj['t2'], adj_flair=adj['flair'],
                        time=z['time'], event=z['event'], clin=z['clin'], pids=pids, clin_cols=z['clin_cols'])
    print(f'    saved {name}_v2.npz')
"""New direction Step 3 training (pure torch, GPU in cm-gnn-py310).

Run order (each gated):
  python step3_train.py --smoke     # CPU: real data load + 1 fwd/bwd, assert no NaN (run before GPU)
  python step3_train.py --train     # GPU: train on UCSF (80/20 early stop) -> save preds for UCSF+UPENN
  python step3_eval.py              # (radiomics_lc) C-index + Uno-AUC@5 timepoints + conditional survival
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from step3_model import MultiTimeModel, discrete_nll, survival_from_hazards

OUT = os.path.join(os.environ.get('GBM_DATA',
                   os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')),
                   'new_direction')
K = 40
T_MAX = 2144.0          # UCSF max observed OS (days); UPENN capped at this for IPCW
W = T_MAX / K
LANDMARKS = np.array([91.0, 182.0, 365.0, 730.0, 1095.0])
CLIN_IDX = [0, 1, 2, 3]   # age, sex_bin, gtr_bin, mgmt_bin (drop idh=const, kps=missing)


def load_npz(name):
    z = np.load(rf'{OUT}\{name}.npz', allow_pickle=True)
    return {k: z[k] for k in z.files}


def clin_prep(train_clin, apply_clin):
    """impute (median) + zscore age; fit on train_clin, apply to both."""
    T = train_clin[:, CLIN_IDX].astype(np.float32)
    A = apply_clin[:, CLIN_IDX].astype(np.float32)
    imp = np.nanmedian(T, axis=0)
    age = T[:, 0]
    mu, sd = float(np.nanmean(age)), float(np.nanstd(age)) + 1e-8

    def _f(C):
        X = np.where(np.isnan(C), imp, C).astype(np.float32).copy()
        X[:, 0] = (X[:, 0] - mu) / sd
        return X
    return _f(T), _f(A)


def node_norm_fit(X):
    flat = X.reshape(-1, X.shape[-1])
    return flat.mean(0), flat.std(0)


def node_norm_apply(X, mu, sd):
    sd = np.where(sd == 0, 1.0, sd)
    return ((X - mu) / sd).astype(np.float32)


def survival_at(hazards, times):  # times: (L,) days -> (B,L) interpolated log-S
    S = survival_from_hazards(hazards)  # (B,K)
    B, Kk = S.shape
    logS = torch.cat([torch.zeros(B, 1, device=S.device), S.log()], dim=1)  # (B,K+1); pos 0..K
    p = torch.as_tensor(times, dtype=torch.float32, device=S.device) / W
    p = p.clamp(0, Kk)
    lo = torch.floor(p).long()
    frac = (p - lo).unsqueeze(0)
    lo_c = lo.clamp(0, Kk - 1)
    v = logS[:, lo_c] * (1 - frac) + logS[:, lo_c + 1] * frac  # (B,L)
    return v.exp()


def to_tensor_list(arrays, dev):
    return [torch.from_numpy(a.astype(np.float32)).to(dev) for a in arrays]


def nodewise(z):  # z: list of 3 (N,246,d)
    return z


# ---------------- smoke gate ----------------
def smoke(dev='cpu'):
    u = load_npz('UCSF_dev')
    N = 8
    Xs = [u['node_t1c'][:N], u['node_t2'][:N], u['node_flair'][:N]]
    As = [u['adj_t1c'][:N], u['adj_t2'][:N], u['adj_flair'][:N]]
    clin = u['clin'][:N]
    # normalize with stats from this tiny slice
    Xsn, Asn = [], []
    for x in Xs:
        mu, sd = node_norm_fit(x)
        Xsn.append(node_norm_apply(x, mu, sd))
    Asn = [a.astype(np.float32) for a in As]
    ctr, cte = clin_prep(clin, clin)
    t = u['time'][:N].astype(float)
    e = u['event'][:N].astype(bool)
    bin_e, n_c = None, None
    from step3_model import bins_and_censor_counts
    bin_e, n_c = bins_and_censor_counts(t, e, K, T_MAX)

    Xt = to_tensor_list(Xsn, dev)
    At = to_tensor_list(Asn, dev)
    ct = torch.from_numpy(ctr).to(dev)
    model = MultiTimeModel([[x.shape[2] for x in Xsn][0], [x.shape[2] for x in Xsn][1], [x.shape[2] for x in Xsn][2]],
                           K, D=8, Dg=8).to(dev)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    h = model(Xt, At, ct)
    loss = discrete_nll(h, bin_e, n_c, e)
    loss.backward()
    ok = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
    S = survival_from_hazards(h)
    mono = bool((S.diff(dim=1) <= 1e-6).all().item())
    St = survival_at(h, LANDMARKS)
    print(f'[smoke] N={N} hazards{h.shape} loss={loss.item():.4f} grads_finite={ok} monotone={mono}')
    print(f'[smoke] S@landmarks shape {St.shape} values {St[0].detach().cpu().numpy().round(3)}')
    assert ok and mono and torch.isfinite(loss)
    print('[smoke] PASS on real data')


# ---------------- train ----------------
def numpy_cindex(event, time, risk):
    order = np.argsort(risk)
    event, time, risk = np.asarray(event)[order], np.asarray(time)[order], np.asarray(risk)[order]
    n, s = 0, 0.0
    for i in range(len(event)):
        if not event[i]:
            continue
        for j in range(i + 1, len(event)):
            if risk[i] == risk[j]:
                continue
            n += 1
            if time[i] < time[j]:
                s += float(risk[i] > risk[j]) + 0.5 * float(risk[i] == risk[j])
            elif time[j] < time[i]:
                s += float(risk[j] > risk[i]) + 0.5 * float(risk[j] == risk[i])
    return s / n if n else 0.5


def rank_loss2(S, t, e, margin=0.0):
    """Bounded pairwise concordance surrogate on survival curves.
    S: (B,K) survival; pair i(event at t_i), j with t_j > t_i:
    want S_j(t_i) > S_i(t_i) (i more likely dead by t_i). d = S_j(t_i)-S_i(t_i) in [-1,1].
    """
    B, K = S.shape
    b = torch.floor(t / max(W, 1e-6)).long().clamp(0, K - 1)  # (B,) bin of each t
    Sb = S.gather(1, b.unsqueeze(1)).squeeze(1)               # S_k(t_k)
    S_at_events = S[:, b]                                    # (B,B): S_j(t_i)
    d = S_at_events.transpose(0, 1) - Sb.unsqueeze(1)        # d[i,j] = S_j(t_i)-S_i(t_i)
    mask = (e.unsqueeze(1) & (t.unsqueeze(0) > t.unsqueeze(1))
            & (~torch.eye(B, dtype=torch.bool, device=S.device)))
    n = mask.sum().clamp(min=1)
    return (torch.relu(margin - d) * mask).sum() / n


def make_batch(Xsn, Asn, clin_n, idx, dev):
    X = [torch.from_numpy(x[idx].astype(np.float32)).to(dev) for x in Xsn]
    A = [torch.from_numpy(a[idx].astype(np.float32)).to(dev) for a in Asn]
    c = torch.from_numpy(clin_n[idx]).to(dev)
    return X, A, c


def train_and_save(dev, mode='full', reverse=False, rank_w=0.0, tag='', data=''):
    global W, T_MAX
    torch.manual_seed(0)
    np.random.seed(0)
    suf = '_v2' if data == 'v2' else ''
    if reverse:
        TRAIN, TEST = 'UPENN_ext' + suf, 'UCSF_dev' + suf
    else:
        TRAIN, TEST = 'UCSF_dev' + suf, 'UPENN_ext' + suf
    u = load_npz(TRAIN)
    p = load_npz(TEST)
    T_MAX = float(np.max(u['time']))
    W = T_MAX / K

    # normalize node features with UCSF stats
    Xsn = []
    Asn = []
    for nm in ['t1c', 't2', 'flair']:
        mu, sd = node_norm_fit(u[f'node_{nm}'])
        Xsn.append(node_norm_apply(u[f'node_{nm}'], mu, sd))
        Asn.append(u[f'adj_{nm}'].astype(np.float32))
    clin_tr, clin_te = clin_prep(u['clin'], p['clin'])
    if mode == 'graph_only':
        clin_tr = np.zeros_like(clin_tr)
        clin_te = np.zeros_like(clin_te)
    if mode == 'clin_only':
        Xsn = [np.zeros_like(x) for x in Xsn]
        Asn = [np.zeros_like(a) for a in Asn]

    t_u = u['time'].astype(float)
    e_u = u['event'].astype(bool)
    t_p = p['time'].astype(float)
    e_p = p['event'].astype(bool)
    from step3_model import bins_and_censor_counts
    bin_e_u, n_c_u = bins_and_censor_counts(t_u, e_u, K, T_MAX)

    N = len(t_u)
    perm = np.random.permutation(N)
    n_tr = int(N * 0.8)
    tr_idx = perm[:n_tr]
    va_idx = perm[n_tr:]

    dims = [Xsn[0].shape[2], Xsn[1].shape[2], Xsn[2].shape[2]]
    model = MultiTimeModel(dims, K, n_clin=4, D=64, Dg=64).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss, best_state, bad = 1e9, None, 0

    def nll_of(idx):
        model.eval()
        X, A, c = make_batch(Xsn, Asn, clin_tr, idx, dev)
        with torch.no_grad():
            h = model(X, A, c)
        return discrete_nll(h, bin_e_u[idx], n_c_u[idx], e_u[idx]).item()

    B = 16
    for ep in range(300):
        model.train()
        perm_ep = np.random.permutation(tr_idx)
        for s in range(0, len(perm_ep), B):
            idx = perm_ep[s:s + B]
            X, A, c = make_batch(Xsn, Asn, clin_tr, idx, dev)
            h = model(X, A, c)
            loss = discrete_nll(h, bin_e_u[idx], n_c_u[idx], e_u[idx])
            if rank_w > 0:
                S = survival_from_hazards(h)
                loss = loss + rank_w * rank_loss2(
                    S,
                    torch.as_tensor(t_u[idx], dtype=torch.float32, device=dev),
                    torch.as_tensor(e_u[idx], dtype=torch.bool, device=dev))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        vl = nll_of(va_idx)
        if vl < best_loss - 1e-5:
            best_loss, best_state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 25:
                print(f'[train] early stop epoch {ep}')
                break
        if ep % 20 == 0:
            print(f'[train] ep {ep} valNLL {vl:.4f} best {best_loss:.4f}')
    model.load_state_dict(best_state)
    model.eval()

    # predictions (risk = -mean survival days)
    def pred_batch(X, A, c, idx):
        Xb, Ab, cb = make_batch(X, A, c, idx, dev)
        with torch.no_grad():
            h = model(Xb, Ab, cb)
        S = survival_from_hazards(h)  # (B,K)
        meanT = (S * W).sum(1)
        risk = -meanT
        St = survival_at(h, LANDMARKS)  # (B,L)
        return risk.cpu().numpy(), St.cpu().numpy()

    def run_set(X, A, c, n):
        rk = np.zeros(n, dtype=np.float32)
        Sm = np.zeros((n, 5), dtype=np.float32)
        for s in range(0, n, 64):
            idx = np.arange(s, min(s + 64, n), dtype=np.int64)
            rk[idx], Sm[idx] = pred_batch(X, A, c, idx)
        return rk, Sm

    # UPENN nodes normalized with UCSF training stats (correct external eval)
    Xsn_p = []
    for nm in ['t1c', 't2', 'flair']:
        muf, sdf = node_norm_fit(u[f'node_{nm}'])
        Xsn_p.append(node_norm_apply(p[f'node_{nm}'], muf, sdf))
    Asn_p = [p['adj_t1c'].astype(np.float32), p['adj_t2'].astype(np.float32), p['adj_flair'].astype(np.float32)]
    if mode == 'clin_only':
        Xsn_p = [np.zeros_like(x) for x in Xsn_p]
        Asn_p = [np.zeros_like(a) for a in Asn_p]

    rk_u, Sm_u = run_set(Xsn, Asn, clin_tr, N)
    rk_p, Sm_p = run_set(Xsn_p, Asn_p, clin_te, len(t_p))

    # internal val C-index (monitor)
    rk_va = rk_u[va_idx]
    ci_va = numpy_cindex(e_u[va_idx], t_u[va_idx], rk_va)
    print(f'[train] internal val C-index = {ci_va:.4f} (n={len(va_idx)})')

    if reverse:
        ucsf_rk, ucsf_Sm, ucsf_tt, ucsf_ee = rk_p, Sm_p, t_p.astype(np.float32), e_p.astype(np.int8)
        upenn_rk, upenn_Sm, upenn_tt, upenn_ee = rk_u, Sm_u, t_u.astype(np.float32), e_u.astype(np.int8)
    else:
        ucsf_rk, ucsf_Sm, ucsf_tt, ucsf_ee = rk_u, Sm_u, t_u.astype(np.float32), e_u.astype(np.int8)
        upenn_rk, upenn_Sm, upenn_tt, upenn_ee = rk_p, Sm_p, t_p.astype(np.float32), e_p.astype(np.int8)
    tag_s = f'_{tag}' if tag else ''
    if reverse:
        out_name = f'pred_gnn_rev_{mode}{tag_s}.npz'
    elif mode != 'full':
        out_name = f'pred_gnn_{mode}{tag_s}.npz'
    else:
        out_name = f'pred_gnn{tag_s}.npz'
    np.savez_compressed(rf'{OUT}\{out_name}',
                        ucsf_pids=u['pids'] if not reverse else p['pids'],
                        ucsf_time=ucsf_tt, ucsf_event=ucsf_ee,
                        ucsf_S=ucsf_Sm, ucsf_risk=ucsf_rk,
                        upenn_pids=p['pids'] if not reverse else u['pids'],
                        upenn_time=upenn_tt, upenn_event=upenn_ee,
                        upenn_S=upenn_Sm, upenn_risk=upenn_rk,
                        landmarks=LANDMARKS)
    print(f'[train] saved {out_name}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--mode', default='full', choices=['full', 'graph_only', 'clin_only'])
    ap.add_argument('--reverse', action='store_true')
    ap.add_argument('--K', type=int, default=40)
    ap.add_argument('--rank', type=float, default=0.0)
    ap.add_argument('--tag', default='')
    ap.add_argument('--data', default='', choices=['', 'v2'])
    ap.add_argument('--cuda', action='store_true')
    args = ap.parse_args()
    if args.smoke:
        dev = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
        smoke(dev)
    elif args.train:
        K = args.K
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('[train] device =', dev, 'mode =', args.mode, 'reverse =', args.reverse,
              'K =', K, 'rank =', args.rank, 'tag =', repr(args.tag))
        train_and_save(dev, args.mode, args.reverse, rank_w=args.rank, tag=args.tag, data=args.data)
    else:
        print('usage: --smoke | --train')
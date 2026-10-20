"""Step 5: LOCO leave-one-cohort-out for SwinUNETR/SALP features (S_swin_salp).

Train head on cohort A (days, cap 120mo), test externally on cohort B (days).
Variants: salp_only (z_salp 1152), salp_full (z_salp + tumor_global + clinical(age,sex) 1159).
Runs in cm-gnn-py310. Eval (C-index + Uno-AUC@6 nodes) done in radiomics_lc.
"""
import os
import numpy as np
import torch
import torch.nn as nn

OUT = os.path.join(os.environ.get('GBM_DATA',
                   os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')),
                   'new_direction')
FE = os.environ.get('SALP_FEATURES', '')  # dir with {pid}.npz SwinUNETR/SALP feature library (proprietary inputs)
LM = np.array([91.0, 182.0, 365.0, 730.0, 1095.0, 1460.0])  # 3/6/12/24/36/48 mo
CAP = 3652.5  # 120 months in days
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


class Head(nn.Module):
    def __init__(self, din, K):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, K))

    def forward(self, x):
        return self.net(x)


def load_cohort(name):
    z = np.load(rf'{OUT}\{name}.npz', allow_pickle=True)
    return z['pids'], z['time'].astype(np.float64), z['event'].astype(bool)


def load_feats(pids, use_clin):
    Z, T, C = [], [], []
    for p in pids:
        f = np.load(rf'{FE}\{p}.npz', allow_pickle=True)
        Z.append(f['z_salp'])
        T.append(f['tumor_global'])
        C.append(f['clinical'])
    Z, T, C = (np.stack(a).astype(np.float32) for a in (Z, T, C))
    return np.concatenate([Z, T, C], axis=1) if use_clin else Z


def edges_from_events(ev, cap=CAP):
    q = np.quantile(np.sort(ev.astype(np.float64)), np.linspace(0, 1, 13))
    return np.unique(np.concatenate([[0.0], q, [cap]]))


def nll(logits, t, e, edges):
    K = logits.shape[1]
    lam = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
    log_s = torch.log1p(-lam)
    cum = torch.cumsum(log_s, dim=1)
    cum_prev = torch.cat([torch.zeros_like(cum[:, :1]), cum[:, :-1]], dim=1)
    b = torch.searchsorted(torch.as_tensor(edges, device=logits.device), t, side='right') - 1
    b = b.clamp(0, K - 1)
    loss = torch.empty_like(t, dtype=torch.float32)
    ge = torch.gather(lam, 1, b.unsqueeze(1)).squeeze(1)
    loss[e] = -(torch.log(ge[e].clamp(min=1e-7)) + cum_prev[e, b[e]])
    loss[~e] = -cum_prev[~e, b[~e]]
    return loss.mean()


def survival_and_risk(logits, edges):
    lam = torch.sigmoid(logits)
    S = torch.cumprod(1 - lam, dim=1)
    ed = torch.as_tensor(edges, device=logits.device)
    w = (ed[1:] - ed[:-1]).unsqueeze(0)
    return S, -(S * w).sum(1)


def survival_at(logits, edges, times):
    lam = torch.sigmoid(logits)
    S = torch.cumprod(1 - lam, dim=1)
    ed = torch.as_tensor(edges, device=logits.device)
    logS = torch.cat([torch.zeros_like(S[:, :1]), torch.log(S.clamp(min=1e-9))], dim=1)
    t = torch.as_tensor(times, dtype=torch.float32, device=logits.device)
    pos = torch.searchsorted(ed, t, side='left').clamp(1, len(edges) - 1)
    lo = pos - 1
    frac = (t - ed[lo]) / (ed[pos] - ed[lo]).clamp(min=1e-9)
    return torch.exp(logS[:, lo] * (1 - frac) + logS[:, pos] * frac)


def train_one(Xtr, ttr, etr, Xte, name, tag, seed=0, max_ep=200, patience=25, B=64):
    ttr = np.minimum(ttr, CAP)  # cap120
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(Xtr)
    perm = np.random.permutation(n)
    nmon = max(8, n // 20)
    mon, sub = perm[:nmon], perm[nmon:]
    mu, sd = Xtr[sub].mean(0), Xtr[sub].std(0) + 1e-8
    Xs = (Xtr - mu) / sd
    edges = edges_from_events(ttr[sub])
    K = len(edges) - 1
    print(f'[{name}] K={K} edges[:3]={edges[:3]} edges[-2:]={edges[-2:]} dim={Xtr.shape[1]}', flush=True)
    model = Head(Xtr.shape[1], K).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    Xt = torch.from_numpy(Xs).to(DEV)
    Xm = torch.from_numpy((Xtr[mon] - mu) / sd).to(DEV)
    tt = torch.from_numpy(ttr).to(DEV)
    et = torch.from_numpy(etr).to(DEV)
    tmon = tt[mon]
    emon = torch.from_numpy(etr[mon]).to(DEV)
    best, best_state, bad = 1e9, None, 0
    for ep in range(max_ep):
        model.train()
        rp = np.random.permutation(sub)
        for s in range(0, len(rp), B):
            idx = rp[s:s + B]
            h = model(Xt[idx])
            loss = nll(h, tt[idx], et[idx], edges)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vm = nll(model(Xm), tmon, emon, edges).item()
        if vm < best - 1e-5:
            best, best_state, bad = vm, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        hte = model(torch.from_numpy((Xte - mu) / sd).to(DEV))
        S_te, risk_te = survival_and_risk(hte, edges)
        St = survival_at(hte, edges, LM)
    print(f'[{name}] best valNLL={best:.4f} ep={ep + 1}', flush=True)
    return risk_te.cpu().numpy(), St.cpu().numpy(), K, edges


def main():
    u_pids, u_t, u_e = load_cohort('UCSF_dev')
    p_pids, p_t, p_e = load_cohort('UPENN_ext')
    U = (u_pids, u_t, u_e)
    P = (p_pids, p_t, p_e)
    for variant, use_clin in [('salp_only', False), ('salp_full', True)]:
        for direction, (tr, te) in [('fwd', (U, P)), ('rev', (P, U))]:
            Xtr = load_feats(tr[0], use_clin)
            Xte = load_feats(te[0], use_clin)
            risk, St, K, edges = train_one(Xtr, tr[1], tr[2], Xte, f'{variant}_{direction}', '')
            np.savez(rf'{OUT}\pred_loco_{direction}_{variant}.npz',
                     test_pids=te[0], test_time=te[1], test_event=te[2].astype(np.int8),
                     risk=risk, S=St, landmarks=LM, n_bins=K, edges=edges)
            print(f'saved pred_loco_{direction}_{variant}.npz', flush=True)


if __name__ == '__main__':
    main()
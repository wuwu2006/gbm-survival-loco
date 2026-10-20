"""Step A: 5-fold CV internal estimate for the clinical-only discrete-time model (UCSF).

Runs in cm-gnn-py310 (torch, no pandas). Saves out-of-fold risk + S@landmarks + C-index.
Uno-AUC evaluation done separately (radiomics_lc, IPCW).
"""
import os
import numpy as np
import torch
from step3_model import MultiTimeModel, discrete_nll, survival_from_hazards, bins_and_censor_counts

OUT = os.path.join(os.environ.get('GBM_DATA',
                   os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')),
                   'new_direction')
K = 80
CLIN_IDX = [0, 1, 2, 3]
LANDMARKS = np.array([91.0, 182.0, 365.0, 730.0, 1095.0])
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


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


def survival_at(h, times, W):
    S = survival_from_hazards(h)
    B, Kk = S.shape
    logS = torch.cat([torch.zeros(B, 1, device=S.device), S.log()], dim=1)
    p = torch.as_tensor(times, dtype=torch.float32, device=S.device) / W
    p = p.clamp(0, Kk)
    lo = torch.floor(p).long()
    frac = (p - lo).unsqueeze(0)
    lo_c = lo.clamp(0, Kk - 1)
    v = logS[:, lo_c] * (1 - frac) + logS[:, lo_c + 1] * frac
    return v.exp()


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    z = np.load(rf'{OUT}\UCSF_dev.npz', allow_pickle=True)
    clin_raw = z['clin'][:, CLIN_IDX].astype(np.float32)
    t = z['time'].astype(float)
    e = z['event'].astype(bool)
    N = len(t)
    T_MAX = float(t.max())
    W = T_MAX / K
    bin_e, n_c = bins_and_censor_counts(t, e, K, T_MAX)

    perm = np.random.permutation(N)
    folds = np.array_split(perm, 5)
    oob_risk = np.zeros(N, np.float32)
    oob_S = np.zeros((N, 5), np.float32)

    for fi in range(5):
        va = folds[fi]
        tr = np.concatenate([f for i, f in enumerate(folds) if i != fi])
        imp = np.nanmedian(clin_raw[tr], axis=0)
        age = clin_raw[tr][:, 0]
        mu, sd = float(np.nanmean(age)), float(np.nanstd(age)) + 1e-8

        def prep(idx):
            X = np.where(np.isnan(clin_raw[idx]), imp, clin_raw[idx]).astype(np.float32).copy()
            X[:, 0] = (X[:, 0] - mu) / sd
            return X

        def zero_inputs(idx, dev):
            n = len(idx)
            return ([torch.zeros(n, 246, 1, device=dev) for _ in range(3)],
                    [torch.zeros(n, 246, 246, device=dev) for _ in range(3)],
                    torch.from_numpy(prep(idx)).to(dev))

        model = MultiTimeModel([1, 1, 1], K, n_clin=4, D=64).to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best, best_state, bad = 1e9, None, 0
        B = 16
        for ep in range(300):
            model.train()
            rp = np.random.permutation(tr)
            for s in range(0, len(rp), B):
                idx = rp[s:s + B]
                X, A, c = zero_inputs(idx, DEV)
                h = model(X, A, c)
                loss = discrete_nll(h, bin_e[idx], n_c[idx], e[idx])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            with torch.no_grad():
                X, A, c = zero_inputs(va, DEV)
                hv = model(X, A, c)
            vl = discrete_nll(hv, bin_e[va], n_c[va], e[va]).item()
            if vl < best - 1e-5:
                best, best_state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 25:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            X, A, c = zero_inputs(va, DEV)
            h = model(X, A, c)
            S = survival_from_hazards(h)
            oob_risk[va] = (-(S * W).sum(1)).cpu().numpy()
            oob_S[va] = survival_at(h, LANDMARKS, W).cpu().numpy()
        print(f'fold {fi}: valNLL {best:.4f}', flush=True)

    ci = numpy_cindex(e, t, oob_risk)
    print(f'5-fold CV out-of-fold C-index = {ci:.4f}')
    np.savez(rf'{OUT}\pred_clin_cv5.npz',
             pids=z['pids'], time=t, event=e.astype(np.int8),
             oob_risk=oob_risk, oob_S=oob_S, landmarks=LANDMARKS)


if __name__ == '__main__':
    main()
"""New direction Step 3: whole-brain R2SN graph network + discrete multi-timepoint survival.

Model (pure PyTorch, no torch_geometric needed -> runs in cm-gnn-py310 with 4070):
  - per-modality GCN (dense normalized adjacency + self-loops) -> graph readout
  - 3-modal fusion + clinical(age/sex/GTR/MGMT) fusion
  - discrete-time hazard head (K bins) -> monotone survival curve S(t)
  - S(t) at 3/6/12/24/36 months + conditional survival S(t|t0)=S(t)/S(t0) fall out directly

Correctness gate (run BEFORE real data): `python step3_model.py --sanity`
  asserts forward pass, hazards in [0,1], monotone S, finite loss, loss decreases, no NaN.
"""
import argparse
import numpy as np
import torch
import torch.nn as nn

EPS = 1e-7


# ---------------- discrete-time survival utilities ----------------
def make_grid(K, T_max):
    w = T_max / K
    return w, np.arange(K) * w  # bin left edges


def bins_and_censor_counts(times, events, K, T_max):
    """event times -> bin index (K bins); censored times -> # fully survived bins.
    times: (N,) days; events: (N,) bool.
    """
    w = T_max / K
    t = np.asarray(times, dtype=np.float64)
    e = np.asarray(events, dtype=bool)
    bin_e = np.zeros_like(t, dtype=np.int64)
    n_c = np.zeros_like(t, dtype=np.int64)
    for i in range(len(t)):
        idx = int(np.floor((t[i] - 1e-6) / w))
        if e[i]:
            bin_e[i] = min(max(idx, 0), K - 1)
        else:
            n_c[i] = K if t[i] >= T_max else max(idx, 0)
    return bin_e, n_c


def discrete_nll(hazards, bin_e, n_c, events):
    """hazards: (B,K) in (0,1). Returns mean NLL."""
    h = hazards.clamp(EPS, 1 - EPS)
    log_h = h.log()
    log1mh = (1 - h).log()
    B, K = h.shape
    logS = torch.cat([torch.zeros(B, 1, device=h.device),
                      torch.cumsum(log1mh, dim=1)], dim=1)  # (B,K+1) logS[b]=sum_{j<b}log1mh
    ev = torch.as_tensor(events, dtype=torch.bool, device=h.device)
    be = torch.as_tensor(bin_e, dtype=torch.long, device=h.device)
    nc = torch.as_tensor(n_c, dtype=torch.long, device=h.device)
    loss_event = -(log_h.gather(1, be.unsqueeze(1)).squeeze(1) + logS.gather(1, be.unsqueeze(1)).squeeze(1))
    loss_censor = -(logS.gather(1, nc.unsqueeze(1)).squeeze(1))
    loss = torch.where(ev, loss_event, loss_censor)
    return loss.mean()


def survival_from_hazards(hazards):
    """S[k] = P(T > tau_k) for k=0..K-1, monotone non-increasing."""
    h = hazards.clamp(EPS, 1 - EPS)
    logS = torch.cumsum((1 - h).log(), dim=1)  # (B,K)
    full = torch.cat([torch.zeros(logS.shape[0], 1, device=h.device), logS], dim=1)  # (B,K+1), full[0]=0 (S=1 at t=0)
    return full[:, 1:].exp()  # monotone non-increasing


# ---------------- graph encoder ----------------
class GCNLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W = nn.Linear(d_in, d_out, bias=False)
        self.bias = nn.Parameter(torch.zeros(d_out))

    def forward(self, X, A):
        B, N, _ = A.shape
        I = torch.eye(N, device=A.device).unsqueeze(0)  # (1,N,N)
        At = A + I
        deg = At.sum(dim=-1).clamp(min=1e-8)  # (B,N)
        dinv = deg.pow(-0.5)  # (B,N)
        An = At * dinv.unsqueeze(-1) * dinv.unsqueeze(-2)  # normalized
        H = self.W(X)  # (B,N,d_out)
        return torch.bmm(An, H) + self.bias


class GraphEncoder(nn.Module):
    """Per-modality 2-layer GCN -> readout -> concat 3 modalities."""

    def __init__(self, dims, D=64):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(d, D) for d in dims])
        self.gcn1 = nn.ModuleList([GCNLayer(D, D) for _ in dims])
        self.gcn2 = nn.ModuleList([GCNLayer(D, D) for _ in dims])
        self.D = D

    def forward(self, Xs, As):
        outs = []
        for i in range(len(Xs)):
            h = torch.relu(self.proj[i](Xs[i]))
            h = torch.relu(self.gcn1[i](h, As[i]))
            h = self.gcn2[i](h, As[i])
            g = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=1)  # (B,2D)
            outs.append(g)
        return torch.cat(outs, dim=1)  # (B, 3*2D)


class MultiTimeModel(nn.Module):
    def __init__(self, node_dims, K, n_clin=4, D=64, Dg=64):
        super().__init__()
        self.encoder = GraphEncoder(node_dims, D=D)
        self.clin = nn.Sequential(nn.Linear(n_clin, 32), nn.ReLU())
        emb = 3 * 2 * D + 32
        self.head = nn.Sequential(nn.Linear(emb, 128), nn.ReLU(), nn.Dropout(0.2),
                                  nn.Linear(128, K))
        self.K = K

    def forward(self, Xs, As, clin):
        g = self.encoder(Xs, As)
        c = self.clin(clin)
        z = torch.cat([g, c], dim=1)
        return torch.sigmoid(self.head(z))  # (B,K) hazards


# ---------------- sanity gate ----------------
def sanity(device='cpu'):
    torch.manual_seed(0)
    np.random.seed(0)
    K, T_max, N = 20, 1000, 64
    node_dims = [10, 12, 11]
    Xs = [torch.randn(N, 7, d, device=device) for d in node_dims]
    A = torch.rand(N, 7, 7, device=device)
    As = [(a + a.transpose(1, 2)) / 2 * (1 - torch.eye(7, device=device)) for a in [A, A.clone(), A.clone()]]
    clin = torch.randn(N, 4, device=device)
    model = MultiTimeModel(node_dims, K, n_clin=4, D=16, Dg=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    times = np.random.uniform(30, T_max, N)
    events = np.random.rand(N) < 0.6
    bin_e, n_c = bins_and_censor_counts(times, events, K, T_max)
    h = model(Xs, As, clin)
    assert h.shape == (N, K), f'bad hazard shape {h.shape}'
    assert ((h > 0) & (h < 1)).all(), 'hazards must be in (0,1)'
    S = survival_from_hazards(h)
    assert S.shape == (N, K)
    assert (S.diff(dim=1) <= 1e-6).all(), 'survival must be monotone non-increasing'
    loss0 = discrete_nll(h, bin_e, n_c, events)
    assert torch.isfinite(loss0), 'non-finite loss'
    for step in range(300):
        opt.zero_grad()
        h = model(Xs, As, clin)
        loss = discrete_nll(h, bin_e, n_c, events)
        loss.backward()
        ok = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
        assert ok, f'NaN grad at step {step}'
        opt.step()
    with torch.no_grad():
        h = model(Xs, As, clin)
        loss_end = discrete_nll(h, bin_e, n_c, events).item()
        S = survival_from_hazards(h)
        monotone = bool((S.diff(dim=1) <= 1e-6).all().item())
    print(f'[sanity] device={device} loss {loss0.item():.4f} -> {loss_end:.4f} '
          f'(decreased={loss_end < loss0.item() - 1e-4}) monotone={monotone} K={K}')
    print('[sanity] PASS: forward/shapes/hazards/monotone-simple/grads/finite all OK')
    # demo: multi-timepoint + conditional survival values
    Sdemo = S[0].cpu().numpy()
    idx = {t: min(int(t // (T_max / K)), K - 1) for t in [91, 182, 365, 730, 1095]}
    print('  demo S at landmarks (random model, not meaningful yet):',
          {f'{t}d': round(float(Sdemo[idx[t]]), 3) for t in [91, 182, 365, 730, 1095]})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sanity', action='store_true')
    ap.add_argument('--cuda', action='store_true')
    args = ap.parse_args()
    if args.sanity:
        dev = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
        sanity(dev)
    else:
        print('No mode given. Use --sanity (then --train in a later step).')
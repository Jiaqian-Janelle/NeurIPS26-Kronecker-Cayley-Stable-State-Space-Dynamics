"""Vectorised Kronecker-Cayley core for the PDE grid experiments (Section 5).

Same operator as core.levels.LevelD_KroneckerCayley: a coupling kernel builds a zero-row-sum
Laplacian L over the N patches, and the state is advanced by Cay(L). The pairwise loop is
vectorised over patch pairs; the N x N Cayley solve runs in float32 for training and float64 for
the reported rollouts (pass fp="float64").
"""
import torch
import torch.nn as nn

from .interaction_kernel import InteractionKernel


class FastKCCore(nn.Module):
    """Kronecker-Cayley transition over N patches; float32 solve for training, float64 for rollout."""
    def __init__(self, d, n_obj, action_dim=4, solve_fp="float32"):
        super().__init__()
        self.N = n_obj
        self.m = d // n_obj
        self.kernel = InteractionKernel(n_obj, self.m, action_dim)
        self.solve_fp = solve_fp

    def build_L(self, z, a):
        B, N, m = z.shape[0], self.N, self.m
        k = self.kernel
        ae = k.action_embed(a)
        ab = k.action_bias(a).squeeze(-1)
        gs = (0.1 + 0.1 * torch.tanh(k.action_scale(a))).squeeze(-1)
        zb = z.reshape(B, N, m)
        diff = zb[:, :, None, :] - zb[:, None, :, :]
        dn2 = (diff.real ** 2 + diff.imag ** 2).sum(-1, keepdim=True)
        aeb = ae[:, None, None, :].expand(B, N, N, ae.shape[-1])
        w = k.pair_net(torch.cat([dn2, aeb], -1)).squeeze(-1)
        w = w * gs[:, None, None] + ab[:, None, None]
        W = (w + w.transpose(-1, -2)) / 2
        idx = torch.arange(N, device=z.device)
        W = W.clone()
        W[:, idx, idx] = 0.0
        L = W.clone()
        L[:, idx, idx] = -W.sum(-1)                       # zero-row-sum Laplacian
        return L

    def cayley(self, z, L, fp=None):
        B, N, m = z.shape[0], self.N, self.m
        cdt = torch.cdouble if (fp or self.solve_fp) == "float64" else torch.cfloat
        Lc = L.to(cdt)
        I = torch.eye(N, dtype=cdt, device=z.device).unsqueeze(0).expand(B, -1, -1)
        U = torch.linalg.solve(I + 1j * Lc / 2, I - 1j * Lc / 2)
        zmat = z.reshape(B, N, m).to(cdt)
        return (U @ zmat).reshape(B, N * m).to(z.dtype)

    def forward(self, z, a):
        return self.cayley(z, self.build_L(z, a))

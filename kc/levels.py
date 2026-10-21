"""The four transition levels A-D and the LRU baseline (Section 4, Figure 4, Table 15).

Level A (diagonal): unconstrained diagonal transition, |d_k| free.
Level B (unit circle): diagonal with |d_k| = 1, norm preserved, no coupling structure.
Level C (unit circle, zero mode): Level B with one mode per object frozen at d_k = 1 (the first mode
of each object in the coordinate basis; N frozen modes in total, no random basis).
Level D (Kronecker-Cayley): the full model, coupling kernel + Cayley of a zero-row-sum Laplacian.
LRU: linear recurrent unit with unit-modulus eigenvalues (|d_k| = 1, d_k = exp(i theta),
theta ~ Uniform(0, 2 pi)) and an additive action input.
"""
import math
import torch
import torch.nn as nn

from .interaction_kernel import InteractionKernel
from .cayley import KroneckerCayley


class LevelA_Diagonal(nn.Module):
    """Level A: diagonal transition with unconstrained eigenvalues."""
    def __init__(self, d, action_dim=4):
        super().__init__()
        self.d = d
        self.diag_real = nn.Parameter(torch.randn(d) * 0.01)
        self.diag_imag = nn.Parameter(torch.randn(d) * 0.01)
        self.act_net = nn.Sequential(nn.Linear(action_dim, 64), nn.SiLU(), nn.Linear(64, d * 2))

    def forward(self, z, a):
        act_mod = self.act_net(a) * 0.1
        dk = torch.complex(self.diag_real + act_mod[:, :self.d], self.diag_imag + act_mod[:, self.d:])
        return z * dk


class LevelB_UnitCircle(nn.Module):
    """Level B: diagonal transition constrained to |d_k| = 1."""
    def __init__(self, d, action_dim=4):
        super().__init__()
        self.d = d
        self.theta = nn.Parameter(torch.randn(d) * 0.1)
        self.act_net = nn.Sequential(nn.Linear(action_dim, 64), nn.SiLU(), nn.Linear(64, d))

    def forward(self, z, a):
        theta = self.theta.unsqueeze(0) + self.act_net(a) * 0.1
        return z * torch.exp(1j * theta)


class LevelC_UnitCircleZeroMode(nn.Module):
    """Level C: unit-circle diagonal, first mode of each object frozen at d_k = 1 (coordinate basis, N modes)."""
    def __init__(self, d, n_obj=3, action_dim=4):
        super().__init__()
        self.d, self.n_obj = d, n_obj
        self.m = d // n_obj
        n_free = d - n_obj                                # one frozen mode per object
        self.theta_free = nn.Parameter(torch.randn(n_free) * 0.1)
        self.act_net = nn.Sequential(nn.Linear(action_dim, 64), nn.SiLU(), nn.Linear(64, n_free))

    def forward(self, z, a):
        theta_free = self.theta_free.unsqueeze(0) + self.act_net(a) * 0.1
        B = z.shape[0]
        theta_full = torch.zeros(B, self.d, device=z.device)
        idx = 0
        for i in range(self.n_obj):
            start = i * self.m
            theta_full[:, start + 1:start + self.m] = theta_free[:, idx:idx + self.m - 1]
            idx += self.m - 1
        return z * torch.exp(1j * theta_full)             # slot start stays 0 -> d_k = 1


class LevelD_KroneckerCayley(nn.Module):
    """Level D: coupling kernel + Kronecker-Cayley transition (the full model)."""
    def __init__(self, d, n_obj=3, action_dim=4):
        super().__init__()
        m = d // n_obj
        self.kernel = InteractionKernel(n_obj, m, action_dim)
        self.cayley = KroneckerCayley(n_obj, m)

    def forward(self, z, a):
        L, _ = self.kernel(z, a)
        return self.cayley(z, L)


class LRUCore(nn.Module):
    """Linear recurrent unit: z_next = diag(exp(i theta)) z + B(action).

    Unit-circle eigenvalues, initialised theta ~ Uniform(0, 2 pi) (Orvieto et al. 2023);
    no Laplacian eigenbasis.
    """
    def __init__(self, d, action_dim=4):
        super().__init__()
        self.d = d
        self.theta = nn.Parameter(torch.rand(d) * 2 * math.pi)
        self.B_proj = nn.Sequential(nn.Linear(action_dim, 64), nn.SiLU(), nn.Linear(64, d))

    def forward(self, z, action):
        d_k = torch.exp(1j * self.theta).to(z.dtype)
        b = self.B_proj(action).to(z.dtype)
        return z * d_k.unsqueeze(0) + b

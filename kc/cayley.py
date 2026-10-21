"""Cayley transform of a Laplacian and the Kronecker-Cayley transition (Section 3).

The Cayley transform Cay(L) = (I + iL/2)^-1 (I - iL/2) maps a real symmetric L to a
unitary matrix. For a state z reshaped to (B, N, m), the Kronecker-Cayley step applies
Cay(L) to the N object slots and copies across the m modes: Cay(L (x) I_m) = Cay(L) (x) I_m.
"""
import torch
import torch.nn as nn


def cayley(L):
    """Cay(L) for a batch of real symmetric matrices L, shape (B, N, N) -> (B, N, N) complex."""
    B, N = L.shape[0], L.shape[-1]
    L_sym = (L + L.transpose(-1, -2)) / 2                 # exact symmetry
    Lc = L_sym.to(torch.cdouble)
    I = torch.eye(N, dtype=torch.cdouble, device=L.device).unsqueeze(0).expand(B, -1, -1)
    return torch.linalg.solve(I + 1j * Lc / 2, I - 1j * Lc / 2)


class KroneckerCayley(nn.Module):
    """Applies Cay(L) to the N object slots of z, shared across the m modes.

    z is (B, N*m) complex; L is (B, N, N) real symmetric. Returns z_next (B, N*m).
    The N x N Cayley solve is done in double precision for numerical stability.
    """
    def __init__(self, N, m):
        super().__init__()
        self.N, self.m = N, m

    def forward(self, z, L):
        B, N, m = z.shape[0], self.N, self.m
        U = cayley(L)                                     # (B, N, N) unitary
        z_mat = z.reshape(B, N, m).to(torch.cdouble)
        z_next = torch.bmm(U, z_mat)
        return z_next.reshape(B, N * m).to(z.dtype)

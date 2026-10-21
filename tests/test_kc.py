"""Smoke tests for the KC transition (paper Section 3, Corollary 4, Algorithm 1).

Checks on random inputs that:
  (i)   the KC step is unitary to round-off,
  (ii)  the sum over object subspaces (latent zero mode) is preserved,
  (iii) Cay(L (x) I_m) = Cay(L) (x) I_m,
  (iv)  the conserved family of Corollary 4 (momentum Sigma_i z_i) is preserved by the KC step,
        while a random Hermitian form outside it is not.

Run: pytest tests/  (or python tests/test_kc.py)
"""
import numpy as np
import torch

from kc.transition import cayley, KroneckerCayley

torch.manual_seed(0)
DTYPE = torch.float64


def _random_symmetric(B, N):
    A = torch.randn(B, N, N, dtype=DTYPE)
    return (A + A.transpose(-1, -2)) / 2


def _zero_row_sum(B, N):
    """Random real symmetric L with exactly zero row sums (a Laplacian)."""
    A = torch.randn(B, N, N, dtype=DTYPE)
    W = (A + A.transpose(-1, -2)) / 2
    W[:, range(N), range(N)] = 0.0
    L = W.clone()
    L[:, range(N), range(N)] = -W.sum(-1)
    return L


def test_i_unitary():
    L = _random_symmetric(4, 6)
    U = cayley(L)                                   # (B, N, N) complex
    I = torch.eye(6, dtype=U.dtype).expand_as(U)
    err = (U.conj().transpose(-1, -2) @ U - I).abs().max().item()
    assert err < 1e-10, f"U^H U != I, max err {err:.2e}"


def test_i_norm_preserving_step():
    N, m = 5, 8
    L = _random_symmetric(3, N)
    z = torch.complex(torch.randn(3, N * m, dtype=DTYPE), torch.randn(3, N * m, dtype=DTYPE))
    z_next = KroneckerCayley(N, m)(z, L)
    n0 = z.abs().pow(2).sum(-1).sqrt()
    n1 = z_next.abs().pow(2).sum(-1).sqrt()
    assert torch.allclose(n0, n1, atol=1e-9), f"norm changed: {(n0 - n1).abs().max().item():.2e}"


def test_ii_subspace_sum_preserved():
    N, m = 4, 8
    L = _zero_row_sum(3, N)
    z = torch.complex(torch.randn(3, N * m, dtype=DTYPE), torch.randn(3, N * m, dtype=DTYPE))
    z_next = KroneckerCayley(N, m)(z, L)
    def zmode(x): return sum(x[:, i * m:(i + 1) * m] for i in range(N))
    err = (zmode(z_next) - zmode(z)).abs().max().item()
    assert err < 1e-9, f"zero mode not preserved, max err {err:.2e}"


def test_iii_kron_identity():
    N, m = 5, 4
    L = _random_symmetric(1, N)
    Im = torch.eye(m, dtype=DTYPE)
    Lbig = torch.kron(L[0], Im).unsqueeze(0)        # (1, N*m, N*m) = L (x) I_m
    lhs = cayley(Lbig)[0]                            # Cay(L (x) I_m)
    rhs = torch.kron(cayley(L)[0].contiguous(), Im.to(torch.cdouble))  # Cay(L) (x) I_m
    err = (lhs - rhs).abs().max().item()
    assert err < 1e-10, f"Kron identity violated, max err {err:.2e}"


def test_iv_conserved_family_vs_outside():
    N, m = 3, 40
    L = _zero_row_sum(1, N)
    z = torch.complex(torch.randn(1, N * m, dtype=DTYPE), torch.randn(1, N * m, dtype=DTYPE))
    z_next = KroneckerCayley(N, m)(z, L)
    # conserved: momentum Sigma_i z_i (Corollary 4)
    def mom(x): return sum(x[0, i * m:(i + 1) * m] for i in range(N))
    mom_err = (mom(z_next) - mom(z)).abs().max().item()
    assert mom_err < 1e-9, f"momentum not conserved, max err {mom_err:.2e}"
    # outside the family: a random Hermitian quadratic form <z, G z> is NOT invariant
    G = torch.complex(torch.randn(N * m, N * m, dtype=DTYPE), torch.randn(N * m, N * m, dtype=DTYPE))
    G = (G + G.conj().T) / 2
    q0 = (z.conj() @ G @ z.T).real.item()
    q1 = (z_next.conj() @ G @ z_next.T).real.item()
    assert abs(q1 - q0) > 1e-6 * max(1.0, abs(q0)), "random Hermitian form unexpectedly conserved"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("all KC smoke tests passed")

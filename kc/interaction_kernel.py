"""Interaction kernel that maps latent state and action to a zero-row-sum Laplacian (Section 3).

For each object pair (i, j) the coupling MLP takes [squared latent distance, action embedding]
and returns a scalar weight; the weights form a symmetric matrix whose diagonal is set to the
negative row sum, giving a zero-row-sum Laplacian L. L here is the negative of the manuscript's
Laplacian (Equation 5); the two Cayley transforms are inverse unitaries with the same conserved set.
Coupling MLP layer sizes: 17 -> 64 -> 32 -> 1 (3,265 parameters); the full transition (with the
action embedding, bias and scale) has 3,963 trainable parameters. The action embedding has
dimension 16, so the MLP input width is 1 + 16 = 17.
"""
import torch
import torch.nn as nn


class InteractionKernel(nn.Module):
    """Pairwise coupling network producing a zero-row-sum Laplacian L (B, N, N)."""
    def __init__(self, N, m, action_dim=4):
        super().__init__()
        self.N, self.m = N, m
        self.action_embed = nn.Sequential(nn.Linear(action_dim, 32), nn.SiLU(), nn.Linear(32, 16))
        self.pair_net = nn.Sequential(nn.Linear(17, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))
        self.action_bias = nn.Linear(action_dim, 1)
        self.action_scale = nn.Linear(action_dim, 1)
        for mod in [self.pair_net]:
            for p in mod.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain=0.02)
        for mod in [self.action_embed, self.action_bias, self.action_scale]:
            for p in mod.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, gain=0.5)

    def forward(self, z, action):
        B, N, m = z.shape[0], self.N, self.m
        ae = self.action_embed(action)
        ab = self.action_bias(action).squeeze(-1)
        gs = 0.1 + 0.1 * torch.tanh(self.action_scale(action))
        W = torch.zeros(B, N, N, device=z.device)
        for i in range(N):
            for j in range(i + 1, N):
                diff = z[:, i * m:(i + 1) * m] - z[:, j * m:(j + 1) * m]
                dn2 = torch.sum(diff.real ** 2 + diff.imag ** 2, dim=-1, keepdim=True)
                w_ij = self.pair_net(torch.cat([dn2, ae], -1)).squeeze(-1) * gs.squeeze(-1) + ab
                W[:, i, j] = w_ij
                W[:, j, i] = w_ij
        L = (W + W.transpose(-1, -2)) / 2                 # symmetric
        L[:, range(N), range(N)] = 0
        L[:, range(N), range(N)] = -L.sum(dim=-1)         # zero row sum
        return L, W

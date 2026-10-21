"""Patch encoder/decoder world model for the PDE grid benchmarks (Section 5, Table 2).

A field is split into N patches; a shared per-patch nonlinear encoder maps each patch to a latent
of m complex modes; a transition core advances the latent; a decoder reconstructs the field.
Decoders: tied (shared per-patch linear map, so total mass is a linear readout of the latent zero
mode), tied_cos (tied with each patch readout scaled by 1/cos(lat) at the patch mean, so the zero
mode reads out the area-weighted integral on the sphere), unconstrained (global MLP), and split
(linear on the zero mode plus a mean-subtracted MLP on the residual modes). build_core selects the
transition: kc, free, level_b, level_c, lru.
"""
import torch
import torch.nn as nn

from .kc_fast import FastKCCore
from .levels import LevelA_Diagonal, LevelB_UnitCircle, LevelC_UnitCircleZeroMode, LRUCore


def patchify(x, c):
    """(B, H, W, C) or (B, L, C) -> (B, N, patchdim); channel count inferred from x."""
    B, Cx = x.shape[0], x.shape[-1]
    if c["kind"] == "2d":
        H, W, Pg = c["H"], c["W"], c["Pg"]
        ph, pw = H // Pg, W // Pg
        x = x.reshape(B, Pg, ph, Pg, pw, Cx).permute(0, 1, 3, 2, 4, 5).reshape(B, Pg * Pg, ph * pw * Cx)
    else:
        L, Np = c["L"], c["Np"]
        pl = L // Np
        x = x.reshape(B, Np, pl, Cx).reshape(B, Np, pl * Cx)
    return x


def unpatchify(z, c):
    """(B, N, patchdim) -> (B, H, W, C) or (B, L, C); output channel count from c['C']."""
    B = z.shape[0]
    if c["kind"] == "2d":
        H, W, Pg = c["H"], c["W"], c["Pg"]
        ph, pw = H // Pg, W // Pg
        return z.reshape(B, Pg, Pg, ph, pw, c["C"]).permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, c["C"])
    L, Np = c["L"], c["Np"]
    pl = L // Np
    return z.reshape(B, Np, pl, c["C"]).reshape(B, L, c["C"])


def build_core(arm, N, m):
    """Select the transition core. arm in {kc, free, level_b, level_c, lru}."""
    d = N * m
    if arm == "kc":
        return FastKCCore(d, N, 4, solve_fp="float32")
    if arm == "level_b":
        return LevelB_UnitCircle(d, 4)
    if arm == "level_c":
        return LevelC_UnitCircleZeroMode(d, N, 4)
    if arm == "free":
        return LevelA_Diagonal(d, 4)
    if arm == "lru":
        return LRUCore(d, 4)
    raise ValueError(arm)


class GridModel(nn.Module):
    """Patch world model. decoder in {tied, tied_cos, unconstrained, split}; patchdim_out may differ from input."""
    def __init__(self, arm, N, patchdim, m=8, decoder="tied", hid=256, patchdim_out=None):
        super().__init__()
        po = patchdim_out or patchdim
        self.N, self.m, self.pd, self.pd_out, self.decoder = N, m, patchdim, po, decoder
        self.enc = nn.Sequential(nn.Linear(patchdim, hid), nn.SiLU(), nn.Linear(hid, 2 * m))
        self.core = build_core(arm, N, m)
        if decoder in ("tied", "tied_cos"):
            self.dec = nn.Linear(2 * m, po)
            if decoder == "tied_cos":
                # per-patch readout scale (set to 1/cos(lat) at the patch mean for the sphere); ones by default
                self.register_buffer("patch_scale", torch.ones(1, N, 1))
        elif decoder == "split":
            self.dec_lin = nn.Linear(2 * m, po)
            self.dec_mlp = nn.Sequential(nn.Linear(2 * m, hid), nn.SiLU(), nn.Linear(hid, po))
        else:
            self.dec = nn.Sequential(nn.Linear(N * 2 * m, hid), nn.SiLU(), nn.Linear(hid, N * po))

    def encode(self, xp):
        h = self.enc(xp)
        return torch.complex(h[..., :self.m], h[..., self.m:]).reshape(xp.shape[0], self.N * self.m)

    def decode(self, z):
        B = z.shape[0]
        if self.decoder in ("tied", "tied_cos"):
            zb = z.reshape(B, self.N, self.m)
            out = self.dec(torch.cat([zb.real, zb.imag], -1))
            if self.decoder == "tied_cos":
                out = out * self.patch_scale                # per-patch 1/cos(lat) scaling (sphere)
            return out
        if self.decoder == "split":
            zb = z.reshape(B, self.N, self.m)
            zbar = zb.mean(1, keepdim=True)                # zero mode
            r = zb - zbar                                  # residual, sums to zero over patches
            lin = self.dec_lin(torch.cat([zbar.real, zbar.imag], -1)).expand(B, self.N, self.pd_out)
            mlp = self.dec_mlp(torch.cat([r.real, r.imag], -1))
            mlp = mlp - mlp.mean(1, keepdim=True)          # mean-subtracted -> no mass leak
            return lin + mlp
        zr = torch.cat([z.real, z.imag], -1)
        return self.dec(zr).reshape(B, self.N, self.pd_out)

    def rollout(self, xp0, T, fp=None):
        z = self.encode(xp0)
        a = torch.zeros(xp0.shape[0], 4, device=xp0.device)
        outs, lats = [], [z]
        for _ in range(T):
            z = self.core.cayley(z, self.core.build_L(z, a), fp=fp) if (isinstance(self.core, FastKCCore) and fp) else self.core(z, a)
            lats.append(z)
            outs.append(self.decode(z))
        return torch.stack(outs, 1), lats

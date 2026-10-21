"""Train and evaluate a per-particle world model on an N-body system (Section 5, Table 3, Figure 6).

Systems (public N-body datasets): charged (5 particles, 3d), gravity (10 particles, 3d), spring
(5 particles, 2d). A shared per-particle nonlinear encoder maps each particle state (position,
velocity) to m complex modes; a transition core advances the latent; a decoder reconstructs the
state. Tied decoder: shared per-particle linear map, so total linear momentum is a linear readout
of the latent zero mode. Unconstrained decoder: a global MLP. Protocol 1 reports position MSE at
frame 40 (10 steps from frame 30); Protocol 2 rolls the core 1000 steps and reports energy,
linear-momentum, angular-momentum and latent zero-mode drift in absolute units (Kronecker-Cayley
rollout in double precision).

Usage : python nbody_train.py <system> <arm> <decoder> <seed> --data_root DIR
        system in {charged, gravity, spring}; arm in {kc, free, level_b, level_c, lru}
        decoder in {tied, unconstrained}
Output: results/nbody_<system>_<arm>_<decoder>_seed<seed>.json
Seeds : 42
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kc.kc_fast import FastKCCore
from kc.levels import LevelA_Diagonal, LevelB_UnitCircle, LevelC_UnitCircleZeroMode, LRUCore

device = "cuda" if torch.cuda.is_available() else "cpu"
SYS = {
    "charged": dict(sub="charged", suf="charged5_initvel1", N=5, dim=3, layout="dimN"),
    "spring": dict(sub="spring", suf="springs5", N=5, dim=2, layout="dimN"),
    "gravity": dict(sub="gravity", suf="gravity10_initvel1", N=10, dim=3, layout="Ndim"),
}


def load(system, split, data_root):
    c = SYS[system]
    d = f"{data_root}/{c['sub']}"
    loc = np.load(f"{d}/loc_{split}_{c['suf']}.npy")
    vel = np.load(f"{d}/vel_{split}_{c['suf']}.npy")
    if c["layout"] == "dimN":
        loc = loc.transpose(0, 1, 3, 2)
        vel = vel.transpose(0, 1, 3, 2)
    return loc.astype(np.float32), vel.astype(np.float32)


def build_core(arm, N, m):
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


class Model(nn.Module):
    """Per-particle encoder + transition core + decoder (tied linear or unconstrained MLP)."""
    def __init__(self, arm, N, dim, m=8, decoder="tied", hidden=128):
        super().__init__()
        self.N, self.dim, self.m, self.decoder = N, dim, m, decoder
        s = 2 * dim
        self.enc = nn.Sequential(nn.Linear(s, hidden), nn.SiLU(), nn.Linear(hidden, 2 * m))
        self.core = build_core(arm, N, m)
        if decoder == "tied":
            self.dec = nn.Linear(2 * m, s, bias=True)
        else:
            self.dec = nn.Sequential(nn.Linear(N * 2 * m, hidden), nn.SiLU(), nn.Linear(hidden, N * s))

    def encode(self, x):
        h = self.enc(x)
        return torch.complex(h[..., :self.m], h[..., self.m:]).reshape(x.shape[0], self.N * self.m)

    def decode(self, z):
        B = z.shape[0]
        if self.decoder == "tied":
            zb = z.reshape(B, self.N, self.m)
            return self.dec(torch.cat([zb.real, zb.imag], -1))
        zr = torch.cat([z.real, z.imag], -1)
        return self.dec(zr).reshape(B, self.N, 2 * self.dim)

    def rollout(self, x0, T, fp=None):
        z = self.encode(x0)
        a = torch.zeros(x0.shape[0], 4, device=x0.device)
        outs, lats = [], [z]
        for _ in range(T):
            z = self.core.cayley(z, self.core.build_L(z, a), fp=fp) if (isinstance(self.core, FastKCCore) and fp) else self.core(z, a)
            lats.append(z)
            outs.append(self.decode(z))
        return torch.stack(outs, 1), lats


def phys(loc, vel):
    P = vel.sum(1)
    if loc.shape[-1] == 3:
        L = torch.cross(loc, vel, dim=-1).sum(1)
    else:
        L = (loc[..., 0] * vel[..., 1] - loc[..., 1] * vel[..., 0]).sum(1, keepdim=True)
    KE = 0.5 * (vel ** 2).sum([1, 2])
    return KE, P, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system", choices=list(SYS))
    ap.add_argument("arm")
    ap.add_argument("decoder", choices=["tied", "unconstrained"])
    ap.add_argument("seed", type=int)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    c = SYS[args.system]
    N, dim, m = c["N"], c["dim"], 8
    loc, vel = load(args.system, "train", args.data_root)
    tloc, tvel = load(args.system, "test", args.data_root)
    F0, FT = 30, 40                                        # frames 30 -> 40 = 1000 steps at freq 100

    def state(L, V, f):
        return torch.tensor(np.concatenate([L[:, f], V[:, f]], -1), device=device)

    Xtr0 = state(loc, vel, F0)
    Ytr = torch.tensor(loc[:, F0 + 1:FT + 1], device=device)
    model = Model(args.arm, N, dim, m, args.decoder).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(200):
        perm = torch.randperm(len(Xtr0), device=device)
        for s0 in range(0, len(Xtr0), 100):
            idx = perm[s0:s0 + 100]
            preds, _ = model.rollout(Xtr0[idx], FT - F0)
            loss = F.mse_loss(preds[..., :dim], Ytr[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    with torch.no_grad():
        x0 = state(tloc, tvel, F0)
        gtT = torch.tensor(tloc[:, FT], device=device)
        preds, _ = model.rollout(x0, FT - F0)
        p1_mse = float(F.mse_loss(preds[:, -1, :, :dim], gtT))
        x0 = state(tloc, tvel, F0)[:256]
        preds, lats = model.rollout(x0, 1000, fp="float64")

        def zeromode(z):
            return z.reshape(z.shape[0], N, m).sum(1)
        d0 = model.decode(lats[0])
        E0, P0, L0 = phys(d0[..., :dim], d0[..., dim:])
        Q0 = zeromode(lats[0])
        dh = model.decode(lats[1000])
        E, P, L = phys(dh[..., :dim], dh[..., dim:])
        prot2 = {
            "energy_drift": float((E - E0).abs().mean()),
            "linmom_drift": float((P - P0).norm(dim=-1).mean()),
            "angmom_drift": float((L - L0).norm(dim=-1).mean() if L.shape[-1] > 1 else (L - L0).abs().mean()),
            "zero_mode_drift": float((zeromode(lats[1000]) - Q0).norm(dim=-1).mean()),
        }
    res = {"system": args.system, "arm": args.arm, "decoder": args.decoder, "seed": args.seed,
           "N": N, "dim": dim, "protocol1_pos_mse": p1_mse, "protocol2_step1000": prot2}
    os.makedirs(args.out_dir, exist_ok=True)
    out = f"{args.out_dir}/nbody_{args.system}_{args.arm}_{args.decoder}_seed{args.seed}.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"pos_mse={p1_mse:.5f} linmom_drift={prot2['linmom_drift']:.3e} zero_mode_drift={prot2['zero_mode_drift']:.3e} -> {out}")


if __name__ == "__main__":
    main()

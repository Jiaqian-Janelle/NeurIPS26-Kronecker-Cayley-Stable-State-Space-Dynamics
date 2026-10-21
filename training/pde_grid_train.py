"""Train and evaluate a patch world model on one PDE grid benchmark (Section 5, Table 2).

Benchmarks: dam_break (128x128, one field), advection (1024, one field), allen_cahn (128x128,
one field), sphere (96x192 shallow water, five input channels u/v/vor/div/pres, target = pres).
For each rollout step it records nRMSE against ground truth, the mass drift and the latent
zero-mode drift. Mass is the cell sum on flat grids and the area-weighted cos(latitude) sum on the
sphere (the physically meaningful integral). On the sphere the input is frame 1 (pres is zero at
frame 0) and the model encodes five channels but decodes only pres.

Usage : python pde_grid_train.py <benchmark> <arm> <decoder> <seed> --data_root DIR [--m 8]
        benchmark in {dam_break, advection, allen_cahn, sphere}
        arm in {kc, level_a, level_b, level_c, lru}; decoder in {tied, tied_cos, unconstrained, split}
        the sphere uses tied_cos (per-patch 1/cos(lat) readout); flat benchmarks use tied
Output: results/grid_<benchmark>_<arm>_<decoder>_seed<seed>.json
Seeds : 42, 43, 44 (dam_break also 45, 46)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from kc.grid_model import GridModel, patchify, unpatchify

CFG = {
    "dam_break": dict(kind="2d", H=128, W=128, C=1, Pg=8, ntr=400, nte=100, epochs=100),
    "advection": dict(kind="1d", L=1024, C=1, Np=64, ntr=400, nte=100, epochs=100),
    "allen_cahn": dict(kind="2d", H=128, W=128, C=1, Pg=8, ntr=400, nte=100, epochs=100),
    "sphere": dict(kind="2d", H=96, W=192, C=1, Cin=5, Pg=8, ntr=400, nte=100, epochs=150),
}
PRES = 4          # pres is the last of the five sphere channels [u, v, vor, div, pres]
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_benchmark(benchmark, data_root):
    """Return an array (n, T, spatial..., C) for the benchmark. See README for the data step."""
    c = CFG[benchmark]
    if benchmark == "dam_break":
        import h5py
        h = h5py.File(f"{data_root}/dam_break.h5", "r")
        ks = sorted(h.keys())
        arr = np.stack([h[k]["data"][:, :, :, 0] for k in ks[:c["ntr"] + c["nte"]]]).astype(np.float32)
        h.close()
        return arr[..., None]
    if benchmark == "allen_cahn":
        tr = np.load(f"{data_root}/allen_cahn_train.npy")
        te = np.load(f"{data_root}/allen_cahn_test.npy")
        return np.concatenate([tr[:c["ntr"]], te[:c["nte"]]], 0).astype(np.float32)[..., None]
    if benchmark == "advection":
        import h5py
        h = h5py.File(f"{data_root}/advection.hdf5", "r")
        arr = h["tensor"][:c["ntr"] + c["nte"]].astype(np.float32)
        h.close()
        return arr[..., None]
    if benchmark == "sphere":
        import h5py

        def seeds(sp):
            return sorted(p.split("seed=")[-1] for p in glob.glob(f"{data_root}/{sp}/seed=*"))
        trs, tes = seeds("train")[:20], seeds("test")[:20]
        fs = []
        for s in trs:
            fs += sorted(glob.glob(f"{data_root}/train/seed={s}/run*/output.nc"))[:20]
        for s in tes:
            fs += sorted(glob.glob(f"{data_root}/test/seed={s}/run*/output.nc"))[:5]
        a = []
        for f in fs:
            hh = h5py.File(f, "r")
            a.append(np.stack([hh["u"][:, 0], hh["v"][:, 0], hh["vor"][:, 0], hh["div"][:, 0], hh["pres"][:]], -1).astype(np.float32))
            hh.close()
        return np.stack(a)                                # (500, 88, 96, 192, 5); test is seed-major
    raise ValueError(benchmark)


def lat_weights(data_root):
    """cos(latitude) per grid row for the sphere area-weighted mass."""
    import h5py
    f = sorted(glob.glob(f"{data_root}/test/seed=*/run*/output.nc"))[0]
    h = h5py.File(f, "r")
    lat = h["lat"][:]
    h.close()
    return torch.tensor(np.cos(np.deg2rad(lat)).astype(np.float32), device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark", choices=list(CFG))
    ap.add_argument("arm")
    ap.add_argument("decoder")
    ap.add_argument("seed", type=int)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--save_ckpt", action="store_true")
    ap.add_argument("--save_fields", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    c = CFG[args.benchmark]
    N = c.get("Pg", 0) ** 2 or c.get("Np")
    m = args.m
    sphere = args.benchmark == "sphere"
    F0 = 1 if sphere else 0                               # sphere input is frame 1

    data = load_benchmark(args.benchmark, args.data_root)
    ntr, nte = c["ntr"], c["nte"]
    T = data.shape[1] - 1 - F0
    Htr = min(10, T)
    tr = torch.tensor(data[:ntr, :F0 + 1 + Htr] if sphere else data[:ntr], device=device)
    te = torch.tensor(data[ntr:ntr + nte], device=device)
    if sphere:
        wlat = lat_weights(args.data_root)

    def pf(frame):
        return patchify(frame, c)

    def mass(field):
        if sphere:
            return (field[..., 0] * wlat[None, :, None]).sum(dim=(1, 2))   # area-weighted cos(lat) sum
        return field.sum(dim=list(range(1, field.ndim)))                   # cell sum

    def nrmse(a, b):
        num = ((a - b) ** 2).mean(dim=list(range(1, a.ndim))).sqrt()
        den = (b ** 2).mean(dim=list(range(1, b.ndim))).sqrt().clamp(min=1e-9)
        return float((num / den).mean())

    pdin = pf(tr[:1, F0]).shape[-1]
    pdout = (c["H"] // c["Pg"]) * (c["W"] // c["Pg"]) * c["C"] if c["kind"] == "2d" else (c["L"] // c["Np"]) * c["C"]
    model = GridModel(args.arm, N, pdin, m, args.decoder, patchdim_out=pdout).to(device)
    if sphere and args.decoder == "tied_cos":
        Pg, ph = c["Pg"], c["H"] // c["Pg"]
        cos_row = torch.stack([wlat[b * ph:(b + 1) * ph].mean() for b in range(Pg)])   # mean cos over each patch row-block
        s_patch = torch.tensor([1.0 / cos_row[p // Pg] for p in range(N)], device=device)
        model.patch_scale = s_patch.view(1, N, 1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(c["epochs"]):
        perm = torch.randperm(ntr, device=device)
        for s0 in range(0, ntr, 16 if args.benchmark != "advection" else 32):
            idx = perm[s0:s0 + (16 if args.benchmark != "advection" else 32)]
            x0 = pf(tr[idx, F0])
            tgt = tr[idx, F0 + 1:F0 + 1 + Htr, :, :, PRES:PRES + 1] if sphere else tr[idx, F0 + 1:F0 + 1 + Htr]
            preds, _ = model.rollout(x0, Htr)
            pr = torch.stack([unpatchify(preds[:, t], c) for t in range(Htr)], 1)
            loss = F.mse_loss(pr, tgt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    with torch.no_grad():
        x0 = pf(te[:, F0])
        preds, lats = model.rollout(x0, T, fp="float64" if args.arm == "kc" else None)
        pr = torch.stack([unpatchify(preds[:, t], c) for t in range(T)], 1)
        gt = te[:, F0 + 1:F0 + 1 + T, :, :, PRES:PRES + 1] if sphere else te[:, F0 + 1:F0 + 1 + T]
        m0 = mass(unpatchify(model.decode(lats[0]), c))

        def zmode(z):
            return z.reshape(z.shape[0], N, m).sum(1)
        Q0 = zmode(lats[0])
        per_step = {}
        for h in range(1, T + 1):
            mh = mass(unpatchify(model.decode(lats[h]), c))
            per_step[f"step{h}"] = {
                "mass_drift": float((mh - m0).abs().mean()),
                "zero_mode_drift": float((zmode(lats[h]) - Q0).norm(dim=-1).mean()),
                "nrmse": nrmse(pr[:, h - 1], gt[:, h - 1]),
            }
    res = {"benchmark": args.benchmark, "arm": args.arm, "decoder": args.decoder, "seed": args.seed,
           "N": N, "m": m, "T": T, "per_step": per_step}
    os.makedirs(args.out_dir, exist_ok=True)
    suf = "" if m == 8 else f"_m{m}"
    out = f"{args.out_dir}/grid_{args.benchmark}_{args.arm}_{args.decoder}_seed{args.seed}{suf}.json"
    json.dump(res, open(out, "w"), indent=2)
    if args.save_ckpt:
        ck = f"{args.out_dir}/grid_{args.benchmark}_{args.arm}_{args.decoder}_seed{args.seed}.pt"
        torch.save({"state_dict": model.state_dict(), "arm": args.arm, "decoder": args.decoder, "m": m, "benchmark": args.benchmark}, ck)
        print(f"saved checkpoint -> {ck}")
    if args.save_fields and c["kind"] == "2d":
        # decoded fields of the first test trajectory at five steps, for the cross-section figure
        ch = PRES if sphere else 0
        steps = [0, T // 4, T // 2, 3 * T // 4, T]
        with torch.no_grad():
            gt_f = np.stack([te[0, F0 + s, :, :, ch].cpu().numpy() for s in steps])
            pred_f = np.stack([unpatchify(model.decode(lats[s])[:1], c)[0, :, :, 0].cpu().numpy() for s in steps])
        np.savez(f"{args.out_dir}/fields_{args.benchmark}_{args.arm}_seed{args.seed}.npz",
                 steps=np.array(steps), gt=gt_f, **{args.arm: pred_f})
    hT = per_step[f"step{T}"]
    print(f"nrmse@T={hT['nrmse']:.4f} mass_drift@T={hT['mass_drift']:.3e} zero_mode_drift@T={hT['zero_mode_drift']:.3e} -> {out}")


if __name__ == "__main__":
    main()

"""Generate the conservative Allen-Cahn benchmark (Section 5, Allen-Cahn benchmark data).

128x128 periodic domain, double-well W(u) = 0.25 (u^2 - 1)^2, mass-conserving
Rubinstein-Sternberg form du/dt = eps^2 Lap(u) - f(u) + mean(f(u)) with f = u^3 - u; the
mean(f) term makes d/dt integral(u) = 0. Time stepping is IMEX spectral:
uhat_next = (uhat - dt*ghat) / (1 + dt*eps^2*k^2), g = f(u) - mean(f), ghat[0] = 0.

Usage : python gen_allen_cahn.py --out_dir DIR [--n_runs 1000]
Output: <out_dir>/allen_cahn_train.npy, <out_dir>/allen_cahn_val.npy, <out_dir>/allen_cahn_test.npy, <out_dir>/gen_report.json
Seeds : per-run seed = 42 + run index; the trainer reads the first 400 train and 100 test runs
"""
import argparse
import json
import os

import numpy as np

n, L, eps, dt, sub, T = 128, 1.0, 0.03, 2e-5, 40, 100
k = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
KX, KY = np.meshgrid(k, k, indexing="ij")
K2 = KX ** 2 + KY ** 2


def energy(u):
    """E = integral(eps^2/2 |grad u|^2 + W(u)), computed spectrally (mean per cell)."""
    ux = np.real(np.fft.ifft2(1j * KX * np.fft.fft2(u)))
    uy = np.real(np.fft.ifft2(1j * KY * np.fft.fft2(u)))
    return float(np.mean(0.5 * eps ** 2 * (ux ** 2 + uy ** 2) + 0.25 * (u ** 2 - 1) ** 2))


def gen_one(seed):
    """Return (frames (T+1, n, n), per-frame mass, per-frame energy) for one run."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1, 1, (n, n)) + rng.normal(0, 0.05, (n, n))   # random initial condition
    denom = 1 + dt * eps ** 2 * K2
    frames = np.zeros((T + 1, n, n), np.float32)
    frames[0] = u
    mass = [u.mean()]
    en = [energy(u)]
    for t in range(T):
        for _ in range(sub):
            f = u ** 3 - u
            g = f - f.mean()
            uh = (np.fft.fft2(u) - dt * np.fft.fft2(g)) / denom
            u = np.real(np.fft.ifft2(uh))
        frames[t + 1] = u
        mass.append(u.mean())
        en.append(energy(u))
    return frames, np.array(mass), np.array(en)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_runs", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    md, en_mono, allf = [], [], []
    for s in range(args.n_runs):
        fr, mass, en = gen_one(42 + s)
        allf.append(fr)
        if s < 20:
            md.append(float(np.abs(mass - mass[0]).max()))
            en_mono.append(bool(np.all(np.diff(en) <= 1e-9)))
        if (s + 1) % 200 == 0:
            print(f"  gen {s + 1}/{args.n_runs}", flush=True)
    allf = np.array(allf, np.float32)
    np.save(os.path.join(args.out_dir, "allen_cahn_train.npy"), allf[:800])
    np.save(os.path.join(args.out_dir, "allen_cahn_val.npy"), allf[800:900])
    np.save(os.path.join(args.out_dir, "allen_cahn_test.npy"), allf[900:args.n_runs])
    rep = {
        "n": n, "eps": eps, "dt": dt, "substeps_per_frame": sub, "frames": T + 1,
        "n_runs": args.n_runs, "boundary": "periodic",
        "shape_train": list(allf[:800].shape),
        "gt_mass_abs_drift_max_over20": float(np.max(md)),
        "gt_mass_abs_drift_mean_over20": float(np.mean(md)),
        "energy_monotone_decrease_all20": bool(all(en_mono)),
    }
    json.dump(rep, open(os.path.join(args.out_dir, "gen_report.json"), "w"), indent=2)
    print(f"[allen_cahn done] mass_abs_drift max={rep['gt_mass_abs_drift_max_over20']:.3e} "
          f"energy_monotone_all20={rep['energy_monotone_decrease_all20']} "
          f"shape_train={rep['shape_train']}", flush=True)


if __name__ == "__main__":
    main()

"""Download the exact shallow-water sphere subset used (Section 5).

Fetches the public Hugging Face dataset pdearena/ShallowWater-2D, restricted to the 500 trajectory
files listed in data/sphere_trajectory_ids.txt (400 train + 100 test). The list is the pinned
identifier set that produced the reported numbers; each line is a path of the form
"<split>/seed=<id>/run<nnnn>/output.nc". The subset is 400 train trajectories drawn from 20 train
initial-condition seeds (20 runs each) and 100 test trajectories from 20 test seeds (5 runs each).

Usage : python download_shallow_water_sphere.py <out_dir> [--repo pdearena/ShallowWater-2D]
Output: <out_dir>/train/... and <out_dir>/test/... (500 output.nc files)
Seeds : dataset initial-condition seeds are fixed by the identifier list (not an RNG seed here)
"""
import argparse
import glob
import os

from huggingface_hub import snapshot_download

IDS_FILE = os.path.join(os.path.dirname(__file__), "sphere_trajectory_ids.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--repo", default="pdearena/ShallowWater-2D")
    args = ap.parse_args()
    with open(IDS_FILE) as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    snapshot_download(args.repo, repo_type="dataset", local_dir=args.out_dir, allow_patterns=ids, max_workers=16)
    ntr = len(glob.glob(f"{args.out_dir}/train/seed=*/run*/output.nc"))
    nte = len(glob.glob(f"{args.out_dir}/test/seed=*/run*/output.nc"))
    print(f"downloaded {len(ids)} pinned trajectories -> train files {ntr}, test files {nte}")


if __name__ == "__main__":
    main()

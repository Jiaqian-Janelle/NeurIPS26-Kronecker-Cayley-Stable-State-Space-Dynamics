#!/usr/bin/env bash
# Download the public benchmarks used by the paper and print their licences.
# Usage: bash data/download_public.sh DATA_DIR
set -euo pipefail
DATA="${1:?usage: download_public.sh DATA_DIR}"; mkdir -p "$DATA"
echo "== PDEBench (dam break 2D_rdb_NA_NA.h5, advection 1D_Advection_Sols_beta1.0.hdf5) =="
echo "   code MIT, data CC BY 4.0 — https://github.com/pdebench/PDEBench"
echo "   place 2D_rdb_NA_NA.h5 -> \$DATA/dam_break.h5 ; 1D_Advection_Sols_beta1.0.hdf5 -> \$DATA/advection.hdf5"
echo "== PDEArena ShallowWater-2D (sphere) =="
echo "   code & data MIT — https://huggingface.co/datasets/pdearena/ShallowWater-2D"
echo "   run: python data/download_shallow_water_sphere.py \$DATA  (pins the 500-trajectory subset)"
echo "== EGNN N-body (charged, gravity, spring) =="
echo "   code MIT — https://github.com/vgsatorras/egnn  (generate loc_/vel_ npy per data/README.md)"
echo "All public data is downloaded under its own licence; see data/README.md and THIRD_PARTY.md."

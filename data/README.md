# Datasets

Each trainer takes `--data_root DATA`; place the files below under `DATA` with the exact names the
loaders expect. Two datasets are generated locally; the rest are public downloads.

## dam_break (PDEBench 2D shallow water)
- Source: PDEBench (https://github.com/pdebench/PDEBench), 2D radial dam-break file `2D_rdb_NA_NA.h5`.
- Put it at: `DATA/dam_break.h5`.
- Loader (`training/pde_grid_train.py`): reads the water-height channel from the first 500 keys after
  sorting `h.keys()` (first 400 train, next 100 test).

## advection (PDEBench 1D advection)
- Source: PDEBench, 1D advection file `1D_Advection_Sols_beta1.0.hdf5`.
- Put it at: `DATA/advection.hdf5`.
- Loader: dataset `tensor`, first 500 trajectories (400 train, 100 test).

## allen_cahn (generated here)
- Generate: `python data/gen_allen_cahn.py --out_dir DATA`.
- Writes: `DATA/allen_cahn_train.npy` (800), `DATA/allen_cahn_val.npy` (100), `DATA/allen_cahn_test.npy`.
- Loader uses the first 400 train and 100 test runs.

## sphere (PDEArena shallow water, sphere)
- Source: public Hugging Face dataset `pdearena/ShallowWater-2D`.
- Download the pinned subset: `python data/download_shallow_water_sphere.py DATA`.
- Layout under `DATA`: `train/seed=<id>/run<nnnn>/output.nc` and `test/seed=<id>/run<nnnn>/output.nc`.
- The exact 500 trajectories are listed in `data/sphere_trajectory_ids.txt` (400 train from 20
  train-split seeds x 20 runs; 100 test from 20 test-split seeds x 5 runs).

## N-body (charged, gravity, spring)
- Source: public N-body simulation datasets (charged particles and springs as in the interaction-network
  and equivariant-GNN literature; gravity as in the equivariant-GNN N-body suite).
- Put per-system files under `DATA`:
  - `DATA/charged/loc_train_charged5_initvel1.npy`, `DATA/charged/vel_train_charged5_initvel1.npy`,
    and the `test` counterparts.
  - `DATA/gravity/loc_train_gravity10_initvel1.npy`, `DATA/gravity/vel_train_gravity10_initvel1.npy`,
    and the `test` counterparts.
  - `DATA/spring/loc_train_springs5.npy`, `DATA/spring/vel_train_springs5.npy`, and the `test` counterparts.
- Loader (`training/nbody_train.py`): position/velocity arrays; frames 30 to 40 for Protocol 1, then a
  1000-step rollout for Protocol 2.


## Licences (summary)
PDEBench: code MIT, data CC BY 4.0. PDEArena / ShallowWater-2D: code & data MIT. EGNN N-body: MIT.
neuraloperator (FNO): MIT. Our code and all generated datasets: MIT (this repository).


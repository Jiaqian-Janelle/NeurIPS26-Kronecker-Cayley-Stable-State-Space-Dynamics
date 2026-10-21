# Third-party data

This repository ships only our own code (MIT, see `LICENSE`). No third-party source tree is
included. The datasets below are obtained separately with the scripts in `data/`, each under its
own licence.

| Dataset | Benchmark | Source | Licence |
|---|---|---|---|
| PDEBench | dam break, advection | https://github.com/pdebench/PDEBench | code MIT, data CC BY 4.0 |
| PDEArena ShallowWater-2D | sphere | https://huggingface.co/datasets/pdearena/ShallowWater-2D | MIT |
| EGNN N-body | charged, gravity, spring | https://github.com/vgsatorras/egnn | MIT |

Our generated dataset (Allen–Cahn) is
produced by the generators in `data/` and are released under this repository's MIT licence.

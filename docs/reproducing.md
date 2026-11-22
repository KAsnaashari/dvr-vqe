# Reproducing the calculations

## Requirements

`requirements.txt` pins the versions that were tested (Python 3.11). The code needs NumPy < 2 and SciPy < 1.14: `lib/dvr2d.py` uses `numpy.math` and `lib/pot_gen.py` imports `scipy.interpolate.interp2d`, both of which were removed in later versions.

On Windows, set `PYTHONUTF8=1` if the output is redirected to a file. The circuit diagrams in the log contain characters that the default console encoding cannot write.

## Running a config

The parameters of each original run script are in `configs/<script name>.yaml`, and the runner performs the same calculation:

```bash
python scripts/run_dvr_vqe.py configs/arhcl_vqe_excited2.yaml
python scripts/run_dvr_vqe.py configs/arhcl_vqe_excited2.yaml --dry-run   # only print the options
```

A config sets the molecule, the potential, the DVR grid (`dvr_options`), the ansatz (`ansatz_options`, or `ansatz_options_list` for several in sequence), the optimisers and number of restarts (`vqe_options`), the noise model, whether to compute excited states, and `cont`, which resumes a previous run from step `cont`. The docstring of `scripts/run_dvr_vqe.py` lists every key.

| Group | Configs | Calculation |
|---|---|---|
| `excited_twolocal` | `*_vqe_excited.yaml`, `*_vqe_excited[1-5].yaml` (18) | Ground and excited states with a TwoLocal ansatz |
| `excited_opt` | `*_vqe_excited_opt*.yaml` (20) | Ground and excited states with a fixed circuit given as a list of CNOTs (`gates`) |
| `greedyent` | `*_vqe_greedyent[1-5].yaml` (10) | Greedy-entangling search |
| `greedyent_const` | `*_vqe_greedyent_const[1-5].yaml`, `*_vqe_greedyent_const.yaml` (18) | Constructive greedy-entangling search (adds CNOTs one at a time) |
| `greedyent_const_noisy` | `*_greedyent_const_noisy.yaml` (2) | The same with the FakeSherbrooke noise model |
| `noent`, `noent_noisy` | `arhcl_vqe_noent*.yaml` (2) | One layer of RY rotations without entangling gates, noiseless and noisy |
| `greedy` | `arhcl_vqe_greedy.yaml` (1) | Greedy layer-by-layer search |
| `block` | `*_vqe_block*.yaml` (4) | Block-structured circuits |

`_64` in a name means the 64-point (6-qubit) grid. Cr₂ configs are named by total spin (`cr2_13_*` is S = 13).

The VQE energy changes slightly from run to run, because the random initial parameters are not seeded (see `lib.vqe.opt_vqe`). As a check against the original calculations, `configs/arhcl_vqe_noent.yaml` (ArHCl, 32-point 2D DVR, 5 qubits, about 2 minutes) gives −94.70158760633788 cm⁻¹, against −94.70158760633757 cm⁻¹ from the original cluster run.

## Data and output directories

Paths default to directories inside the repository. Override them with environment variables or the runner's flags:

| Variable | Flag | Default | Used for |
|---|---|---|---|
| `DVR_VQE_DATA_DIR` | `--data-dir` | repository root | `arhcl.txt`, `mgnh.txt` |
| `DVR_VQE_SCRATCH_DIR` | `--scratch-dir` | `scratch/` | configs with `output_root: scratch` |
| `DVR_VQE_RESULTS_DIR` | `--results-dir` | `results/` | configs with `output_root: results` |

The potential files have three columns: R (Å), cos θ (ArHCl) or θ in radians (MgNH), and V (cm⁻¹).

## Running on a Slurm cluster

The job scripts in `scripts/` were written for ETH Euler. Submit them from the repository root:

```bash
mkdir -p outputs
sbatch scripts/arhcl_vqe_excited_opt1.sh
```

Each job loads the Python module, activates the virtual environment in `$DVR_VQE_VENV`, sets the three directories (scratch defaults to `$SCRATCH/dvr_vqe`) and runs the runner with the matching config. Slurm logs go to `outputs/`. To run another config, copy a job script and change the config path. Adjust the `module load` line and the resources for other clusters.

## How the configs were checked

`tools/verify_configs.py` runs every original script, as of commit `58aec81`, with `DVR_VQE`, `get_pot` and `numpy.savez` replaced by recorders, so that no calculation is performed. It then does the same for the runner with the matching config, and compares the two: molecule parameters, potential, DVR / ansatz / VQE options, log directory, noise model, run flags and saved files, including Python types. All 75 configs match:

```bash
python tools/verify_configs.py
```

The original scripts are kept in `legacy/` (see `legacy/README.md`).

## Other scripts

`circuit_opt.py`, `cr2_vqe_excited.py`, `cr2_vqe_optim.py`, `single_param_vqe.py` and `single_param_vqe_layered.py` have their own logic (custom circuit searches, DVR-grid optimisation, grid inspection) that the runner does not cover.

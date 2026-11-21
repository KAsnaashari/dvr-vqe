# dvr-vqe

This repository computes molecular vibrational energy levels with the variational quantum eigensolver (VQE), using a discrete variable representation (DVR) of the vibrational Hamiltonian. The DVR Hamiltonian is built on a grid of 2ⁿ points, written as a sum of Pauli strings on n qubits, and solved with VQE in Qiskit, noiselessly or with the noise model of a real IBM device (FakeSherbrooke). The code covers a 1D DVR for diatomic molecules (Cr₂) and a 2D Jacobi-coordinate DVR for atom–diatom complexes (ArHCl, MgNH). It provides fixed (TwoLocal) ansätze and adaptive circuit construction: greedy layer-by-layer search, greedy selection of entangling gates, and block-structured circuits. Excited states are computed with variational quantum deflation.

## Citation

If you use this code, please cite:

> K. Asnaashari, D. Bondarenko and R. V. Krems, "Advantages of discrete variable representation in variational quantum eigensolvers for vibrational energy calculations", *Phys. Chem. Chem. Phys.* **28**, 7900–7912 (2026). [doi:10.1039/D5CP02860D](https://doi.org/10.1039/D5CP02860D)

```bibtex
@article{Asnaashari2026,
  author  = {Asnaashari, K. and Bondarenko, D. and Krems, R. V.},
  title   = {Advantages of discrete variable representation in variational quantum eigensolvers for vibrational energy calculations},
  journal = {Phys. Chem. Chem. Phys.},
  volume  = {28},
  pages   = {7900--7912},
  year    = {2026},
  doi     = {10.1039/D5CP02860D}
}
```

## Installation

Requires **Python 3.11** (tested with 3.11.15).

```bash
git clone https://github.com/KAsnaashari/dvr-vqe.git
cd dvr-vqe
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions that were tested. The code needs NumPy < 2 and SciPy < 1.14: `lib/dvr2d.py` uses `numpy.math` and `lib/pot_gen.py` imports `scipy.interpolate.interp2d`, both of which were removed in later versions.

On Windows, set `PYTHONUTF8=1` if the output is redirected to a file. The circuit diagrams in the log contain characters that the default console encoding cannot write.

## Quick start

The smallest end-to-end calculation is the vibrational ground state of Cr₂ (S = 13) on a 16-point 1D DVR (4 qubits), with a two-layer TwoLocal ansatz on the noiseless simulator. It takes under a minute on a laptop:

```bash
python scripts/run_dvr_vqe.py configs/quickstart.yaml --exact
```

The last lines of the output are:

```
L_BFGS_B, Energy: -548.6563545644602, Counts: 1001
Exact DVR eigenvalues (cm^-1), lowest 6: -548.675543, -497.153252, -449.260890, -404.674109, -362.990064, -325.722056
```

The VQE energy is within about 0.02 cm⁻¹ of the exact ground state of the same DVR Hamiltonian. `--exact` prints that reference after the run. Logs and `.npz` results are written to `scratch/quickstart/`. The VQE energy changes slightly from run to run, because the random initial parameters are not seeded (see `lib.vqe.opt_vqe`).

As a check against the original calculations, `configs/arhcl_vqe_noent.yaml` (ArHCl, 32-point 2D DVR, 5 qubits, one layer of RY rotations, about 2 minutes) gives −94.70158760633788 cm⁻¹, against −94.70158760633757 cm⁻¹ in the output of the original cluster run. Its `--exact` reference, −115.177966 cm⁻¹, matches the 32-point ground-state energy (−115.17796582 cm⁻¹) used in `vqe_opt.ipynb`.

## Repository layout

| Path | Contents |
|---|---|
| `lib/` | Core library. `dvr1d.py`: 1D DVR (Cr₂); `dvr2d.py`: 2D Jacobi DVR (ArHCl, MgNH); `pot_gen.py`: potential-energy surfaces (`cr2_pot.py` duplicates its Cr₂ part); `vqe.py`: VQE loops, excited states and the `DVR_VQE` driver class; `greedy_circs.py`, `block_circs.py`: adaptive circuit builders; `utils.py`: constants and Pauli decomposition; `paths.py`: data, scratch and results directories. |
| `scripts/run_dvr_vqe.py` | Runner: performs the calculation described by a config file. |
| `scripts/*.sh` | Slurm job scripts (ETH Euler cluster) that call the runner. |
| `configs/` | One YAML config per original run script (`configs/<script>.yaml`), plus `quickstart.yaml`. |
| `tools/verify_configs.py` | Checks that each config reproduces the options of the original script (see below). |
| `arhcl.txt`, `mgnh.txt` | Potential-energy surfaces of ArHCl and MgNH: three columns, R (Å), cos θ (ArHCl) or θ in radians (MgNH), and V (cm⁻¹). |
| `*.ipynb` | Analysis and exploration notebooks (see below). |
| `legacy/` | The original run scripts that the configs replace (see `legacy/README.md`). |
| `circuit_opt.py`, `cr2_vqe_excited.py`, `cr2_vqe_optim.py`, `single_param_vqe.py`, `single_param_vqe_layered.py` | Scripts with their own logic (custom circuit searches, DVR-grid optimisation, grid inspection) that the runner does not cover. |
| `mgnh_vqe_excited.py`, `mgnh_vqe_greedy.py` | Early scripts written for an older version of `lib/`; they do not run with the current code. |

## Reproducing the calculations

The parameters of each original run script are now in `configs/<script name>.yaml`, and the runner performs the same calculation:

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

`_64` in a name means the 64-point (6-qubit) grid; Cr₂ configs are named by total spin (`cr2_13_*` is S = 13).

### Data and output directories

Paths default to directories inside the repository. Override them with environment variables or the runner's flags:

| Variable | Flag | Default | Used for |
|---|---|---|---|
| `DVR_VQE_DATA_DIR` | `--data-dir` | repository root | `arhcl.txt`, `mgnh.txt` |
| `DVR_VQE_SCRATCH_DIR` | `--scratch-dir` | `scratch/` | configs with `output_root: scratch` |
| `DVR_VQE_RESULTS_DIR` | `--results-dir` | `results/` | configs with `output_root: results` |

### Running on a Slurm cluster

The job scripts in `scripts/` were written for ETH Euler. Submit them from the repository root:

```bash
mkdir -p outputs
sbatch scripts/arhcl_vqe_excited_opt1.sh
```

Each job loads the Python module, activates the virtual environment in `$DVR_VQE_VENV` (default `$HOME/envs/spinenv3_ubuntu`), sets the three directories (scratch defaults to `$SCRATCH/dvr_vqe`) and runs the runner with the matching config. Slurm logs go to `outputs/`. To run another config, copy a job script and change the config path. Adjust the `module load` line and the resources for other clusters.

### How the configs were checked

`tools/verify_configs.py` runs every original script, as of commit `58aec81`, with `DVR_VQE`, `get_pot` and `numpy.savez` replaced by recorders, so that no calculation is performed. It then does the same for the runner with the matching config, and compares the two: molecule parameters, potential, DVR / ansatz / VQE options, log directory, noise model, run flags and saved files, including Python types. All 75 configs match:

```bash
python tools/verify_configs.py
```

## Notebooks

The notebooks are the analysis and exploration done during the project. Notebooks whose saved outputs contained error tracebacks or local file paths have had their outputs cleared. Some rely on results files that are not in the repository, or on functions that have since been renamed, so not all of them run as they are.

| Notebook | Contents |
|---|---|
| `analysis.ipynb` | Plots saved VQE results (excited states, greedy and greedy-entangling circuits, qubit partitions, noisy vs noiseless ArHCl) against reference energies |
| `ansatz.ipynb` | Draws TwoLocal and greedy-entangling ansatz circuits |
| `arhcl.ipynb` | Plots the ArHCl potential-energy surface from `arhcl.txt` |
| `block_circuits.ipynb` | Exploratory: tests of the qubit-partition helpers in `lib/greedy_circs.py` |
| `circuit_opt.ipynb` | Exploratory: early ArHCl DVR and circuit tests |
| `cr_pots.ipynb` | Plots the Cr₂ potential curves (S = 1–13) and tests the 1D DVR and VQE on them |
| `dvr_arhcl.ipynb` | 2D DVR of ArHCl: grid, Hamiltonian and eigenvalues |
| `dvr_arhcl_vqe.ipynb` | VQE with a TwoLocal ansatz on the ArHCl DVR Hamiltonian |
| `dvr_arhcl_vqe_excited.ipynb` | Excited-state VQE for ArHCl |
| `dvr_cr2_vqe.ipynb` | Cr₂ 1D DVR Hamiltonian and its Pauli decomposition |
| `dvr_mgnh.ipynb` | 2D DVR of MgNH: grid, Hamiltonian and eigenvalues |
| `dvr_mgnh_vqe.ipynb` | VQE with a TwoLocal ansatz on the MgNH DVR Hamiltonian |
| `dvr_optim.ipynb` | Choice of the Cr₂ 1D DVR grid (box limits and spacing) for a given number of qubits |
| `entangle_opt.ipynb` | Exploratory: greedy-entangling ansatz tests for ArHCl |
| `mgnh.ipynb` | Plots the MgNH potential-energy surface from `mgnh.txt` |
| `noise_test.ipynb` | Cr₂ ground- and excited-state VQE with the FakeSherbrooke noise model, compared with exact eigenvalues |
| `pauli_scaling.ipynb` | Number of Pauli strings in the DVR Hamiltonians and the effect of truncating off-diagonal elements |
| `refs.ipynb` | Exact reference energies (DVR eigenvalues) for ArHCl, MgNH and Cr₂ |
| `refs_constrained.ipynb` | Exact reference energies for the constrained (truncated) grids |
| `single_param.ipynb` | Single-parameter circuit search for ArHCl (with `single_param_vqe.py`) |
| `visualization.ipynb` | Cr₂ excited-state VQE results, noiseless and noisy, compared with exact eigenvalues |
| `visualization_arhcl.ipynb` | ArHCl excited-state energies and state fidelities from the saved results |
| `visualization_mgnh.ipynb` | MgNH excited-state energies and state fidelities from the saved results |
| `vqe_opt.ipynb` | VQE convergence for ArHCl compared with the exact 32-point and converged ground-state energies |

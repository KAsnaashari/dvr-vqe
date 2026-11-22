# Notebooks

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

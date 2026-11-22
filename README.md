# dvr-vqe

Variational quantum eigensolver (VQE) calculations of molecular vibrational energy levels using a discrete variable representation (DVR) of the Hamiltonian, in Qiskit. The code covers 1D (Cr₂) and 2D (ArHCl, MgNH) DVRs, adaptive circuit construction, excited states, and simulations with a real-device noise model.

Code for K. Asnaashari, D. Bondarenko and R. V. Krems, "Advantages of discrete variable representation in variational quantum eigensolvers for vibrational energy calculations", *Phys. Chem. Chem. Phys.* **28**, 7900–7912 (2026). [doi:10.1039/D5CP02860D](https://doi.org/10.1039/D5CP02860D)

<details>
<summary>BibTeX</summary>

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
</details>

## Installation

Requires Python 3.11.

```bash
git clone https://github.com/KAsnaashari/dvr-vqe.git
cd dvr-vqe
pip install -r requirements.txt
```

## Quick start

Ground state of Cr₂ on a 16-point DVR (4 qubits), noiseless simulator, under a minute on a laptop:

```bash
python scripts/run_dvr_vqe.py configs/quickstart.yaml --exact
```

The VQE energy comes out within about 0.02 cm⁻¹ of the exact DVR ground state (−548.676 cm⁻¹), which `--exact` prints for comparison.

## Repository layout

- `lib/`: DVR, Pauli decomposition, ansatz construction and VQE driver
- `scripts/run_dvr_vqe.py`: runs the calculation described by a config file
- `configs/`: one config per calculation in the paper
- `arhcl.txt`, `mgnh.txt`: potential-energy surfaces
- `*.ipynb`: analysis and plotting notebooks

## More

- [Reproducing the calculations](docs/reproducing.md): configs, data and output directories, Slurm jobs, and how the configs were checked against the original scripts
- [Notebooks](docs/notebooks.md): what each notebook contains

## License

MIT, see [LICENSE](LICENSE).

# Legacy run scripts

These are the original run scripts. Each one is reproduced exactly by `configs/<script name>.yaml` and `scripts/run_dvr_vqe.py`; `tools/verify_configs.py` checks this. Use the runner for new runs.

To run a legacy script, run it as a module from the repository root so that `lib` can be imported:

```bash
python -m legacy.arhcl_vqe_excited2
```

Data and output directories are set as described in the main README (`DVR_VQE_DATA_DIR`, `DVR_VQE_SCRATCH_DIR`, `DVR_VQE_RESULTS_DIR`). Unlike the runner, these scripts do not create their output directory, so it must exist before the run.

`mgnh_vqe_excited.py` and `mgnh_vqe_greedy.py` were written for an older version of `lib/` and do not run with the current code; they have no config.

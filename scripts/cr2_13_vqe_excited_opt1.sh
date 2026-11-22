#!/bin/bash

# Submit from the repository root:  mkdir -p outputs && sbatch scripts/cr2_13_vqe_excited_opt1.sh
# Optional: set DVR_VQE_VENV, DVR_VQE_DATA_DIR, DVR_VQE_SCRATCH_DIR or
# DVR_VQE_RESULTS_DIR before submitting to override the defaults below.

#SBATCH -n 1
#SBATCH --cpus-per-task=128
#SBATCH --time=4-00:00:00
#SBATCH --job-name="cr2_13_1"
#SBATCH --mem-per-cpu=1024
#SBATCH --output="outputs/cr2_13_vqe_excited_opt1.out"
#SBATCH --error="outputs/cr2_13_vqe_excited_opt1.err"
#SBATCH --open-mode=truncate
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module load stack/2024-06 gcc/12.2.0 python/3.11.6
source "${DVR_VQE_VENV:-$HOME/envs/dvr-vqe}/bin/activate"

cd "$SLURM_SUBMIT_DIR"
export DVR_VQE_DATA_DIR="${DVR_VQE_DATA_DIR:-$SLURM_SUBMIT_DIR}"
export DVR_VQE_SCRATCH_DIR="${DVR_VQE_SCRATCH_DIR:-$SCRATCH/dvr_vqe}"
export DVR_VQE_RESULTS_DIR="${DVR_VQE_RESULTS_DIR:-$SLURM_SUBMIT_DIR/results}"

python scripts/run_dvr_vqe.py configs/cr2_13_vqe_excited_opt1.yaml

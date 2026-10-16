#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=128
#SBATCH --time=4-00:00:00
#SBATCH --job-name="cr2_13"
#SBATCH --mem-per-cpu=1024
#SBATCH --output="/cluster/home/kasnaashari/projects/bin/dvr_vqe/outputs/cr2_13_vqe_greedyent_const_noisy.out"
#SBATCH --error="/cluster/home/kasnaashari/projects/bin/dvr_vqe/outputs/cr2_13_vqe_greedyent_const_noisy.err"
#SBATCH --open-mode=truncate
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=k.asnaashari@gmail.com

# module load gcc/9.3.0 python/3.11.2 openmpi/4.1.4
module load stack/2024-06 gcc/12.2.0 python/3.11.6
# source /cluster/home/kasnaashari/envs/spinenv3/bin/activate
source /cluster/home/kasnaashari/envs/spinenv3_ubuntu/bin/activate
export PYTHONPATH="$HOME/projects/share:$HOME/projects/lib/python:$HOME/projects/lib/my_libs:$HOME/projects/lib/pyrho:$PYTHONPATH:"
 
python /cluster/home/kasnaashari/projects/bin/dvr_vqe/cr2_13_vqe_greedyent_const_noisy.py

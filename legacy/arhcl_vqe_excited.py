import numpy as np
from lib.dvr2d import *
from lib.utils import *
from lib.vqe import DVR_VQE
from lib.pot_gen import get_pot

from lib.local_utils import base_dir, scratch_dir

mol_params = arhcl_params
pot_file = base_dir + 'arhcl.txt'

dvr_options = {
    'type': 'jacobi',
    'J': 0,
    'N_R': 35, 
    'N_theta': 4,
    'l': 1,
    'K_max': 0,
    'r_min': 3.4,
    'r_max': 5, 
    'trunc': 0
}

vqe_options = {
    # 'optimizers': ['COBYLA.8000', 'L_BFGS_B.8000', 'SLSQP.1000'],
    'optimizers': ['COBYLA.100'],
    'repeats': 1, 
    # 'repeats': 5, 
    'beta': [100, 100],
    'seed': 42
}

reps_list = [4]
# reps_list = [3]
ansatz_options_list = []

for reps in reps_list:
    ansatz_options_list.append({
        'type': 'twolocal',
        'rotation_blocks': ['ry'],
        'entanglement_blocks': 'cx',
        'entanglement': 'linear',
        'reps': reps
    })

pot = get_pot(pot_file, take_cos=True)
dvr_vqe = DVR_VQE(mol_params, pot, log_dir=scratch_dir + 'test/')
for ansatz_options in ansatz_options_list:
    dvr_vqe.get_dvr_vqe(dvr_options, ansatz_options, vqe_options, excited_states=True)
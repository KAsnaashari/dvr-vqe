from math import cos
import numpy as np
from dvr2d import *
from utils import *
from vqe import DVR_VQE

# from local_utils import base_dir, scratch_dir

mol_params = mgnh_params
pot_file = base_dir + mol_params['name'] + '.txt'

dvr_options = {
    'type': 'jacobi',
    'J': 0,
    'N_R': 20, 
    'N_theta': 4,
    'l': 1,
    'K_max': 0,
    'r_min': 3,
    'r_max': 6, 
    'trunc': 0
}

vqe_options = {
    'optimizers': ['COBYLA.8000', 'L_BFGS_B.8000', 'SLSQP.1000'],
    # 'optimizers': ['COBYLA.500'],
    # 'repeats': 2, 
    'repeats': 5, 
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

dvr_vqe = DVR_VQE(mol_params, pot_file, take_cos=False, log_dir=scratch_dir + 'excited_states/')
# print(dvr_vqe.get_h_dvr(dvr_options))
for ansatz_options in ansatz_options_list:
    dvr_vqe.get_dvr_vqe(dvr_options, ansatz_options, vqe_options, excited_states=True)
from math import cos
import numpy as np
from lib.dvr2d import *
from lib.utils import *
from lib.vqe import DVR_VQE
from lib.pot_gen import get_pot

# from local_utils import base_dir, scratch_dir

mol_params = mgnh_params
pot_file = base_dir + 'mgnh.txt'

# 32 points
# dvr_options = {
#     'type': 'jacobi',
#     'N_R': 20, 
#     'N_theta': 4,
#     'l': 1,
#     'K_max': 0,
#     'r_min': 3,
#     'r_max': 6, 
#     'trunc': 0,
#     'J': 0
# }

# ansatz_options = {
#     'type': 'greedyent',
#     'constructive': True,
#     'layers': 10,
#     'num_keep': 1, 
#     'num_qubits': 6,
#     'reps': 1
# }

# 64 points
dvr_options = {
    'type': 'jacobi',
    'J': 0,
    'N_R': 40, 
    'N_theta': 4,
    'l': 1,
    'K_max': 0,
    'r_min': 3,
    'r_max': 9, 
    'trunc': 0
}

ansatz_options = {
    'type': 'twolocal',
    'rotation_blocks': ['ry'],
    'entanglement_blocks': 'cx',
    'entanglement': 'linear',
    'reps': 2
}

vqe_options = {
    'optimizers': ['COBYLA.8000', 'L_BFGS_B.8000', 'SLSQP.1000'],
    # 'optimizers': ['COBYLA.500'],
    'repeats': 10, 
    'seed': 42
}

pot = get_pot(pot_file, take_cos=False)
dvr_vqe = DVR_VQE(mol_params, pot, log_dir=scratch_dir + 'excited_states/mgnh_64/')
dvr_vqe.get_dvr_vqe(dvr_options, ansatz_options, vqe_options, excited_states=True)
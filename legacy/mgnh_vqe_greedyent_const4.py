import numpy as np
from lib.dvr2d import *
from lib.utils import *
from lib.vqe import DVR_VQE
from lib.pot_gen import get_pot

# from lib.local_utils import base_dir, scratch_dir

mol_params = mgnh_params
pot_file = base_dir + 'mgnh.txt'

# 32 points
dvr_options = {
    'type': 'jacobi',
    'N_R': 20, 
    'N_theta': 4,
    'l': 1,
    'K_max': 0,
    'r_min': 3,
    'r_max': 6, 
    'trunc': 0,
    'J': 0
}

ansatz_options = {
    'type': 'greedyent',
    'constructive': True,
    'layers': 40,
    'num_keep': 1, 
    'num_qubits': 5,
    'reps': 4,
    'partitions': [[0,1,2,3],[4]]
}

# 64 points
# dvr_options = {
#     'type': 'jacobi',
#     'J': 0,
#     'N_R': 40, 
#     'N_theta': 4,
#     'l': 1,
#     'K_max': 0,
#     'r_min': 3,
#     'r_max': 9, 
#     'trunc': 0
# }

# ansatz_options = {
#     'type': 'greedyent',
#     'constructive': True,
#     'layers': 60,
#     'num_keep': 1, 
#     'num_qubits': 6,
#     'reps': 4,
#     'partitions': [[0,1,2,3,4], [5]]
# }

vqe_options = {
    'optimizers': ['L_BFGS_B.8000', 'SLSQP.1000'],
    # 'optimizers': ['COBYLA.500'],
    'repeats': 5, 
    'seed': 42
}

pot = get_pot(pot_file, take_cos=False)
dvr_vqe = DVR_VQE(mol_params, pot, log_dir=scratch_dir + 'partitions/mgnh_32/012-34/')
dvr_vqe.get_dvr_vqe(dvr_options, ansatz_options, vqe_options, excited_states=False)
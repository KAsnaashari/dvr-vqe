import numpy as np
from lib.dvr2d import *
from lib.utils import *
from lib.vqe import DVR_VQE
from lib.pot_gen import get_pot

# from lib.local_utils import base_dir, scratch_dir

mol_params = arhcl_params
pot_file = base_dir + 'arhcl.txt'

# 32 points
# dvr_options = { 
#     'type': 'jacobi',
#     'N_R': 35, 
#     'N_theta': 4,
#     'l': 1,
#     'K_max': 0,
#     'r_min': 3.4,
#     'r_max': 5, 
#     'trunc': 0,
#     'J': 0
# }

# ansatz_options = {
#     'type': 'greedyent',
#     'constructive': True,
#     'layers': 30,
#     'num_keep': 1, 
#     'num_qubits': 5,
#     'reps': 3,
#     'partitions': [[0], [1,2,3,4]]
# }

# 64 points
dvr_options = {
    'type': 'jacobi',
    'J': 0,
    'N_R': 50, 
    'N_theta': 4,
    'l': 1,
    'K_max': 0,
    'r_min': 3,
    'r_max': 7, 
    'trunc': 0
}

ansatz_options = {
    'type': 'greedyent',
    'constructive': True,
    'layers': 45,
    'num_keep': 1, 
    'num_qubits': 6,
    'reps': 3,
    'partitions': [[0], [1,2,3,4,5]]
}

vqe_options = {
    'optimizers': ['L_BFGS_B.8000', 'SLSQP.1000'],
    # 'optimizers': ['COBYLA.100'],
    # 'repeats': 1, 
    'repeats': 5, 
    'seed': 42
}

pot = get_pot(pot_file, take_cos=True)
# dvr_vqe = DVR_VQE(mol_params, pot, log_dir=scratch_dir + 'partitions/arhcl_32/')
dvr_vqe = DVR_VQE(mol_params, pot, log_dir=scratch_dir + 'partitions/arhcl_64/')
# dvr_vqe = DVR_VQE(mol_params, pot, log_dir=scratch_dir + 'test/')
dvr_vqe.get_dvr_vqe(dvr_options, ansatz_options, vqe_options, excited_states=False)
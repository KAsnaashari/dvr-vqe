from math import cos
import numpy as np
from lib.dvr2d import *
from lib.utils import *
from lib.vqe import DVR_VQE
from lib.pot_gen import get_pot

# from local_utils import base_dir, scratch_dir

mol_params = mgnh_params
pot_file = base_dir + 'mgnh.txt'

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

vqe_options = {
    'optimizers': ['L_BFGS_B.8000', 'SLSQP.1000'],
    # 'optimizers': ['COBYLA.100'],
    # 'repeats': 1, 
    'repeats': 10, 
    'beta': [1000 for i in range(10)],
    'seed': 42
}

ansatz_options = {
    'type': 'twolocal',
    'entanglement': 'opt',
    'gates': [39, 4, 21, 8, 13, 19, 7, 34, 30, 14],
    'reps': 4,
}

pot = get_pot(pot_file, take_cos=False)
dvr_vqe = DVR_VQE(mol_params, pot, log_dir=scratch_dir + 'excited_states/')
dvr_vqe.get_dvr_vqe(dvr_options, ansatz_options, vqe_options, excited_states=True)
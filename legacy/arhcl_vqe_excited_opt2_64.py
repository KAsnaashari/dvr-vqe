from math import cos
import numpy as np
from lib.dvr2d import *
from lib.utils import *
from lib.vqe import DVR_VQE
from lib.pot_gen import get_pot

# from lib.local_utils import base_dir, scratch_dir

mol_params = arhcl_params
pot_file = base_dir + 'arhcl.txt'

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

vqe_options = {
    'optimizers': ['L_BFGS_B.8000', 'SLSQP.2000'],
    # 'optimizers': ['COBYLA.100'],
    # 'repeats': 1, 
    'repeats': 5, 
    'beta': [1000 for i in range(10)],
    'seed': 42
}

ansatz_options = {
    'type': 'twolocal',
    'entanglement': 'opt',
    'gates': [0, 7, 23, 30, 9, 20, 17, 45, 8, 14, 39, 46],
    'reps': 4,
}

pot = get_pot(pot_file, take_cos=True)
dvr_vqe = DVR_VQE(mol_params, pot, log_dir=scratch_dir + 'excited_states/arhcl_64/')
dvr_vqe.get_dvr_vqe(dvr_options, ansatz_options, vqe_options, excited_states=True, cont=5)
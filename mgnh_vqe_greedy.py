import numpy as np
from dvr2d import *
from utils import *
from vqe import DVR_VQE

# from local_utils import base_dir, scratch_dir

mol_params = mgnh_params
pot_file = base_dir + mol_params['name'] + '.txt'

dvr_options = {
    'N_R': 20, 
    'N_theta': 4,
    'l': 1,
    'K_max': 0,
    'r_min': 3,
    'r_max': 6, 
    'trunc': 0
}

vqe_options = {
    'optimizers': ['SLSQP.1000'],
    # 'optimizers': ['COBYLA.500'],
    'repeats': 3, 
    # 'repeats': 5, 
    'seed': 42
}

ansatz_options = {
    'type': 'greedy',
    'layers': 8,
    'add_h': True,
    'add_rs': True,
    'samples': 50,
    'num_keep': 5, 
    'max_gate': 2, 
    'num_qubits': 5
}

dvr_vqe = DVR_VQE(mol_params, pot_file, take_cos=False, log_dir=scratch_dir + 'greedy_circuits/')
dvr_vqe.get_dvr_vqe(dvr_options, ansatz_options, vqe_options, J=0, excited_states=False)
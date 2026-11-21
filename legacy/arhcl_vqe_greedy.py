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
    'N_R': 35, 
    'N_theta': 4,
    'l': 1,
    'K_max': 0,
    'r_min': 3.4,
    'r_max': 5, 
    'trunc': 0,
    'J': 0
}

vqe_options = {
    'optimizers': ['SLSQP.1000'],
    # 'optimizers': ['COBYLA.500'],
    'repeats': 5, 
    'seed': 42
}

ansatz_options = {
    'type': 'greedy',
    'layers': 12,
    'add_h': True,
    'add_rs': True,
    'samples': 20,
    'num_keep': 5, 
    'max_gate': 3, 
    'num_qubits': 5
}

pot = get_pot(pot_file, take_cos=True)
dvr_vqe = DVR_VQE(mol_params, pot, log_dir=scratch_dir + 'greedy_circuits/')
dvr_vqe.get_dvr_vqe(dvr_options, ansatz_options, vqe_options, excited_states=False)
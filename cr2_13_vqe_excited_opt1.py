from math import cos
import numpy as np
from lib.dvr1d import *
from lib.utils import *
from lib.vqe import DVR_VQE
from lib.pot_gen import get_pot_cr2

# from lib.local_utils import base_dir, scratch_dir

mol_params = cr2_params
spin = 13
mol_params['name'] += f'_{spin}'

# 16 points
params16 = [5.2, 9]
dvr_options = {
    'type': '1d',
    'box_lims': (params16[0], params16[1]),
    'dx': (params16[1] - params16[0]) / 16,
    'count': 16
}
vqe_options = {
    'optimizers': ['L_BFGS_B.8000', 'SLSQP.1000'],
    # 'optimizers': ['COBYLA.100'],
    # 'repeats': 1, 
    'repeats': 10, 
    'beta': [10000 for i in range(5)],
    'seed': 42
}

ansatz_options = {
    'type': 'twolocal',
    'entanglement': 'opt',
    'gates': [3, 1, 9],
    'reps': 2,
}

pot, lims = get_pot_cr2(spin)
dvr_vqe = DVR_VQE(mol_params, pot, log_dir=scratch_dir + 'excited_states/')
dvr_vqe.get_dvr_vqe(dvr_options, ansatz_options, vqe_options, excited_states=True)
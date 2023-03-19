import numpy as np
from lib.dvr1d import *
from lib.utils import *
from lib.vqe import DVR_VQE
from lib.pot_gen import get_pot_cr2

from lib.local_utils import base_dir, scratch_dir

mol_params = cr2_params
spin = 3
pot_file = base_dir + 'arhcl.txt'

dvr_options = {
    'type': '1d',
    'box_lims': (3, 4.5),
    'dx': 0.1
}

vqe_options = {
    # 'optimizers': ['COBYLA.8000', 'L_BFGS_B.8000', 'SLSQP.1000'],
    'optimizers': ['COBYLA.100'],
    'repeats': 1, 
    # 'repeats': 5, 
    'beta': [100, 100],
    'seed': 42
}

ansatz_options = {
    'type': 'twolocal',
    'rotation_blocks': ['ry'],
    'entanglement_blocks': 'cx',
    'entanglement': 'linear',
    'reps': 2
}

pot, lims = get_pot_cr2(spin)
dvr_vqe = DVR_VQE(mol_params, pot, log_dir=scratch_dir + 'test/')
rs = dvr_vqe.get_DVR_Rtheta(dvr_options)
print(rs.shape)
print(dvr_vqe.get_dvr_vqe(dvr_options, ansatz_options, vqe_options, excited_states=True))
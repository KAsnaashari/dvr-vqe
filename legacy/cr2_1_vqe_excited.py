import numpy as np
from lib.dvr1d import *
from lib.utils import *
from lib.vqe import DVR_VQE
from lib.pot_gen import get_pot_cr2

# from lib.local_utils import base_dir, scratch_dir

mol_params = cr2_params
spin = 1
mol_params['name'] += f'_{spin}'

# dvr_options = {
#     'type': '1d',
#     'box_lims': (3, 3.8),
#     'dx': 0.1
# }

params16 = [2.8, 4]
dvr_options = {
    'type': '1d',
    'box_lims': (params16[0], params16[1]),
    'dx': (params16[1] - params16[0]) / 16,
    'count': 16
}

vqe_options = {
    'optimizers': ['COBYLA.8000', 'L_BFGS_B.8000', 'SLSQP.1000'],
    # 'optimizers': ['COBYLA.100'],
    # 'repeats': 1, 
    'repeats': 5, 
    'beta': [10000 for i in range(5)],
    'seed': 42
}

reps_list = [1, 2, 3]
ansatz_options_list = [{
    'type': 'twolocal',
    'rotation_blocks': ['ry'],
    'entanglement_blocks': 'cx',
    'entanglement': 'linear',
    'reps': reps
} for reps in reps_list]

pot, lims = get_pot_cr2(spin)
for ansatz_options in ansatz_options_list:
    dvr_vqe = DVR_VQE(mol_params, pot, log_dir=scratch_dir + 'excited_states/cr2_16_new/')
    # rs = dvr_vqe.get_DVR_Rtheta(dvr_options)
    # print(rs.shape)
    dvr_vqe.get_dvr_vqe(dvr_options, ansatz_options, vqe_options, excited_states=True)
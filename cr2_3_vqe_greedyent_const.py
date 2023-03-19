import numpy as np
from lib.dvr1d import *
from lib.utils import *
from lib.vqe import DVR_VQE
from lib.pot_gen import get_pot_cr2

# from lib.local_utils import base_dir, scratch_dir

mol_params = cr2_params
spin = 3
mol_params['name'] += f'_{spin}'

# 16 points
params16 = [3.2, 4.5]
dvr_options = {
    'type': '1d',
    'box_lims': (params16[0], params16[1]),
    'dx': (params16[1] - params16[0]) / 16,
    'count': 16
}

ansatz_options_list = [{
    'type': 'greedyent',
    'constructive': True,
    'layers': 6 * reps,
    'num_keep': 1, 
    'num_qubits': 4,
    'reps': reps,
    'partitions': [[0,1,2,3,4,5]]
} for reps in [1, 2, 3]]

vqe_options = {
    'optimizers': ['L_BFGS_B.8000', 'SLSQP.1000'],
    # 'optimizers': ['COBYLA.100'],
    # 'repeats': 1, 
    'repeats': 5, 
    'seed': 42
}

pot, lims = get_pot_cr2(spin)
dvr_vqe = DVR_VQE(mol_params, pot, log_dir=scratch_dir + 'greedyent_circuits/cr2_16/')
# dvr_vqe = DVR_VQE(mol_params, pot, log_dir=scratch_dir + 'test/')
for ansatz_options in ansatz_options_list:
    dvr_vqe.get_dvr_vqe(dvr_options, ansatz_options, vqe_options, excited_states=False)
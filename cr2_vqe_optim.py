import numpy as np
from lib.dvr1d import *
from lib.utils import *
from lib.vqe import DVR_VQE
from lib.pot_gen import get_pot_cr2
from scipy.optimize import minimize, Bounds
import os

# from lib.local_utils import base_dir, scratch_dir

# spins = [1, 3, 5, 7, 9, 11, 13]
spins = [13]

vqe_options = {
    'optimizers': ['SLSQP.1000'],
    # 'optimizers': ['SLSQP.100'],
    # 'repeats': 1, 
    'repeats': 3, 
    'beta': [],
    'seed': 42
}

ansatz_options = {
    'type': 'twolocal',
    'rotation_blocks': ['ry'],
    'entanglement_blocks': 'cx',
    'entanglement': 'linear',
    'reps': 3
}

def get_gs_energy_vqe(params, dvr_vqe, n, log_file=None):
    r_min = params
    dx = 0.1
    
    dvr_options = {
        'type': '1d',
        'box_lims': (r_min, r_min + n * dx),
        'dx': dx,
        'count': 16
    }
    print_log(params, log_file)

    h_dvr = dvr_vqe.get_h_dvr(dvr_options)[:dvr_options['count'], :dvr_options['count']] * hartree
    num_qubits = int(np.log2(h_dvr.shape[0]))
    h_dvr_pauli = pauli_decompose(h_dvr)

    ansatz = dvr_vqe.get_ansatz(ansatz_options, num_qubits)
    converge_cnts, converge_vals, best_params, best_energies = dvr_vqe.opt_vqe(h_dvr_pauli, ansatz, vqe_options, 
                                                                                log_file=log_file)
    return converge_vals[0][-1]

for spin in spins:
    mol_params = cr2_params
    mol_params['name'] = f'cr2_{spin}'

    pot, lims = get_pot_cr2(spin)
    dvr_vqe = DVR_VQE(mol_params, pot, log_dir=scratch_dir + 'dvr_optim/')

    vqe_id = f"{dvr_vqe.mol_params['name']}_opt_{ansatz_options['type']}_{ansatz_options['entanglement']}_{ansatz_options['reps']}"
    
    if dvr_vqe.log_dir is not None:
        log_dir_id = dvr_vqe.log_dir + vqe_id + '/'
        offset = 0
        madedir = False
        while not madedir:
            if not os.path.exists(log_dir_id):
                os.mkdir(log_dir_id)
                madedir = True
            else:
                offset += 1
                log_dir_id = dvr_vqe.log_dir + vqe_id + f'({offset})/'

        log_file = log_dir_id + 'vqe.txt'
    else: 
        log_file = None

    print_log(mol_params['name'], log_file)
    print_log(lims, log_file)

    bounds = [(lims[0], lims[0] + 5)]
    fun16 = lambda p: get_gs_energy_vqe(p, dvr_vqe, 16, log_file=log_file)

    opt16 = minimize(fun16, (lims[0] + 2), method='L-BFGS-B', bounds=bounds, options={'verbose': 1})
    # opt16 = minimize(fun16, (lims[0] + 0.49, 0.05), method='trust-constr', bounds=Bounds(bounds[:, 0], bounds[:, 1]), options={})
    print_log((opt16.x, opt16.fun), log_file)
    print_log('------------------', log_file)
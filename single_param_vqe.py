import numpy as np
from lib.utils import *
# from lib.local_utils import *
from lib.vqe import DVR_VQE
from lib.pot_gen import get_pot
from lib.dvr2d import arhcl_params
from copy import deepcopy

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter

mol_params = arhcl_params
pot_file = base_dir + 'arhcl.txt'

# 32 points
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
    # 'optimizers': ['COBYLA.8000', 'L_BFGS_B.8000', 'SLSQP.1000'],
    'optimizers': ['COBYLA.500'],
    'repeats': 10, 
    'seed': None
}

def build_circuit(gates, n_qubits=5):
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter

    out = QuantumCircuit(5)
    theta = Parameter(f'$\\theta$')
    for g in gates:
        if g[0] == 'cnot':
            out.cnot(*g[1])
            
        if g[0] == 'cz':
            out.cz(*g[1])

        elif g[0] == 'ry':
            out.ry(g[2] * theta, g[1])

        elif g[0] == 'h':
            out.h(g[1])

        elif g[0] == 'y':
            out.y(g[1])

        elif g[0] == 'x':
            out.x(g[1])

        elif g[0] == 'rz':
            out.rz(theta, g[1])
    return out

pot = get_pot(pot_file, take_cos=True)
log_file = scratch_dir + 'single_params_test/' + 'log4.txt'
dvr_vqe = DVR_VQE(mol_params, pot, log_dir=scratch_dir + 'single_params_test/')
from lib.dvr2d import pauli_decompose

h_dvr = dvr_vqe.get_h_dvr(dvr_options) * hartree
h_dvr_pauli = pauli_decompose(h_dvr)
# print_log(h_dvr_pauli, log_file)

n_qubits = 5
n_select = 5
layers = 30
rands = 8
recalc_factors = 5
recalc_rands = 20
print_log(f'n_select: {n_select}', log_file)
print_log(f'layers: {layers}', log_file)
print_log(f'rands: {rands}', log_file)
print_log(f'recalc_factors: {recalc_factors}', log_file)
print_log(f'recalc_rands: {recalc_rands}', log_file)

init_ansatzes = []
for i in range(n_qubits):
    init_ansatzes.append([['ry', i, 1]])

init_best_ansatz = None
init_best_energy = None
for i in range(len(init_ansatzes)):
    ansatz = build_circuit(init_ansatzes[i])
    print_log(ansatz, log_file)
    converge_cnts, converge_vals, best_params, best_energies = dvr_vqe.opt_vqe(h_dvr_pauli, ansatz, vqe_options)
    print_log(best_energies, log_file)
    # print_log(best_params, log_file)
    if (init_best_energy is None) or (np.min(best_energies) < init_best_energy):
        init_best_ansatz = i
        init_best_energy = np.min(best_energies)
    print_log('-----------------------', log_file)
print_log(init_best_ansatz, log_file)
print_log(init_best_energy, log_file)

touched_qubits = [set([init_best_ansatz]) for i in range(n_select)]
ansatz0s = [init_ansatzes[init_best_ansatz]]
# print_log(ansatz0s, log_file)
layer_energies = [init_best_energy]
for l in range(layers):
    print_log(f'Layer: {l+1}', log_file)
    layer_ansatzes = []
    layer_gates = []
    # if (l + 1) % force_removal == 0:
    for a, ansatz0 in enumerate(ansatz0s):
        for g in range(len(ansatz0) - 1):
            ansatz = ansatz0[:g] + ansatz0[g+1:]
            if len(build_circuit(ansatz).parameters) > 0:
                layer_ansatzes.append(ansatz)
                layer_gates.append(ansatz0[g] + ['r'])
                
    if (l + 1) % recalc_factors == 0:
        for a, ansatz0 in enumerate(ansatz0s):
            layer_ansatzes.append(ansatz0)
            layer_gates.append(['recalc'])
            for rand in range(recalc_rands):
                ansatz = []
                for g in ansatz0:
                    if g[0] == 'ry':
                        x = np.random.uniform(low=-2, high=2)
                        ansatz.append(['ry', g[1], x])
                    else:
                        ansatz.append(g)
                layer_ansatzes.append(ansatz)
                layer_gates.append(['recalc'])
    else:
        for i in range(n_qubits):
            for a, ansatz0 in enumerate(ansatz0s):  
                for j in touched_qubits[a]:
                    if i != j:
                        ansatz = ansatz0 + [['cnot', [j, i]]]
                        layer_ansatzes.append(ansatz)
                        layer_gates.append(['cnot', [j, i]])
        # print_log(layer_ansatzes, log_file)
        for i in range(n_qubits):
            for a, ansatz0 in enumerate(ansatz0s):
                    ansatz = ansatz0 + [['h', i]]
                    layer_ansatzes.append(ansatz)
                    layer_gates.append(['h', i])
        # # print_log(layer_ansatzes, log_file)
        # for i in range(n_qubits):
        #     for a, ansatz0 in enumerate(ansatz0s):
        #             ansatz = ansatz0 + [['x', i]]
        #             layer_ansatzes.append(ansatz)
        
        for i in range(n_qubits):
            for a, ansatz0 in enumerate(ansatz0s):
                for rand in range(rands):
                    x = np.random.uniform(low=-2, high=2)
                    added = False
                    ansatz = deepcopy(ansatz0)
                    for g in ansatz[::-1]:
                        if isinstance(g[1], list) and (i in g[1]):
                            break
                        if g[1] == i:
                            if g[0] == 'ry':
                                g[2] += x
                                added = True
                            break
                    if not added:
                        ansatz += [['ry', i, x]]
                    layer_ansatzes.append(ansatz)
                    layer_gates.append(['ry', i, x])

    print_log(layer_ansatzes, log_file)
    print_log(layer_gates, log_file)
    print_log('----------------', log_file)

    best_ansatz = []
    best_energy = []
    for i in range(len(layer_ansatzes)):
        ansatz = build_circuit(layer_ansatzes[i])
        print_log(ansatz, log_file)
        print_log(layer_gates[i], log_file)
        converge_cnts, converge_vals, best_params, best_energies = dvr_vqe.opt_vqe(h_dvr_pauli, ansatz, vqe_options)
        print_log(best_params, log_file)
        if (len(best_energy) < n_select) or (np.any(best_energy > np.min(best_energies))):
            best_ansatz.append(i)
            best_energy.append(np.min(best_energies))
            if len(best_energy) > n_select:
                del best_energy[np.argmax(best_energy)]
                del best_ansatz[np.argmax(best_energy)]
        print_log(best_energy, log_file)
        print_log('-----------------------', log_file)
    
    print_log(best_ansatz, log_file)
    layer_energies.append(np.min(best_energy))
    touched_qubits = [set([]) for i in range(n_select)]
    ansatz0s = []
    for i, a in enumerate(best_ansatz):
        ansatz0s.append(layer_ansatzes[a])
        for g in layer_ansatzes[a]:
            if g[0] == 'cnot':
                touched_qubits[i].add(g[1][0])
                touched_qubits[i].add(g[1][1])
            else:
                touched_qubits[i].add(g[1])
    
    for i in range(len(ansatz0s)):
        print_log(ansatz0s[i], log_file)
        print_log(touched_qubits[i], log_file)
        print_log(best_energy[i], log_file)
    print_log('--------------------------------------------', log_file)

print_log(layer_energies, log_file)
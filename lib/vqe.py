from os import remove
from .utils import print_log

class DVR_VQE:
    def __init__(self, mol_params, pot_fun, log_dir=None) -> None:
        self.mol_params = mol_params
        self.log_dir = log_dir
        self.pot = pot_fun

    def gen_id(self):
        import time
        return str(int(time.time()))

    def get_DVR_Rtheta(self, dvr_options):
        from .utils import au_to_angs
        from .dvr2d import get_DVR_Rtheta
        from .dvr1d import get_dvr_r
        
        if dvr_options['type'] == 'jacobi':
            Rs_DVR, Xs_K = get_DVR_Rtheta(dvr_options)
            return Rs_DVR, Xs_K
        elif dvr_options['type'] == '1d':
            Rs_DVR = get_dvr_r(dvr_options)
            return Rs_DVR

    def get_h_dvr(self, dvr_options, J=0):
        if dvr_options['type'] == '1d':
            from .dvr1d import get_ham_DVR
            return get_ham_DVR(self.pot, dvr_options, mol_params=self.mol_params)
        elif dvr_options['type'] == 'jacobi':
            from .dvr2d import get_ham_DVR
            return get_ham_DVR(self.pot, dvr_options, mol_params=self.mol_params)

    def get_ansatz(self, ansatz_options, num_qubits):
        from qiskit.circuit.library import TwoLocal
        from .greedy_circs import build_circuit_ent
        
        if 'gates' in ansatz_options.keys():
            options = {
                'num_qubits': num_qubits,
                'reps': ansatz_options['reps'],
                'constructive': True
            }
            return build_circuit_ent(options, ansatz_options['gates'], simplify=True)

        return TwoLocal(num_qubits, rotation_blocks=ansatz_options['rotation_blocks'], entanglement_blocks=ansatz_options['entanglement_blocks'], entanglement=ansatz_options['entanglement'], reps=ansatz_options['reps']).decompose()

    def opt_vqe(self, h_dvr_pauli, ansatz, vqe_options, log_file=None, opt_params=None):
        import numpy as np
        from .utils import print_log
        from qiskit import Aer
        from qiskit.algorithms import VQE
        from qiskit.utils import QuantumInstance, algorithm_globals

        converge_cnts = np.empty([len(vqe_options['optimizers'])], dtype=object)
        converge_vals = np.empty([len(vqe_options['optimizers'])], dtype=object)
        best_params = np.empty([len(vqe_options['optimizers'])], dtype=object)
        best_energies = np.empty([len(vqe_options['optimizers'])], dtype=float)

        params = None
        # params = np.array([0.0 for i in range(ansatz1.num_parameters)])
        for i in range(len(vqe_options['optimizers'])):
            optimizer = self.parse_optimizer_string(vqe_options['optimizers'][i])
            print_log('Optimizer: {}        '.format(type(optimizer).__name__), log_file)
            
            if vqe_options['seed'] is not None:
                algorithm_globals.random_seed = vqe_options['seed']

            def store_intermediate_result(eval_count, parameters, mean, std):
                counts.append(eval_count)
                values.append(mean)
                if eval_count % 10 == 0 or eval_count == 1:
                    print_log(f'{eval_count}, {mean}', log_file, end='', overwrite=True if eval_count > 1 else False)
            
            best_res = None
            for j in range(vqe_options['repeats']):
                counts = []
                values = []
                vqe = VQE(ansatz, optimizer, callback=store_intermediate_result, initial_point=params, 
                        quantum_instance=QuantumInstance(backend=Aer.get_backend('statevector_simulator')))
                if opt_params is None:
                    result = vqe.compute_minimum_eigenvalue(operator=h_dvr_pauli)
                else:
                    result = optimizer.minimize(lambda p: self.excited_cost_function(h_dvr_pauli, ansatz, opt_params, vqe_options['beta'], vqe, p), 
                                                np.random.rand(ansatz.num_parameters) * 2 * np.pi)
                print_log('', log_file, end='\n')

                if (best_res is None) or (values[-1] < best_res):
                    best_res = values[-1]
                    if opt_params is None:
                        best_params[i] = result.optimal_point
                    else:
                        best_params[i] = result.x
                    converge_cnts[i] = np.asarray(counts)
                    converge_vals[i] = np.asarray(values)
                    best_energies[i] = best_res
                    
        print_log('Optimization complete', log_file)

        return converge_cnts, converge_vals, best_params, best_energies

    def excited_cost_function(self, h_dvr_pauli, ansatz, opt_p_list, betas, vqe, p):
        from qiskit import Aer

        backend = Aer.get_backend('statevector_simulator')
        out = vqe.get_energy_evaluation(h_dvr_pauli)(p)
        new_state = ansatz.assign_parameters(p)

        for beta, opt_p in zip(betas, opt_p_list):
            state = ansatz.assign_parameters(opt_p)
            circ = state.compose(new_state.inverse())
            result = backend.run(circ).result()
            sv = result.get_statevector()

            out += beta * sv.probabilities()[0]
        
        return out

    def parse_optimizer_string(self, s):
        from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, SLSQP, NELDER_MEAD

        name, maxiter = s.split('.')
        maxiter = int(maxiter)
        if name == 'COBYLA':
            return COBYLA(maxiter=maxiter)
        elif name == 'L_BFGS_B':
            return L_BFGS_B(maxfun=maxiter)
        elif name == 'SLSQP':
            return SLSQP(maxiter=maxiter)
        elif name == 'NELDER_MEAD':
            return NELDER_MEAD(maxfev=maxiter)

    def print_options(self, options_list, options_file):
        import json

        with open(options_file, 'w') as f:
            f.write(json.dumps(options_list))

    def get_greedy_ansatz(self, ansatz_options, best_circs, log_file=None):
        import numpy as np
        from .greedy_circs import build_circuit

        layer_list = [np.random.randint(0, ansatz_options['max_gate'] + 1, (ansatz_options['num_qubits'], 1)) 
                        for _ in range(ansatz_options['samples'])]
        circ_list = []
        if best_circs is None or len(best_circs) == 0:
            circ_list = layer_list
        else:
            for c in best_circs:
                circ_list.extend([np.concatenate([c, l], axis=-1) for l in layer_list])
        
        ansatz_list = []
        circ_list_final = []
        ansatz_string_list = []
        for c in circ_list:
            ansatz = build_circuit(c, ansatz_options, opt_level=2)
            s = str(ansatz)
            if (ansatz.num_parameters > 0) and (s not in ansatz_string_list):
                ansatz_list.append(ansatz)
                circ_list_final.append(c)
                ansatz_string_list.append(s)
                
        return circ_list_final, ansatz_list

    def opt_greedy_vqe(self, h_dvr_pauli, ansatz_options, vqe_options, log_file=None, prev_circs=None):
        import numpy as np
        from .utils import print_log
        from qiskit import Aer
        from qiskit.algorithms import VQE
        from qiskit.utils import QuantumInstance, algorithm_globals

        if prev_circs is None or len(prev_circs) == 0:
            l = 0
        else:
            l = prev_circs[0].shape[0]

        circ_list, ansatz_list = self.get_greedy_ansatz(ansatz_options, prev_circs, log_file)
        
        converge_cnts_list, converge_vals_list, best_params_list, best_energies_list = [], [], [], []
        for i, ansatz in enumerate(ansatz_list):
            print_log('--------', log_file)
            print_log(f'Optimizing ansatz {i+1}/{len(ansatz_list)} in layer {l+1}', log_file)
            converge_cnts, converge_vals, best_params, best_energies = self.opt_vqe(h_dvr_pauli, ansatz, vqe_options, 
                                                                                    log_file=log_file)
            opt_ind = np.argmin(best_energies)
            converge_cnts_list.append(converge_cnts[opt_ind])
            converge_vals_list.append(converge_vals[opt_ind])
            best_params_list.append(best_params[opt_ind])
            best_energies_list.append(best_energies[opt_ind])

        inds = np.argsort(best_energies_list)
        keep_inds = inds[:ansatz_options['num_keep']]
        best_circs = []
        converge_cnts, converge_vals, best_params, best_energies = [], [], [], []
        for ind in keep_inds:
            best_circs.append(circ_list[ind])
            converge_cnts.append(converge_cnts_list[ind])
            converge_vals.append(converge_vals_list[ind])
            best_params.append(best_params_list[ind])
            best_energies.append(best_energies_list[ind])

        return best_circs, converge_cnts, converge_vals, best_params, best_energies

    def opt_greedyent_vqe(self, h_dvr_pauli, ansatz_options, vqe_options, allowed_gates, log_file=None, gates=None):
        import numpy as np
        from .utils import print_log
        from .greedy_circs import build_circuit_ent

        num_qubits = ansatz_options['num_qubits']
        reps = ansatz_options['reps']
        num_gates = num_qubits * (num_qubits - 1) // 2

        ansatz_list = []
        gate_list = []
        for i in range(num_gates * reps):
            if (i not in gates) and (i in allowed_gates):
                ansatz_list.append(build_circuit_ent(ansatz_options, gates + [i], simplify=True))
                gate_list.append(i)

        if gates is None:
            l = 0
        else:
            l = len(gates)
        
        converge_cnts_list, converge_vals_list, best_params_list, best_energies_list = [], [], [], []
        for i, ansatz in enumerate(ansatz_list):
            print_log('--------', log_file)
            print_log(ansatz, log_file)
            print_log(f'Optimizing ansatz {i+1}/{len(ansatz_list)} in layer {l+1}', log_file)
            converge_cnts, converge_vals, best_params, best_energies = self.opt_vqe(h_dvr_pauli, ansatz, vqe_options, 
                                                                                    log_file=log_file)
            opt_ind = np.argmin(best_energies)
            converge_cnts_list.append(converge_cnts[opt_ind])
            converge_vals_list.append(converge_vals[opt_ind])
            best_params_list.append(best_params[opt_ind])
            best_energies_list.append(best_energies[opt_ind])

        ind = np.argmin(best_energies_list)
        best_gate = gate_list[ind]
        converge_cnts = converge_cnts_list[ind]
        converge_vals = converge_vals_list[ind]
        best_params = best_params_list[ind]
        best_energies = best_energies_list[ind]

        return best_gate, converge_cnts, converge_vals, best_params, best_energies

    def get_dvr_vqe(self, dvr_options, ansatz_options, vqe_options, excited_states=False, cont=0):
        from .dvr2d import pauli_decompose
        from .utils import hartree
        from .greedy_circs import build_circuit, build_circuit_ent, find_allowed_gates
        from .block_circs import block_divider, build_circuit_from_s
        import os
        import numpy as np

        h_dvr = self.get_h_dvr(dvr_options, J=0) * hartree
        num_qubits = int(np.log2(h_dvr.shape[0]))
        h_dvr_pauli = pauli_decompose(h_dvr)

        vqe_id = self.gen_id()
        if ansatz_options['type'] == 'twolocal':
            vqe_id = f"{self.mol_params['name']}_{ansatz_options['type']}_{ansatz_options['entanglement']}_{ansatz_options['reps']}"
        elif ansatz_options['type'] == 'greedy':
            vqe_id = f"{self.mol_params['name']}_{ansatz_options['type']}"
        elif ansatz_options['type'] == 'greedyent':
            vqe_id = f"{self.mol_params['name']}_{ansatz_options['type']}_{ansatz_options['reps']}"
            if ansatz_options['constructive']:
                vqe_id += '_const'
        elif ansatz_options['type'] == 'block':
            vqe_id = f"{self.mol_params['name']}_{ansatz_options['type']}"

        if self.log_dir is not None:
            log_dir_id = self.log_dir + vqe_id + '/'
            offset = 0
            madedir = False
            while (not madedir) and (cont == 0):
                if not os.path.exists(log_dir_id):
                    os.mkdir(log_dir_id)
                    madedir = True
                else:
                    offset += 1
                    log_dir_id = self.log_dir + vqe_id + f'({offset})/'

            log_file = log_dir_id + 'vqe.txt'
        else: 
            log_file = None

        self.print_options([self.mol_params, dvr_options, ansatz_options, vqe_options], log_dir_id + 'options.json')
        
        if ansatz_options['type'] == 'greedy':
            best_circs = []
            for l in range(ansatz_options['layers']):
                print_log(f'Layer {l+1} ...', log_file)
                best_circs, converge_cnts, converge_vals, best_params, best_energies = self.opt_greedy_vqe(h_dvr_pauli, 
                                                                                            ansatz_options, 
                                                                                            vqe_options,
                                                                                            prev_circs=best_circs, 
                                                                                            log_file=log_file)
                print_log('*****************************************', log_file)
                print_log(f'Layer {l+1} results:', log_file)
                for i in range(len(best_circs)):
                    ansatz = build_circuit(best_circs[i], ansatz_options)
                    print_log(str(ansatz), log_file)
                    print_log(f'Depth : {ansatz.depth()}, Energy: {best_energies[i]}, Counts: {converge_cnts[i][-1]}', log_file)

                if self.log_dir is not None:
                    np.savez(log_dir_id + f'vqe_greedy{l+1}.npz', counts=converge_cnts, vals=converge_vals, 
                            params=best_params, energies=best_energies, best_circs=best_circs)

                print_log('*****************************************', log_file)
            
            ind = np.argmin(best_energies)
            ansatz = build_circuit(best_circs[ind], ansatz_options)

            print_log(f'Final circuit after {ansatz_options["layers"]} layers: ', log_file)
            print_log(str(ansatz), log_file)
            print_log(f'Depth : {ansatz.depth()}, Energy: {best_energies[ind]}, Counts: {converge_cnts[ind][-1]}', log_file)
            if self.log_dir is not None:
                np.savez(log_dir_id + f'vqe_greedy_final.npz', counts=converge_cnts[ind], vals=converge_vals[ind], 
                        params=best_params[ind], energies=best_energies[ind], best_circs=best_circs[ind])
        elif ansatz_options['type'] == 'greedyent':
            gates = []
            allowed_gates = find_allowed_gates(ansatz_options['num_qubits'], ansatz_options['reps'], ansatz_options['partitions'])
            if cont > 0:
                prev_data = np.load(log_dir_id + f'vqe_greedyent{cont}.npz')
                gates = list(prev_data['removed_gates'])

            for l in range(cont, len(allowed_gates)):
                print_log(gates, log_file)
                print_log(f'Layer {l+1} ...', log_file)
                best_gate, converge_cnts, converge_vals, best_param, best_energy = self.opt_greedyent_vqe(h_dvr_pauli, 
                                                                                            ansatz_options, vqe_options, 
                                                                                            allowed_gates, gates=gates, 
                                                                                            log_file=log_file)
                gates += [best_gate]
                print_log('*****************************************', log_file)
                print_log(f'Layer {l+1} results:', log_file)
                ansatz = build_circuit_ent(ansatz_options, gates, simplify=True)
                print_log(str(ansatz), log_file)
                print_log(f'Depth : {ansatz.depth()}, Energy: {best_energy}, Counts: {converge_cnts[-1]}', log_file)

                if self.log_dir is not None:
                    np.savez(log_dir_id + f'vqe_greedyent{l+1}.npz', counts=converge_cnts, vals=converge_vals, 
                            params=best_param, energies=best_energy, removed_gates=gates)

                print_log('*****************************************', log_file)
            
            ansatz = build_circuit_ent(ansatz_options, gates, simplify=True)

            print_log(f'Final circuit after {ansatz_options["layers"]} layers: ', log_file)
            print_log(str(ansatz), log_file)
            print_log(f'Depth : {ansatz.depth()}, Energy: {best_energy}, Counts: {converge_cnts[-1]}', log_file)
            if self.log_dir is not None:
                np.savez(log_dir_id + f'vqe_greedy_final.npz', counts=converge_cnts, vals=converge_vals, 
                        params=best_param, energies=best_energy, removed_gates=gates)
        elif ansatz_options['type'] == 'block':
            print_log(f'Block circuits ...', log_file)
            blocks = block_divider(list(range(num_qubits)))
            print_log(blocks, log_file)
            best_block = None
            best_block_energy = None

            for b in range(len(blocks)):
                block = blocks[b]
                if b < cont:
                    prev_data = np.load(log_dir_id + f'vqe_block{b+1}.npz')
                    best_energies = prev_data['energies']
                else:
                    ansatz = build_circuit_from_s(block)
                    print_log(f'Block circuit {b+1} ...', log_file)
                    print_log('-----------------------', log_file)
                    print_log(ansatz, log_file)
                    converge_cnts, converge_vals, best_params, best_energies = self.opt_vqe(h_dvr_pauli, ansatz, vqe_options, 
                                                                                            log_file=log_file)
                    print_log('*****************************************', log_file)
                    print_log(f'Block circuit {b+1} results:', log_file)
                    for i in range(len(best_energies)):
                        optimizer = self.parse_optimizer_string(vqe_options['optimizers'][i])
                        print_log(f'{type(optimizer).__name__}, Energy: {best_energies[i]}, Counts: {converge_cnts[i][-1]}', log_file)

                    if self.log_dir is not None:
                        np.savez(log_dir_id + f'vqe_block{b+1}.npz', counts=converge_cnts, vals=converge_vals, params=best_params, energies=best_energies)

                if (best_block is None) or (np.min(best_energies) < best_block_energy):
                    best_block = block
                    best_block_energy = np.min(best_energies)

            print_log('*****************************************', log_file)
        
            ansatz = build_circuit_from_s(best_block)

            print_log(f'Best block circuit: {best_block}', log_file)
            print_log(str(ansatz), log_file)
            print_log(f'Depth : {ansatz.depth()}, Energy: {best_block_energy}', log_file)
        
        else:
            ansatz = self.get_ansatz(ansatz_options, num_qubits)
            opt_params = None
            if cont == 0:
                print_log(ansatz, log_file)
                print_log('-----------------------------------------', log_file)
                converge_cnts, converge_vals, best_params, best_energies = self.opt_vqe(h_dvr_pauli, ansatz, vqe_options, 
                                                                                        log_file=log_file)
                print_log('*****************************************', log_file)
                for i in range(len(best_energies)):
                    optimizer = self.parse_optimizer_string(vqe_options['optimizers'][i])
                    print_log(f'{type(optimizer).__name__}, Energy: {best_energies[i]}, Counts: {converge_cnts[i][-1]}', log_file)

                if self.log_dir is not None:
                    np.savez(log_dir_id + 'vqe.npz', counts=converge_cnts, vals=converge_vals, params=best_params, energies=best_energies)
            else:
                gs_data = np.load(log_dir_id + f'vqe.npz', allow_pickle=True)
                best_energies = gs_data['energies']
                opt_params = [gs_data['params'][np.argmin(best_energies)]]
                for i in range(cont):
                    prev_data = np.load(log_dir_id + f'vqe_excited{i+1}.npz', allow_pickle=True)
                    best_energies1 = prev_data['energies']
                    opt_params.append(prev_data['params'][np.argmin(best_energies1)])

            if excited_states:
                if opt_params is None:
                    opt_params = [best_params[np.argmin(best_energies)]]
                for i in range(cont, len(vqe_options['beta'])):
                    vqe_options_excited = dict(vqe_options)
                    vqe_options_excited['beta'] = vqe_options_excited['beta'][:i+1]

                    print_log('-----------------------------------------', log_file)
                    print_log(f'Excited state ({i+1}) calculations', log_file)
                    converge_cnts1, converge_vals1, best_params1, best_energies1 = self.opt_vqe(h_dvr_pauli, ansatz, vqe_options_excited, 
                                                                                        log_file=log_file, opt_params=opt_params)
                    if self.log_dir is not None:
                        np.savez(log_dir_id + f'vqe_excited{i+1}.npz', counts=converge_cnts1, vals=converge_vals1, 
                                params=best_params1, energies=best_energies1)
                    
                    opt_params.append(best_params1[np.argmin(best_energies1)])
                    print_log('*****************************************', log_file)

                    for i in range(len(best_energies1)):
                        optimizer = self.parse_optimizer_string(vqe_options['optimizers'][i])
                        print_log(f'{type(optimizer).__name__}, Energy: {best_energies1[i]}, Counts: {converge_cnts1[i][-1]}', log_file)
                    
        print_log('-----------------------------------------', log_file)
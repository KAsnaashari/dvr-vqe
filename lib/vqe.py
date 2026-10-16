from os import remove, cpu_count
import numpy as np
from .utils import print_log
from multiprocessing import Pool

def run_vqe(n, max_n, l, ansatz, h_dvr_pauli, vqe_options, log_file=None, noise_model=None):
    print_log('--------', log_file)
    print_log(f'Optimizing ansatz {n+1}/{max_n} in layer {l+1}', log_file)
    converge_cnts, converge_vals, best_params, best_energies = opt_vqe(h_dvr_pauli, ansatz, vqe_options, noise_model=noise_model,
                                                                            log_file=log_file)
    opt_ind = np.argmin(best_energies)
    return converge_cnts[opt_ind], converge_vals[opt_ind], best_params[opt_ind], best_energies[opt_ind]

def parse_optimizer_string(s, ansatz=None, sampler=None):
    from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, SLSQP, NELDER_MEAD, QNSPSA

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
    elif name == 'QNSPSA':
        return QNSPSA(QNSPSA.get_fidelity(ansatz, sampler), maxiter=maxiter, blocking=True, allowed_increase=None)


def opt_vqe(h_dvr_pauli, ansatz, vqe_options, log_file=None, opt_params=None, noise_model=None):
    import numpy as np
    from .utils import print_log
    from qiskit_algorithms.minimum_eigensolvers import VQE
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
    from qiskit_aer.noise import NoiseModel
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import EstimatorV2 as AerEstimator
    from qiskit_aer.primitives import SamplerV2 as AerSampler
    from qiskit_algorithms.utils import algorithm_globals

    converge_cnts = np.empty([len(vqe_options['optimizers'])], dtype=object)
    converge_vals = np.empty([len(vqe_options['optimizers'])], dtype=object)
    best_params = np.empty([len(vqe_options['optimizers'])], dtype=object)
    best_energies = np.empty([len(vqe_options['optimizers'])], dtype=float)

    for i in range(len(vqe_options['optimizers'])):
        if noise_model == 'sherbrooke':
            fake = FakeSherbrooke()
            nm = NoiseModel.from_backend(fake)
            estimator = AerEstimator(options={"run_options": {"shots": 20_000, "seed": 123},
                                                "backend_options": {"noise_model": nm,"basis_gates": fake.configuration().basis_gates,
                                                                    "basis_gates": fake.configuration().basis_gates,
                                                                    "coupling_map": fake.configuration().coupling_map}})
            sampler = AerSampler(options={"run_options": {"shots": 20_000, "seed": 123},
                                                "backend_options": {"noise_model": nm,"basis_gates": fake.configuration().basis_gates,
                                                                    "basis_gates": fake.configuration().basis_gates,
                                                                    "coupling_map": fake.configuration().coupling_map}})
        else:
            estimator = AerEstimator()
            sampler = AerSampler()

        optimizer = parse_optimizer_string(vqe_options['optimizers'][i], ansatz, sampler)
        print_log('Optimizer: {}        '.format(type(optimizer).__name__), log_file)
        
        if vqe_options['seed'] is not None:
            algorithm_globals.random_seed = vqe_options['seed']

        def store_intermediate_result(eval_count, parameters, mean, std):
            counts.append(eval_count)
            values.append(mean)
            if eval_count % 1 == 0 or eval_count == 1:
                print_log(f'{eval_count}, {mean}', log_file, end='', overwrite=True if eval_count > 1 else False)
        
        best_res = None
        for j in range(vqe_options['repeats']):
            counts = []
            values = []
            params = np.random.uniform(low=-2 * np.pi, high=2 * np.pi, size=len(ansatz.parameters))

            vqe = VQE(estimator=estimator,
                ansatz=ansatz,
                optimizer=optimizer,
                callback=store_intermediate_result,
                initial_point=params
            )
            if opt_params is None:
                result = vqe.compute_minimum_eigenvalue(operator=h_dvr_pauli)
            else:
                excited_cost = excited_cost_function(h_dvr_pauli, ansatz, opt_params,
                                                            vqe_options['beta'], estimator,
                                                            noise_model=noise_model, 
                                                            callback=store_intermediate_result)
                x0 = np.random.rand(ansatz.num_parameters) * 2 * np.pi
                result = optimizer.minimize(excited_cost, x0)
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

def swap_test_overlap_sq(ansatz, p_a, p_b, estimator):
    """Return |<psi(p_a)|psi(p_b)>|^2 via SWAP test measured with noise."""
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import Sampler
    
    n = ansatz.num_qubits
    qc_a = ansatz.assign_parameters(p_a, inplace=False)
    qc_b = ansatz.assign_parameters(p_b, inplace=False)

    qc = QuantumCircuit(1 + 2*n, 1)  # ancilla + two registers, measure ancilla
    # put the two states into the two registers
    qc = qc.compose(qc_a, qubits=range(1, 1+n))
    qc = qc.compose(qc_b, qubits=range(1+n, 1+2*n))

    # SWAP test
    qc.h(0)
    for i in range(n):
        qc.cswap(0, 1+i, 1+n+i)
    qc.h(0)
    qc.measure(0, 0)

    # run directly on the backend with noise & shots
    sampler = Sampler(backend_options=estimator.options.backend_options, run_options=estimator.options.run_options)
    quasi = sampler.run([qc]).result().quasi_dists[0]

    p0 = quasi.get(0, 0.0)  # prob ancilla=0
    return max(0.0, min(1.0, 2*p0 - 1))  # |<ψ|φ>|^2 = 2*P(anc=0) - 1

def excited_cost_function(h_dvr_pauli, ansatz, opt_p_list, betas, estimator, noise_model=None, callback=None):
    from qiskit.quantum_info import Statevector

    eval_count = 0
    if noise_model is None:
        prev_statevectors = []
        for opt_p in opt_p_list:
            circ_ref = ansatz.assign_parameters(opt_p, inplace=False)
            sv_ref = Statevector.from_instruction(circ_ref)
            prev_statevectors.append(sv_ref)

    def cost(p):
        # 1) Energy expectation via Estimator
        # Estimator.run accepts a list of (circuit, observable, param_values) tuples
        job = estimator.run([(ansatz, h_dvr_pauli, list(p))])
        result = job.result()[0]
        energy = result.data.evs

        # 2) Orthogonality penalties via overlap with previously found states
        if noise_model is None:
            circ_new = ansatz.assign_parameters(p, inplace=False)
            sv_new = Statevector.from_instruction(circ_new)

            penalty = 0.0
            for beta, sv_ref in zip(betas, prev_statevectors):
                # |<psi_ref | psi_new>|^2
                overlap_prob = abs(sv_ref.inner(sv_new))**2
                penalty += beta * overlap_prob
        else:
            penalty = 0.0
            for beta, ref_p in zip(betas, opt_p_list):
                ov2 = swap_test_overlap_sq(ansatz, ref_p, p, estimator)
                penalty += beta * ov2

        nonlocal eval_count
        eval_count += 1
        if callback is not None:
            callback(eval_count, p, energy + penalty, 0)
            
        return float(energy + penalty)

    return cost

class DVR_VQE:
    def __init__(self, mol_params, pot_fun, log_dir=None, noise_model=None, multiprocess=False) -> None:
        self.mol_params = mol_params
        self.log_dir = log_dir
        self.pot = pot_fun
        self.noise_model = noise_model
        self.multiprocess = multiprocess

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

    def get_h_dvr(self, dvr_options, J=0, offdiagonals=None):
        if dvr_options['type'] == '1d':
            from .dvr1d import get_ham_DVR
            ham_dvr = get_ham_DVR(self.pot, dvr_options, mol_params=self.mol_params)
            if offdiagonals is not None:
                mask = np.tri(ham_dvr.shape[0], k=offdiagonals) - np.tri(ham_dvr.shape[0], k=-offdiagonals - 1)
                ham_dvr[np.logical_not(np.asarray(mask, dtype=bool))] = 0
            return ham_dvr
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

    # def excited_cost_function(self, h_dvr_pauli, ansatz, opt_p_list, betas, vqe, p):
    #     from qiskit import Aer

    #     backend = Aer.get_backend('statevector_simulator')
    #     out = vqe.get_energy_evaluation(h_dvr_pauli)(p)
    #     new_state = ansatz.assign_parameters(p)

    #     for beta, opt_p in zip(betas, opt_p_list):
    #         state = ansatz.assign_parameters(opt_p)
    #         circ = state.compose(new_state.inverse())
    #         result = backend.run(circ).result()
    #         sv = result.get_statevector()

    #         out += beta * sv.probabilities()[0]
        
    #     return out

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

        if prev_circs is None or len(prev_circs) == 0:
            l = 0
        else:
            l = prev_circs[0].shape[0]

        circ_list, ansatz_list = self.get_greedy_ansatz(ansatz_options, prev_circs, log_file)
        
        converge_cnts_list, converge_vals_list, best_params_list, best_energies_list = [], [], [], []
        for i, ansatz in enumerate(ansatz_list):
            print_log('--------', log_file)
            print_log(f'Optimizing ansatz {i+1}/{len(ansatz_list)} in layer {l+1}', log_file)
            converge_cnts, converge_vals, best_params, best_energies = opt_vqe(h_dvr_pauli, ansatz, vqe_options, noise_model=self.noise_model,
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

        if self.multiprocess:
            input_list = [(n, len(ansatz_list), l, ansatz, h_dvr_pauli, vqe_options, log_file, self.noise_model) for n, ansatz in enumerate(ansatz_list)]
            with Pool(min(cpu_count(), len(ansatz_list))) as pool:
                results = pool.starmap(run_vqe, input_list)
            for res in results:
                converge_cnts_list.append(res[0])
                converge_vals_list.append(res[1])
                best_params_list.append(res[2])
                best_energies_list.append(res[3])
        else:
            for i, ansatz in enumerate(ansatz_list):
                print_log('--------', log_file)
                print_log(ansatz, log_file)
                print_log(f'Optimizing ansatz {i+1}/{len(ansatz_list)} in layer {l+1}', log_file)
                converge_cnts, converge_vals, best_params, best_energies = opt_vqe(h_dvr_pauli, ansatz, vqe_options, noise_model=self.noise_model,
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
                    converge_cnts, converge_vals, best_params, best_energies = opt_vqe(h_dvr_pauli, ansatz, vqe_options, noise_model=self.noise_model,
                                                                                            log_file=log_file)
                    print_log('*****************************************', log_file)
                    print_log(f'Block circuit {b+1} results:', log_file)
                    for i in range(len(best_energies)):
                        optimizer = parse_optimizer_string(vqe_options['optimizers'][i])
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
        
        elif ansatz_options['type'] == 'twolocal':
            ansatz = self.get_ansatz(ansatz_options, num_qubits)
            opt_params = None
            if cont == 0:
                print_log(ansatz, log_file)
                print_log('-----------------------------------------', log_file)
                converge_cnts, converge_vals, best_params, best_energies = opt_vqe(h_dvr_pauli, ansatz, vqe_options, noise_model=self.noise_model,
                                                                                        log_file=log_file)
                print_log('*****************************************', log_file)
                for i in range(len(best_energies)):
                    optimizer = parse_optimizer_string(vqe_options['optimizers'][i])
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
                    converge_cnts1, converge_vals1, best_params1, best_energies1 = opt_vqe(h_dvr_pauli, ansatz, vqe_options_excited, noise_model=self.noise_model,
                                                                                        log_file=log_file, opt_params=opt_params)
                    if self.log_dir is not None:
                        np.savez(log_dir_id + f'vqe_excited{i+1}.npz', counts=converge_cnts1, vals=converge_vals1, 
                                params=best_params1, energies=best_energies1)
                    
                    opt_params.append(best_params1[np.argmin(best_energies1)])
                    print_log('*****************************************', log_file)

                    for i in range(len(best_energies1)):
                        optimizer = parse_optimizer_string(vqe_options['optimizers'][i])
                        print_log(f'{type(optimizer).__name__}, Energy: {best_energies1[i]}, Counts: {converge_cnts1[i][-1]}', log_file)

            return opt_params
                    
        print_log('-----------------------------------------', log_file)
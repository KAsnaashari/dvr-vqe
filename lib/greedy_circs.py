"""Circuit builders for the greedy and greedy-entangling ansatze."""

def build_circuit(c, ansatz_options, opt_level=2):
    """Build a circuit from an integer gate-code array (greedy ansatz).

    Args:
        c: (num_qubits, depth) integer array. Code 1 = RY (skipped if the qubit already ends in an
            RY), 2 = CNOT from qubit q to q-1, 3 = CZ on (q, q-2), 0 = nothing.
        ansatz_options: Needs ``add_h`` (start with Hadamards) and ``add_rs`` (start with RY on
            every qubit).
        opt_level: Qiskit transpiler optimisation level applied to the result.

    Returns:
        Transpiled QuantumCircuit.
    """
    import numpy as np
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Parameter

    n_qubits = c.shape[0]
    depth = c.shape[1]
    out = QuantumCircuit(n_qubits)
    n_params = np.count_nonzero(c == 1) + (n_qubits if ansatz_options['add_rs'] else 0)
    if n_params == 0:
        theta = Parameter('X[0]')
        out.ry(theta, 0)
    thetas = [Parameter(f'X[{i}]') for i in range(n_params)]
    p = 0

    last_r = [False for i in range(n_qubits)]
    if ansatz_options['add_h']:
        for i in range(n_qubits):
            out.h(i)
    if ansatz_options['add_rs']:
        for i in range(n_qubits):
            out.ry(thetas[p], i)
            last_r[i] = True
            p += 1
    
    for i in range(depth):
        layer = c[:, i]
        for q, gate in enumerate(layer):
            if gate == 0:
                pass
            # elif gate == 1:
            #     out.h(q)
            #     last_r[q] = False
            elif gate == 1 and not last_r[q]:
                out.ry(thetas[p], q)
                last_r[q] = True
                p += 1
            elif gate == 2:
                out.cx(q, q - (gate - 1))
                last_r[q] = False
                last_r[q - (gate - 1)] = False
            elif gate == 3:
                out.cz(q, q - (gate - 1))
                last_r[q] = False
                last_r[q - (gate - 1)] = False

    out = transpile(out, optimization_level=opt_level)
    return out

def find_allowed_gates(num_qubits, reps, partitions):
    """List the CNOT indices allowed by a qubit partition.

    CNOTs are indexed over the upper-triangular qubit pairs, repeated for each of ``reps`` layers.
    A CNOT is allowed if both qubits lie in the same part of ``partitions``.

    Args:
        num_qubits: Number of qubits.
        reps: Number of entangling layers.
        partitions: List of lists of qubit indices.
    """
    import numpy as np

    tri_inds = np.triu_indices(num_qubits, k=1)
    num_gates = len(tri_inds[0])
    
    allowed_gates = []
    for i in range(reps):
        for j, (q1, q2) in enumerate(zip(*tri_inds)):
            ind1 = [k for k, el in enumerate(partitions) if q1 in el]
            ind2 = [k for k, el in enumerate(partitions) if q2 in el]
            if ind1[0] == ind2[0]:
                allowed_gates.append(j + i * num_gates)
    return allowed_gates

def partition_string_to_list(p_string):
    """Convert a partition string such as ``'012-34'`` to ``[[0, 1, 2], [3, 4]]``."""
    p_list = p_string.split('-')
    out = list(map(lambda p: [int(s) for s in p], p_list))
    return out

def partition_list_to_string(p_list):
    """Convert a partition such as ``[[0, 1, 2], [3, 4]]`` to ``'012-34'``."""
    out = list(map(lambda p: ''.join(map(lambda i: str(i), p)), p_list))
    out = '-'.join(out)
    return out

def build_circuit_ent(ansatz_options, gates, simplify=False):
    """Build a greedy-entangling ansatz: RY layers separated by selected CNOT layers.

    Args:
        ansatz_options: Needs ``num_qubits``, ``reps`` (number of entangling layers) and
            ``constructive`` (True: ``gates`` lists the CNOTs to include; False: the CNOTs to omit).
        gates: CNOT indices over the upper-triangular qubit pairs, offset by layer.
        simplify: Skip an RY on a qubit whose previous gate was already an RY.

    Returns:
        QuantumCircuit.
    """
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter

    num_qubits = ansatz_options['num_qubits']
    reps = ansatz_options['reps']
    constructive = ansatz_options['constructive']
    ansatz = QuantumCircuit(num_qubits)

    tri_inds = np.triu_indices(num_qubits, k=1)
    num_gates = len(tri_inds[0])
    
    p = 0
    prev_rots = []
    for i in range(reps):
        for q in range(num_qubits):
            if (not simplify) or ((simplify) and (q not in prev_rots)):
                theta = Parameter(f'$x_{{{p}}}$')
                p += 1
                # ansatz.rz(theta, q)
                ansatz.ry(theta, q)
                prev_rots.append(q)
        ansatz.barrier(range(num_qubits))
        for j, (q1, q2) in enumerate(zip(*tri_inds)):
            if constructive:
                if j + i * num_gates in gates:
                    ansatz.cx(q1, q2)
                    if q1 in prev_rots:
                        prev_rots.remove(q1)
                    if q2 in prev_rots:
                        prev_rots.remove(q2)
            else: 
                if j + i * num_gates not in gates:
                    ansatz.cx(q1, q2)
                    if q1 in prev_rots:
                        prev_rots.remove(q1)
                    if q2 in prev_rots:
                        prev_rots.remove(q2)
        ansatz.barrier(range(num_qubits))
    for q in range(num_qubits):
        if (not simplify) or ((simplify) and (q not in prev_rots)):
            theta = Parameter(f'$x_{{{p}}}$')
            # theta = Parameter(f'')
            p += 1
            ansatz.ry(theta, q)
            prev_rots.append(q)
    return ansatz

def build_circuit(c, opt_level=2, add_h=False, add_rs=False):
    import numpy as np
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Parameter

    n_qubits = c.shape[0]
    depth = c.shape[1]
    out = QuantumCircuit(n_qubits)
    n_params = np.count_nonzero(c == 1) + (n_qubits if add_rs else 0)
    if n_params == 0:
        theta = Parameter('X[0]')
        out.ry(theta, 0)
    thetas = [Parameter(f'X[{i}]') for i in range(n_params)]
    p = 0

    last_r = [False for i in range(n_qubits)]
    if add_h:
        for i in range(n_qubits):
            out.h(i)
    if add_rs:
        for i in range(n_qubits):
            out.ry(thetas[p], i)
            last_r[i] = True
            p += 1
    
    
    # s = np.array(list(map(lambda x: list(x), c)))
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
            elif gate > 1:
                out.cx(q, q - (gate - 1))
                last_r[q] = False
                last_r[q - (gate - 1)] = False

    out = transpile(out, optimization_level=opt_level)
    return out

def simplify_circuit(c):
    import numpy as np
    n_qubits = c.shape[0]
    depth = c.shape[1]
    c2 = np.copy(c)
    for q in range(n_qubits):
        for g in range(depth - 1, 0, -1):
            prev_gate = g - 1
            while c2[q, prev_gate] == 0 and prev_gate > 1:
                prev_gate -= 1
            if c2[q, g] == c2[q, prev_gate]:
                if c2[q, g] == 2:
                    c2[q, g] = 0
                if c2[q, g] == 1:
                    c2[q, g] = 0
                    c2[q, prev_gate] = 0
            

    return c2
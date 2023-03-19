def block_divider(l):
    if len(l) == 2:
        return [f'{l[0]}{l[1]}']

    if len(l) == 3:
        out = set([f'{l[0]}{l[1]}{l[2]}', f'{l[0]},{l[1]},{l[2]}'])
        for i in range(len(l)):
            a = block_divider(l[:i] + l[i+1:])
            for b in a:
                b += f',{l[i]}'
                bs = b.split(',')
                bs.sort()
                for s in bs:
                    s = sorted(s)
                b = ','.join(bs)
            # print(a)
                out.add(b)
            # print(out)
        return out
    
    out = set([])
    for i in range(len(l)):
        a = block_divider(l[:i] + l[i+1:])
        for b in a:
            b += f',{l[i]}'
            bs = b.split(',')
            bs.sort()
            for s in bs:
                s = sorted(s)
            b = ','.join(bs)
            out.add(b)
        if i < len(l) - 1:
            for j in range(i+1, len(l)):
                a = block_divider(l[:i] + l[i+1:j] + l[j+1:])
                for b in a:
                    b += f',{l[i]}{l[j]}'
                    bs = b.split(',')
                    bs.sort()
                    for s in bs:
                        s = sorted(s)
                    b = ','.join(bs)
                    out.add(b)
                if j < len(l) - 1:
                    for k in range(j+1, len(l)):
                        a = block_divider(l[:i] + l[i+1:j] + l[j+1:k] + l[k+1:])
                        for b in a:
                            b += f',{l[i]}{l[j]}{l[k]}'
                            bs = b.split(',')
                            bs.sort()
                            for s in bs:
                                s = sorted(s)
                            b = ','.join(bs)
                            out.add(b)
    return out

def build_circuit_from_s(s):
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    
    num_qubits = int(sorted(s)[-1]) + 1
    ansatz = QuantumCircuit(num_qubits)

    p = 0
    prev_rots = []
    for q in range(num_qubits):
        theta = Parameter(f'$x_{{{p}}}$')
        p += 1
        ansatz.ry(theta, q)
        prev_rots.append(q)

    blocks = s.split(',')
    for b in blocks:
        if len(b) == 2:
            q1, q2 = int(b[0]), int(b[1])
            ansatz.cnot(q1, q2)
            if q1 in prev_rots:
                prev_rots.remove(q1)
            if q2 in prev_rots:
                prev_rots.remove(q2)
        if len(b) == 3:
            q1, q2, q3 = int(b[0]), int(b[1]), int(b[2])
            ansatz.cnot(q1, q3)
            ansatz.cnot(q1, q2)
            ansatz.cnot(q2, q3)
            if q1 in prev_rots:
                prev_rots.remove(q1)
            if q2 in prev_rots:
                prev_rots.remove(q2)
            if q3 in prev_rots:
                prev_rots.remove(q3)
    
    for q in range(num_qubits):
        if q not in prev_rots:
            theta = Parameter(f'$x_{{{p}}}$')
            p += 1
            ansatz.ry(theta, q)

    return ansatz
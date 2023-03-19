import numpy as np

base_dir = '/arc/project/st-rkrems-1/kasra/dvr_vqe/2d/'
scratch_dir = '/scratch/st-rkrems-1/kasra/dvr_vqe/'

hartree = 219474.64
da_to_au = 1822.888486209
au_to_angs = 0.529177249

def print_log(s, file, end='\n', overwrite=False):
   s = str(s)
   if not overwrite:
      print(s, end=end)
   else:
      print('\r' + s, end=end)
   if file is not None:
      if not overwrite:
         f = open(file, 'a', encoding="utf-8")
         f.write(s)
         f.write(end)
      else:
         f = open(file, 'r', encoding="utf-8")
         lines = f.readlines()[:-1]
         f.close()
         lines.append(s + end)
         f = open(file, 'w', encoding="utf-8")
         f.writelines(lines)

      f.close()

def pauli_decompose(h):
    from qiskit.opflow import X, Y, Z, I
    import numpy as np
    
    pauli_list = [X, Y, Z, I]
    num_qubits = np.log2(h.shape[0])
    k = 1
    while k < num_qubits:
        new_list = []
        for p in pauli_list:
            new_list.append(p ^ X)
            new_list.append(p ^ Y)
            new_list.append(p ^ Z)
            new_list.append(p ^ I)
        pauli_list = new_list
        k += 1

    coefs = np.zeros(len(pauli_list))
    for i, p in enumerate(pauli_list):
        if np.all(np.imag(p.to_matrix()) == 0):
            coefs[i] = np.trace(np.dot(p.to_matrix(), h)) / 2**num_qubits
        else:
            coefs[i] = 0

    out = None
    for i in range(coefs.shape[0]):
        if coefs[i] != 0:
            if out is None:
                out = coefs[i] * pauli_list[i]
            else:
                out += coefs[i] * pauli_list[i]

    return out
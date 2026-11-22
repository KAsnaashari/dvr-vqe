"""Physical constants, logging and Pauli decomposition shared by the DVR and VQE code.

Constants: ``hartree`` (cm^-1 per hartree), ``da_to_au`` (electron masses per dalton),
``au_to_angs`` (angstrom per bohr). ``base_dir``, ``scratch_dir`` and ``results_dir``
come from :mod:`lib.paths`.
"""

import numpy as np

from .paths import get_data_dir, get_scratch_dir, get_results_dir

# Set through DVR_VQE_DATA_DIR / DVR_VQE_SCRATCH_DIR / DVR_VQE_RESULTS_DIR (see lib/paths.py).
base_dir = get_data_dir()
scratch_dir = get_scratch_dir()
results_dir = get_results_dir()

hartree = 219474.64
da_to_au = 1822.888486209
au_to_angs = 0.529177249

def print_log(s, file, end='\n', overwrite=False):
   """Print ``s`` and append it to a log file.

   Args:
       s: Object to log (converted with ``str``).
       file: Log file path, or ``None`` to only print.
       end: Line terminator.
       overwrite: If True, replace the last line (on screen with a carriage return and in the
           file), used for progress counters.
   """
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

def pauli_decompose(h, tol=0):
    """Decompose a real symmetric matrix into Pauli strings.

    Args:
        h: Square matrix of size 2**n.
        tol: Coefficients with absolute value below ``tol`` are dropped.

    Returns:
        SparsePauliOp: the operator sum_P c_P P with c_P = Tr(P h) / 2**n. Pauli strings with an
        imaginary matrix (odd number of Y) are skipped, which is exact for real symmetric ``h``.
    """
    from qiskit.quantum_info import SparsePauliOp
    import numpy as np

    pauli_list = ['X', 'Y', 'Z', 'I']
    num_qubits = np.log2(h.shape[0])
    k = 1
    while k < num_qubits:
        new_list = []
        for p in pauli_list:
            new_list.append(p + 'X')
            new_list.append(p + 'Y')
            new_list.append(p + 'Z')
            new_list.append(p + 'I')
        pauli_list = new_list
        k += 1

    coefs = np.zeros(len(pauli_list), dtype=complex)
    for i, p in enumerate(pauli_list):
        op = SparsePauliOp.from_list([(p, 1)])
        if np.all(np.imag(op.to_matrix()) == 0):
            coefs[i] = np.trace(np.dot(op.to_matrix(), h)) / 2**num_qubits
        else:
            coefs[i] = 0
        if np.abs(coefs[i]) < tol:
            coefs[i] = 0

    op_list = []
    for i in range(coefs.shape[0]):
        if coefs[i] != 0:
            op_list.append((pauli_list[i], coefs[i]))

    out = SparsePauliOp.from_list(op_list)
    return out
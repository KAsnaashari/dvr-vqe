"""Two-dimensional DVR for atom-diatom complexes in Jacobi coordinates (ArHCl, MgNH).

The diatom is a rigid rotor of length ``r_e``. Coordinates are mass-scaled with
factor ``S`` and three-body reduced mass ``mu``. The radial basis is Sturmian
(generalised Laguerre) functions and the angular basis is associated Legendre functions;
both are converted to a DVR by diagonalising the coordinate matrix. Masses are in electron
masses, lengths in bohr unless stated, Hamiltonians in hartree. ``arhcl_params`` and
``mgnh_params`` hold ``name``, total mass ``M``, ``mu``, ``S`` and the scaled rotor
length ``r_e``.
"""

from .utils import *

m_ar = 39.9623831238 * da_to_au
m_h = 1.007825031898 * da_to_au
m_cl = 34.96885269 * da_to_au

M_arhcl = m_ar + m_h + m_cl
mu_arhcl = np.sqrt(m_ar * m_h * m_cl / M_arhcl)
S_arhcl = np.power(M_arhcl * m_h * m_cl / (m_ar * (m_h + m_cl)**2), 0.25)
r_e_hcl = 1.2908 / au_to_angs

arhcl_params = {
    'name': 'arhcl',
    'M': M_arhcl,
    'mu': mu_arhcl,
    'S': S_arhcl,
    'r_e': r_e_hcl * S_arhcl
}

m_mg = 23.985041689 * da_to_au
m_n = 14.003074004251 * da_to_au

M_mgnh = m_mg + m_n + m_h
mu_mgnh = np.sqrt(m_mg * m_n * m_h / M_mgnh)
S_mgnh = np.power(M_mgnh * m_n * m_h / (m_mg * (m_n + m_h)**2), 0.25)
r_e_nh = 1.0367 / au_to_angs

mgnh_params = {
    'name': 'mgnh',
    'M': M_mgnh,
    'mu': mu_mgnh,
    'S': S_mgnh,
    'r_e': r_e_nh * S_mgnh
}


def get_FBR_R(N_R, l):
    """Matrix of the radial coordinate in the Sturmian basis.

    Args:
        N_R: Number of Sturmian basis functions.
        l: Sturmian angular parameter (``dvr_options['l']``).

    Returns:
        (N_R, N_R) tridiagonal matrix; its eigenvalues are the radial DVR points in units of
        bohr / S.
    """
    import numpy as np

    offset_R = l + 1
    rows, cols = np.indices((N_R, N_R))
    delta_R = np.zeros((N_R, N_R), dtype=float)

    # delta_R += 2 * np.diag(np.arange(N_R) + offset_R)

    # rows_1 = np.diag(rows, k=1)
    # cols_1 = np.diag(cols, k=1)
    # rows_2 = np.diag(rows, k=-1)
    # cols_2 = np.diag(cols, k=-1)
    # delta_R[rows_1, cols_1] = -np.sqrt((cols_1 + offset_R + l) * (cols_1 + offset_R - l - 1))
    # delta_R[rows_2, cols_2] = -np.sqrt((cols_2 + offset_R - l) * (cols_2 + offset_R + l + 1))
    for i in range(N_R):
        for j in range(N_R):
            if i == j:
                delta_R[i, j] = 2 * (i + offset_R)
            elif i == j - 1:
                delta_R[i, j] = -np.sqrt((j + offset_R + l) * (j + offset_R - l - 1))
            elif i == j + 1:
                delta_R[i, j] = -np.sqrt((j + offset_R - l) * (j + offset_R + l + 1))

    # print(delta_R, delta_R2)
    return delta_R# / 2

def get_DVR_R(N_R, l, return_T=True):
    """Radial DVR points and transformation matrix.

    Args:
        N_R: Number of basis functions.
        l: Sturmian parameter.
        return_T: Also return the eigenvector matrix.

    Returns:
        R (radial points, bohr / S) and, if ``return_T``, the FBR-to-DVR transformation T_R.
    """
    import numpy as np

    delta_R = get_FBR_R(N_R, l=l)
    R, eigvecs = np.linalg.eigh(delta_R)
    T_R = np.stack(eigvecs)
    if return_T:
        return R, T_R
    return R

def get_FBR_X(N_theta, K):
    """Matrix of x = cos(theta) in the associated Legendre basis P_j^K, j = K, ..., K + N_theta - 1.

    Args:
        N_theta: Number of angular basis functions.
        K: Projection quantum number.

    Returns:
        (N_theta, N_theta) tridiagonal matrix.
    """
    import numpy as np

    offset_theta = K
    rows, cols = np.indices((N_theta, N_theta))
    delta_thetaK = np.zeros((N_theta, N_theta), dtype=float)
    # print(delta_thetaK)

    # rows_1 = np.diag(rows, k=1)
    # cols_1 = np.diag(cols, k=1)
    # rows_2 = np.diag(rows, k=-1)
    # cols_2 = np.diag(cols, k=-1)
    # delta_thetaK[rows_1, cols_1] = -np.sqrt((cols_1 + offset_theta - K) * (cols_1 + offset_theta + K) / ((2 * (cols_1 + offset_theta) + 1) * (2 * (cols_1 + offset_theta) - 1)))
    # delta_thetaK[rows_2, cols_2] = -np.sqrt((cols_2 + offset_theta - K + 1) * (cols_2 + offset_theta + K + 1) / ((2 * (cols_2 + offset_theta) + 1) * (2 * (cols_2 + offset_theta) + 3)))
    for i in range(N_theta):
        for j in range(N_theta):
            if i == j + 1:
                delta_thetaK[i, j] = np.sqrt((j + offset_theta + K + 1) * (j + offset_theta - K + 1) / ((2 * (j + offset_theta) + 1) * (2 * (j + offset_theta) + 3)))
            elif i == j - 1:
                delta_thetaK[i, j] = np.sqrt((j + offset_theta + K) * (j + offset_theta - K) / ((2 * (j + offset_theta) + 1) * (2 * (j + offset_theta) - 1)))

    return delta_thetaK

def get_DVR_X(N_theta, K, return_T=True):
    """Angular DVR points x = cos(theta) and transformation matrix for a given K.

    Args:
        N_theta: Number of angular basis functions.
        K: Projection quantum number.
        return_T: Also return the eigenvector matrix.

    Returns:
        X (cos(theta) points) and, if ``return_T``, the FBR-to-DVR transformation T_thetaK.
    """
    import numpy as np

    delta_thetaK = get_FBR_X(N_theta, K)
    X, eigvecs = np.linalg.eigh(delta_thetaK)
    T_thetaK = np.stack(eigvecs)
    if return_T:
        return X, T_thetaK
    return X

def get_sturmian_int(i1, i2, l, p):
    """Analytic overlap-type integral of two Sturmian functions with an extra power p of r.

    Purpose inferred from the code; please check.

    Args:
        i1, i2: Basis indices (starting at l + 1).
        l: Sturmian parameter.
        p: Power of r in the integrand.

    Returns:
        Value of the integral (float).
    """
    import numpy as np

    out = (-1)**(i1 + i2 + 2*l) * np.math.factorial(2 * l + p + 2)
    out *= np.sqrt(np.math.factorial(i1 - l - 1) * np.math.factorial(i2 - l - 1) / (np.math.factorial(i1 + l) * np.math.factorial(i2 + l)))
    
    s = 0
    for t in range(min(i1 - l - 1, i2 - l - 1) + 1):
        if p + 1 >= 0:
            a = np.math.comb(p + 1, i1 - l - t - 1)
            b = np.math.comb(p + 1, i2 - l - t - 1)
        else:
            a = (-1)**(i1 - l - t - 1) * np.math.comb(i1 - l - t - p - 3, i1 - l - t - 1)
            b = (-1)**(i2 - l - t - 1) * np.math.comb(i2 - l - t - p - 3, i2 - l - t - 1)
        c = np.math.comb(2 * l + p + 2 + t, t)
        s += a * b * c
        
    return out * s

def get_orthonormal_sturmian_int(i1, i2, l, p):
    """Integral for the orthonormal Sturmian basis: :func:`get_sturmian_int` with p - 1.

    Purpose inferred from the code; please check.
    """
    # scale = np.abs(p) + 1
    # if scale == 0:
    #     scale = 1
    return get_sturmian_int(i1, i2, l, p - 1)

def get_D_R_FBR(N_R, l, mol_params=arhcl_params):
    """Radial kinetic-energy matrix in the Sturmian basis.

    Args:
        N_R: Number of basis functions.
        l: Sturmian parameter.
        mol_params: Molecule parameters; uses ``mu``.

    Returns:
        (N_R, N_R) matrix (hartree, for R in units of bohr / S).
    """
    import numpy as np

    offset_R = l + 1
    # print(offset_R)
    def get_D_R_FBR_ij(i, j):
        # print(i, j)
        out = (j - l*(l+1) - 3/4) * get_orthonormal_sturmian_int(i, j, l, -2)
        # print(out)
        out += (j - 1/2) * get_orthonormal_sturmian_int(i, j, l, -1)
        # print(out)
        if i == j:
            out -= 1/4
            # pass
        # print(out)
        if j > (l + 1):
            out -= np.sqrt(j*(j-1) - l*(l+1)) * get_orthonormal_sturmian_int(i, j-1, l, -2)
        # print(out)
        out *= 1 / (2 * mol_params['mu'])
        # print('----------')
        return out
    
    D_R = np.zeros((N_R, N_R), dtype=float)
    for i in range(N_R):
        for j in range(N_R):
            D_R[i, j] = get_D_R_FBR_ij(i + offset_R, j + offset_R)
    return D_R

def get_D_R_DVR(N_R, l, T_R=None, mol_params=arhcl_params):
    """Radial kinetic-energy matrix transformed to the DVR.

    Args:
        N_R, l: As in :func:`get_D_R_FBR`.
        T_R: FBR-to-DVR transformation (computed if None).
        mol_params: Molecule parameters; uses ``mu``.

    Returns:
        (N_R, N_R) matrix (hartree).
    """
    import numpy as np

    if T_R is None:
        _, T_R = get_DVR_R(N_R, l=l, return_T=True)
    
    D_R_FBR = get_D_R_FBR(N_R, l=l, mol_params=mol_params)
    # print(D_R_FBR)
    return np.matmul(T_R.T, np.matmul(D_R_FBR, T_R))

def get_D_thetaK_FBR(N_theta, K, mol_params=arhcl_params):
    """Diagonal angular factor j(j+1) / (2 mu) in the associated Legendre basis.

    Multiplied by (1/R^2 + 1/r_e^2) in :func:`get_ham_DVR` to give the rotational kinetic energy.

    Args:
        N_theta: Number of basis functions.
        K: Projection quantum number; j runs from K to K + N_theta - 1.
        mol_params: Molecule parameters; uses ``mu``.

    Returns:
        (N_theta, N_theta) diagonal matrix (hartree bohr^2).
    """
    import numpy as np

    offset_theta = K
    js = np.arange(N_theta) + offset_theta
    D_thetaK = np.diag(js * (js + 1)) / (2 * mol_params['mu'])
    return D_thetaK

def get_D_thetaK_DVR(N_theta, K, T_thetaK=None):
    """Angular factor of :func:`get_D_thetaK_FBR` transformed to the DVR.

    Args:
        N_theta, K: As in :func:`get_D_thetaK_FBR`.
        T_thetaK: FBR-to-DVR transformation (computed if None).

    Returns:
        (N_theta, N_theta) matrix.
    """
    import numpy as np

    if T_thetaK is None:
        _, T_thetaK = get_DVR_X(N_theta, K=K, return_T=True)
    
    D_thetaK_FBR = get_D_thetaK_FBR(N_theta, K=K)
    return np.matmul(T_thetaK.T, np.matmul(D_thetaK_FBR, T_thetaK))

def get_lambda_plus(J, K):
    """Coriolis factor sqrt(J(J+1) - K(K+1))."""
    import numpy as np
    return np.sqrt(J * (J + 1) - K * (K + 1))

def get_lambda_neg(J, K):
    """Coriolis factor sqrt(J(J+1) - K(K-1))."""
    import numpy as np
    return np.sqrt(J * (J + 1) - K * (K - 1))

def get_B_plus(K, T_thetaK):
    """Matrix of lambda_+(j, K) in the angular DVR for a given K (Coriolis coupling K -> K+1).

    Args:
        K: Projection quantum number.
        T_thetaK: Angular FBR-to-DVR transformation for this K.
    """
    js = np.arange(T_thetaK.shape[0]) + K
    lam = np.diag(get_lambda_plus(js, K))
    return np.matmul(T_thetaK.T, np.matmul(lam, T_thetaK))

def get_B_neg(K, T_thetaK):
    """Matrix of lambda_-(j, K) in the angular DVR for a given K (Coriolis coupling K -> K-1).

    Args:
        K: Projection quantum number.
        T_thetaK: Angular FBR-to-DVR transformation for this K.
    """
    js = np.arange(T_thetaK.shape[0]) + K
    lam = np.diag(get_lambda_neg(js, K))
    return np.matmul(T_thetaK.T, np.matmul(lam, T_thetaK))

def get_DVR_Rtheta(dvr_options, mol_params=arhcl_params, return_T=False):
    """Radial and angular DVR grids for the 2D Jacobi DVR.

    Args:
        dvr_options: dict with ``N_R``, ``l``, ``N_theta``, ``K_max``, ``r_min`` and ``r_max``.
            The radial points (converted to angstrom) are shifted by ``r_min`` (angstrom), and
            points above ``r_max`` (angstrom; None for no limit) are dropped.
        mol_params: Molecule parameters; uses ``S``.
        return_T: Also return the transformation matrices.

    Returns:
        (Rs_DVR, Xs_K): radial points (bohr / S) and a list of cos(theta) grids, one per
        K = 0..K_max. With ``return_T``: ((Rs_DVR, T_R), (Xs_K, T_thetaK)).
    """
    Rs_S, T_R = get_DVR_R(dvr_options['N_R'], l=dvr_options['l'])
    Xs_K = []
    T_thetaK = []
    for K in range(dvr_options['K_max'] + 1):
        Xs, T_theta = get_DVR_X(dvr_options['N_theta'], K)
        Xs_K.append(Xs)
        T_thetaK.append(T_theta)
    Rs = Rs_S * mol_params['S']
    Rs_angs = Rs * au_to_angs

    if dvr_options['r_max'] is None:
        Rs_DVR = (Rs_angs + dvr_options['r_min']) / au_to_angs / mol_params['S']
    else:
        Rs_lim = Rs_angs[Rs_angs + dvr_options['r_min'] <= dvr_options['r_max']] + dvr_options['r_min']
        Rs_lim /= au_to_angs
        Rs_DVR = Rs_lim / mol_params['S']
    
    if return_T:
        return (Rs_DVR, T_R), (Xs_K, T_thetaK)
    return Rs_DVR, Xs_K

def get_ham_DVR(pot2d, dvr_options, mol_params=arhcl_params, count_thresh=None, v_thresh=None):
    """2D DVR Hamiltonian of an atom-rigid-diatom complex.

    Args:
        pot2d: Potential V(R, cos(theta)) with R in angstrom and V in cm^-1 (from
            :func:`lib.pot_gen.get_pot`).
        dvr_options: dict with ``N_R``, ``N_theta``, ``l``, ``K_max``, ``r_min``, ``r_max`` (see
            :func:`get_DVR_Rtheta`), ``J`` (total angular momentum) and ``trunc`` (0 for the full
            product basis; otherwise keep the lowest ``trunc`` radial eigenvectors per angle).
        mol_params: Molecule parameters (``mu``, ``S``, ``r_e``).
        count_thresh: If given (and ``v_thresh`` is None), keep only the ``count_thresh`` grid
            points with the lowest potential.
        v_thresh: If given, keep only grid points with V < ``v_thresh`` (cm^-1).

    Returns:
        Square Hamiltonian matrix (hartree) over the (K, theta, R) grid.
    """
    (Rs_DVR, T_R), (Xs_K, T_thetaK) = get_DVR_Rtheta(dvr_options, mol_params=mol_params, return_T=True)
    N_R_lim = Rs_DVR.shape[0]

    # K = 0 # temp
    JK = np.min([dvr_options['J'], dvr_options['K_max']]) + 1
    Vs_K = []
    for K in range(JK):
        Rs_grid, Xs_grid = np.meshgrid(Rs_DVR * mol_params['S'] * au_to_angs, Xs_K[K])
        Rs_grid, Xs_grid = Rs_grid.flatten(), Xs_grid.flatten()
        Vs = pot2d(Rs_grid, Xs_grid)
        Vs_K.append(Vs)
    Vs_K = np.array(Vs_K)
    # print(np.any(np.isnan(Vs)))

    D_R = get_D_R_DVR(dvr_options['N_R'], dvr_options['l'], T_R=T_R, mol_params=mol_params)[:N_R_lim, :N_R_lim]
    D_thetaK = []
    for K in range(JK):
        D_theta = get_D_thetaK_DVR(dvr_options['N_theta'], K=K, T_thetaK=T_thetaK[K])
        D_thetaK.append(D_theta)
    D_thetaK = np.array(D_thetaK)
    # print(np.any(np.isnan(D_R)))
    # print(np.any(np.isnan(D_theta)))

    C_s = []
    E_s = []
    if dvr_options['trunc'] == 0:
        C_s = [np.eye(N_R_lim) for i in range(dvr_options['N_theta'])]
    else:
        for j1 in range(dvr_options['N_theta']):
            Vs_beta = Vs[j1 * N_R_lim:(j1 + 1) * N_R_lim]
            D_R_beta = D_R + np.diag(Vs_beta) / hartree

            eigvals, eigvecs = np.linalg.eigh(D_R_beta)

            if dvr_options['trunc'] == 0:
                cutoff = eigvals.shape[0]
            else:
                cutoff = dvr_options['trunc']

            E_s.append(eigvals[:cutoff])
            C_s.append(eigvecs[:, :cutoff])
    C_s = np.array(C_s)

    h_dvr = np.zeros((JK, dvr_options['N_theta'], N_R_lim, JK, dvr_options['N_theta'], N_R_lim), dtype=float)

    for K in range(JK):
        for j1 in range(len(Xs_K[K])):
            h_dvr[K, j1, :, K, j1, :] += D_R
        for i1 in range(N_R_lim):
            h_dvr[K, :, i1, K, :, i1] += D_thetaK[K] * (1 / np.square(Rs_DVR[i1]) + 1 / np.square(mol_params['r_e']))

            coef = 1 / (2 * mol_params['mu'] * np.square(Rs_DVR[i1]))
            if K + 1 < JK:
                lam_plus = get_lambda_plus(dvr_options['J'], K)
                b_plus = get_B_plus(K, T_thetaK[K])
                coef_plus = -np.sqrt(2) if K == 0 else 1
                h_dvr[K + 1, :, i1, K, :, i1] = coef * coef_plus * lam_plus * b_plus
            if K > 0:
                lam_neg = get_lambda_neg(dvr_options['J'], K)
                b_neg = get_B_plus(K, T_thetaK[K])
                coef_neg = -np.sqrt(2) if K == 1 else 1
                h_dvr[K - 1, :, i1, K, :, i1] = coef * coef_neg * lam_neg * b_neg

            for j1 in range(len(Xs_K[K])):
                h_dvr[K, j1, i1, K, j1, i1] += (dvr_options['J'] * (dvr_options['J'] + 1) - 2 * K**2) / (2 * mol_params['mu'] * np.square(Rs_DVR[i1]))
                h_dvr[K, j1, i1, K, j1, i1] += Vs_K[K, i1 + j1 * N_R_lim] / hartree

    # print(np.any(np.isnan(h_dvr)))
    h_dvr2 = np.einsum('ijk,oijplm,lmn->oikpln', C_s, h_dvr, C_s)
    h_dvr2 = h_dvr2.reshape((h_dvr2.shape[0] * h_dvr2.shape[1] * h_dvr2.shape[2], h_dvr2.shape[3] * h_dvr2.shape[4] * h_dvr2.shape[5]))
    # print(np.any(np.isnan(h_dvr2)))

    if (v_thresh is None) and (count_thresh is not None) and (count_thresh < h_dvr2.shape[0]):
        v_thresh = np.sort(Vs)[count_thresh]
    if v_thresh is not None:
        h_dvr2 = h_dvr2[Vs < v_thresh][:, Vs < v_thresh]
    

    return h_dvr2
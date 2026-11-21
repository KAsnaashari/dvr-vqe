"""One-dimensional DVR for diatomic molecules (used for Cr2).

Atomic units throughout: distances in bohr, energies in hartree, masses in electron
masses. ``cr2_params`` holds the name and reduced mass of 52Cr2.
"""

from .utils import da_to_au
m_cr = 51.940509 * da_to_au

cr2_params = {
    'name': 'cr2',
    'mu': m_cr * m_cr / (m_cr + m_cr)
}

def dvr(v, N, xlim, m=1):
    """Colbert-Miller DVR Hamiltonian on a uniform grid over a finite interval.

    Args:
        v: Potential function of x.
        N: Number of grid points.
        xlim: (x_min, x_max), endpoints included.
        m: Mass.

    Returns:
        (N, N) Hamiltonian matrix (atomic units if v is in hartree, x in bohr, m in electron
        masses).
    """
    import numpy as np

    dx = (xlim[1] - xlim[0]) / (N - 1)
    dvr_v = np.diag([v(x) for x in np.linspace(xlim[0], xlim[1], N)])
    dvr_t = np.eye(N) * np.square(np.pi) / (6 * m * np.square(dx))
    for i in range(N):
        for j in range(i):
            # print(i, j)
            # print(np.square(dx * (i - j)))
            dvr_t[i, j] = (-1)**(i - j) / np.square(dx * (i - j)) / m
            dvr_t[j, i] = dvr_t[i, j]
            
    return dvr_t + dvr_v

def dvr2(x, v, m=1):
    """Same as :func:`dvr` but for given grid points ``x`` and potential values ``v``.

    Args:
        x: Uniform grid (array).
        v: Potential values on the grid.
        m: Mass.

    Returns:
        Hamiltonian matrix.
    """
    import numpy as np

    dx = x[1] - x[0]
    N = x.shape[0]
    dvr_v = np.diag(v)
    dvr_t = np.eye(N) * np.square(np.pi) / (6 * m * np.square(dx))
    for i in range(N):
        for j in range(i):
            # print(i, j)
            # print(np.square(dx * (i - j)))
            dvr_t[i, j] = (-1)**(i - j) / np.square(dx * (i - j)) / m
            dvr_t[j, i] = dvr_t[i, j]
            
    return dvr_t + dvr_v

def dvr_0inf(r, v, m=1, start_ind=0):
    """Colbert-Miller radial DVR Hamiltonian for r in (0, inf).

    Args:
        r: Uniform radial grid (bohr); the spacing is taken from r[1] - r[0].
        v: Potential values on the grid (hartree).
        m: Reduced mass (electron masses).
        start_ind: Index of r[0] on the full grid starting at r = 0, so that the (0, inf)
            boundary terms are evaluated correctly for a grid that starts away from the origin.

    Returns:
        Hamiltonian matrix (hartree).
    """
    import numpy as np

    dr = r[1] - r[0]
    N = r.shape[0]
    dvr = np.zeros((r.shape[0], r.shape[0]), dtype=float)
    f = 1 / (2 * m * np.square(dr))
    for i in range(N):
        for j in range(i):
            dvr[i, j] = (-1)**(i - j) * f * (2 / np.square(i - j) - 2 / (np.square(i + j + 2 * start_ind + 2)))
            dvr[j, i] = dvr[i, j]
        dvr[i, i] = v[i] + f * (np.square(np.pi) / 3 - 0.5 / (i + start_ind + 1)**2)
            
    return dvr

def get_dvr_r(dvr_options):
    """Radial grid for the 1D DVR.

    Args:
        dvr_options: dict with ``box_lims`` (r_min, r_max) in bohr, ``dx`` grid spacing in bohr
            and ``count`` number of points.

    Returns:
        Array of the first ``count`` points of ``linspace(0, r_max, int(r_max / dx) + 1)`` that are
        larger than r_min (bohr). The actual spacing is r_max / int(r_max / dx), which equals ``dx``
        only when r_max is a multiple of ``dx``.
    """
    import numpy as np
    box_max = dvr_options['box_lims'][1]
    box_min = dvr_options['box_lims'][0]
    count = int(box_max / dvr_options['dx'])

    r = np.linspace(0, box_max, count + 1)
    r_box = r[r > box_min]
    return r_box[:dvr_options['count']]

def get_ham_DVR(pot1d, dvr_options, mol_params):
    """1D DVR Hamiltonian of a diatomic molecule.

    Args:
        pot1d: Potential function V(r), r in bohr, V in hartree.
        dvr_options: Grid options, see :func:`get_dvr_r`.
        mol_params: dict with reduced mass ``mu`` (electron masses).

    Returns:
        (count, count) Hamiltonian matrix (hartree).
    """
    import numpy as np
    r_box = get_dvr_r(dvr_options)
    N = r_box.shape[0]
    start_ind = int(r_box[0] / dvr_options['dx']) - 1
    v = pot1d(r_box)
    return dvr_0inf(r_box, v, m=mol_params['mu'], start_ind=start_ind)

def dvr_0pi(theta, v, m=1):
    """DVR Hamiltonian for an angle theta in (0, pi).

    Not used by the run scripts. Purpose inferred from the code; please check.

    Args:
        theta: Angular grid (radians).
        v: Potential values on the grid.
        m: Mass (or moment-of-inertia factor).

    Returns:
        Hamiltonian matrix.
    """
    import numpy as np

    N = theta.shape[0]
    dvr_v = np.diag(v)
    dvr_t = np.diag(1 / (4 * m) * ((2 * np.square(N) + 1) / 3 - 1 / (np.sin(theta))))
    for i in range(N):
        for j in range(i):
            dvr_t[i, j] = (-1)**(i - j) / (4 * m) * (np.square(np.sin((theta[i] - theta[j]) / 2)) - np.square(np.sin((theta[i] + theta[j]) / 2)))
            dvr_t[j, i] = dvr_t[i, j]
            
    return dvr_t + dvr_v

def dvr_xy(r, theta, v, m=1, start_ind=0):
    """Product DVR Hamiltonian on an (r, theta) grid.

    Not used by the run scripts. Purpose inferred from the code; please check.

    Args:
        r, theta: Flattened grid coordinates (bohr, radians).
        v: Potential values on the grid.
        m: Mass.
        start_ind: Offset of the radial grid, as in :func:`dvr_0inf`.

    Returns:
        Hamiltonian matrix.
    """
    import numpy as np

    N_r = np.unique(r).shape[0]
    dr = np.abs(np.unique(r)[1] - np.unique(r)[0])
    N_theta = np.unique(theta).shape[0]
    out = np.zeros((N_r * N_theta, N_r * N_theta))

    for i in range(N_r * N_theta):
        for j in range(N_r * N_theta):
            if (r[i] == r[j]) and (theta[i] == theta[j]):
                out[i, j] = out[j, i] = v[i] + 1 / (4 * m) * ((2 * np.square(N_theta) + 1) / 3 - 1 / (np.sin(theta[i]))) + \
                            1 / (2 * m * np.square(dr)) * (np.square(np.pi) / 3 - 0.5 / (i + start_ind + 1)**2)
            elif r[i] == r[j]:
                i1 = theta[i] / np.pi * N_theta
                i2 = theta[j] / np.pi * N_theta
                out[i, j] = (-1)**(i1 - i2) / (4 * m) * (np.square(np.sin((theta[i] - theta[j]) / 2)) - np.square(np.sin((theta[i] + theta[j]) / 2)))
                out[j, i] = out[i, j]
            elif theta[i] == theta[j]:
                i1 = r[i] / dr
                i2 = r[j] / dr
                out[i, j] = (-1)**(i1 - i2) * 1 / (2 * m * np.square(dr)) * (2 / np.square(i1 - i2) - 2 / (np.square(i1 + i2 + 2 * start_ind + 2)))
                out[j, i] = out[i, j]
    return out
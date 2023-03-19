from .utils import da_to_au
m_cr = 51.940509 * da_to_au

cr2_params = {
    'name': 'cr2',
    'mu': m_cr * m_cr / (m_cr + m_cr)
}

def dvr(v, N, xlim, m=1):
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
    import numpy as np
    box_max = dvr_options['box_lims'][1]
    box_min = dvr_options['box_lims'][0]
    count = int(box_max / dvr_options['dx'])

    r = np.linspace(0, box_max, count + 1)
    r_box = r[r > box_min]
    return r_box[:dvr_options['count']]

def get_ham_DVR(pot1d, dvr_options, mol_params):
    import numpy as np
    r_box = get_dvr_r(dvr_options)
    N = r_box.shape[0]
    start_ind = int(r_box[0] / dvr_options['dx']) - 1
    v = pot1d(r_box)
    return dvr_0inf(r_box, v, m=mol_params['mu'], start_ind=start_ind)

def dvr_0pi(theta, v, m=1):
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
"""
Bundle all functions together which investigate interesting properties
"""

import numpy as np
import networkx as nx
import matplotlib.pylab as plt

from utils import save

from tqdm import tqdm


def compute_correlation_matrix(sols):
    """ Compute pairwise node-correlations of solution
    """
    cmat = np.empty((sols.shape[0], sols.shape[0], sols.shape[1]))
    for i, sol in enumerate(sols):
        for j, osol in enumerate(sols):
            cmat[i, j] = np.cos(sol - osol)
    return cmat

def compute_dcm(corr_mat):
    """ Compute dynamic connectivity matrix
    """
    dcm = []
    for thres in [0.9]: #np.linspace(0, 1, 5):
        cur_mat = corr_mat > thres
        dcm.append(cur_mat)
    return dcm[0]

def compute_sync_time(dcm, ts):
    """ Compute time it takes to synchronize from DCM
    """
    sync_time = -np.ones((dcm.shape[0], dcm.shape[0]))
    for t, state in enumerate(dcm.T):
        inds = np.argwhere((state == 1) & (sync_time < 0))
        sync_time[tuple(inds.T)] = ts[t]
    return sync_time

def compute_cluster_num(sols, graph_size, thres=.05):
    """ Compute pairwise node-variances of solution
    """
    series = []
    for t, state in enumerate(sols.T):
        # compute sine sum
        sin_sum = 0
        for i_theta in state:
            for j_theta in state:
                sin = np.sin(i_theta - j_theta)**2
                if sin > thres:
                    sin_sum += sin

        # compute actual value 'c'
        c = graph_size**2 / (graph_size**2 - 2 * sin_sum)

        series.append(c)
    return series

def investigate_laplacian(graph):
    """ Compute Laplacian
    """
    w = nx.laplacian_spectrum(graph)

    pairs = []
    for i, w in enumerate(sorted(w)):
        if abs(w) < 1e-5: continue
        inv_w = 1 / w
        pairs.append((inv_w, i))

    fig = plt.figure()
    plt.scatter(*zip(*pairs))
    plt.xlabel(r'$\frac{1}{\lambda_i}$')
    plt.ylabel(r'rank index')
    save(fig, 'laplacian_spectrum')

def reconstruct_coupling_params(conf, data, verbose=True):
    """ Try to reconstruct A and B from observed data
    """
    # aggregate solution data
    aggr_sols = []
    for sys_conf, all_sols in data:
        for sol in all_sols:
            t_points = np.array(range(len(conf.ts)-1))

            slices = sol.T[t_points]
            slices_nex = sol.T[t_points+1]
            aggr_sols.extend(zip(
                slices, slices_nex,
                [sys_conf.o_vec] * len(t_points),
                conf.ts[t_points]))

    aggr_sols = np.array(aggr_sols)

    sol_dim = aggr_sols.shape[0] * conf.A.shape[0]
    syst_dim = conf.A.shape[0]**2

    # create linear system
    rhs = []
    lhs = []

    encountered_rows = set()
    round_fac = 1

    nondiag_inds = np.where(
        np.arange(conf.A.size) % (conf.A.shape[0]+1) != 0)
    diag_inds = np.where(
        np.arange(conf.A.size) % (conf.A.shape[0]+1) == 0)

    for theta, theta_nex, o_vec, t in tqdm(aggr_sols, nested=not verbose):
        for i in range(conf.A.shape[0]):
            # coefficient matrix
            coeffs_A = np.zeros(conf.A.shape)
            tmp = np.reshape(theta, (len(theta), 1))
            coeffs_A[i,:] = np.sin(tmp.transpose() - tmp)[i,:]

            flat_A = coeffs_A.reshape(conf.A.shape[0]**2)[nondiag_inds]

            coeffs_B = np.zeros(conf.A.shape[0])
            coeffs_B[i] = np.sin(conf.Phi(t) - theta[i])

            row = flat_A.tolist() + coeffs_B.tolist()
            append_check = not tuple(np.round(row, round_fac).tolist()) in encountered_rows

            if append_check:
                encountered_rows.add(tuple(np.round(row, round_fac).tolist()))
                rhs.append(row)

            # solution vector
            theta_dot = (theta_nex[i] - theta[i]) / (conf.skip * conf.dt)
            if append_check:
                lhs.append(theta_dot - o_vec[i])

    rhs = np.array(rhs)
    lhs = np.array(lhs)

    if verbose:
        print('Using', rhs.shape[0], 'out of', sol_dim, 'data points to solve system of', syst_dim, 'variables')

    # solve system
    x = np.linalg.lstsq(rhs, lhs)[0]

    extr = x[:-conf.A.shape[0]]
    for i in diag_inds[0]: extr = np.insert(extr, i, 0)

    rec_a = extr.reshape(conf.A.shape)
    rec_b = x[-conf.A.shape[0]:]

    # show result
    if verbose:
        print('Original A:\n', conf.A)
        print('Reconstructed A:\n', rec_a)
        print()
        print('Original B:', conf.B)
        print('Reconstructed B:', rec_b)

    return rec_a, rec_b

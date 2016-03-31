"""
Bundle all functions together which investigate interesting properties
"""

import numpy as np
import networkx as nx
import matplotlib.pylab as plt

from utils import save


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

def reconstruct_coupling_params(bundle, verbose=True):
    """ Try to reconstruct A and B from observed data
    """
    c = bundle.system_config

    # aggregate solution data
    aggr_sols = []
    for sol in bundle.all_sols:
        t_points = np.array(range(len(bundle.ts)-1))

        slices = sol.T[t_points]
        slices_nex = sol.T[t_points+1]
        aggr_sols.extend(zip(slices, bundle.ts[t_points], slices_nex))

    aggr_sols = np.array(aggr_sols)

    sol_dim = aggr_sols.shape[0] * c.A.shape[0]
    syst_dim = c.A.shape[0]**2 + c.A.shape[0]

    if verbose:
        print('Using', sol_dim, 'data points to solve system of', syst_dim, 'variables')

    # create linear system
    rhs = np.empty((sol_dim, syst_dim))
    lhs = np.empty(sol_dim)

    for t_ind, (theta, t, theta_nex) in enumerate(aggr_sols):
        for i in range(c.A.shape[0]):
            ind = i * aggr_sols.shape[0] + t_ind

            # coefficient matrix
            coeffs_A = np.zeros(c.A.shape)
            for j in range(c.A.shape[1]):
                coeffs_A[i,j] = np.sin(theta[j] - theta[i])

            coeffs_B = np.zeros(c.A.shape[0])
            coeffs_B[i] = np.sin(c.Phi(t) - theta[i])

            row = np.hstack((coeffs_A.reshape(c.A.shape[0]**2), coeffs_B))
            rhs[ind,:] = row

            # solution vector
            theta_dot = (theta_nex[i] - theta[i]) / c.dt
            lhs[ind] = theta_dot - c.o_vec[i]

    # solve system
    x = np.linalg.lstsq(rhs, lhs)[0]
    rec_a = x[:-c.A.shape[0]].reshape(c.A.shape)
    rec_b = x[-c.A.shape[0]:]

    # show result
    if verbose:
        print('Original A:\n', c.A)
        print('Reconstructed A:\n', rec_a)
        print()
        print('Original B:', c.B)
        print('Reconstructed B:', rec_b)

    return rec_a, rec_b

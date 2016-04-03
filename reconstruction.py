"""
Try to reconstruct system parameters from observed solutions
"""

import numpy as np
import networkx as nx

import matplotlib as mpl
import matplotlib.pylab as plt

from tqdm import tqdm, trange

from utils import DictWrapper as DW, save, solve_system
from investigations import reconstruct_coupling_params
from generators import *


def generate_systems(max_size=5, o_size=20):
    """ Generate population of systems
    """
    para_range = range(2, max_size)
    o_range = np.random.uniform(0, 3, o_size)

    systs = []
    for size in para_range:
        systs.append([])

        # setup network
        graph = generate_ring_graph(size)

        dim = len(graph.nodes())
        Bvec = np.random.uniform(0, 5, size=dim)

        for omega in o_range:
            # setup dynamical system
            OMEGA = 3
            system_config = DW({
                'A': nx.to_numpy_matrix(graph),
                'B': Bvec,
                'o_vec': np.ones(dim) * omega,
                'Phi': lambda t: OMEGA * t,
                'OMEGA': OMEGA,
                'dt': 0.01,
                'tmax': 0.1
            })

            systs[-1].append(DW({
                'graph': graph,
                'system_config': system_config
            }))

    return systs, para_range

def process(bundle_pack, reps=10):
    """ Solve system bundle and return data
    """
    # assemble final bundle from pack
    repr_bundle = bundle_pack[0]

    data = []
    for bundle in tqdm(bundle_pack, nested=True):
        all_sols = []
        for _ in range(reps):
            sols, ts = solve_system(bundle.system_config)
            all_sols.append(sols)
        data.append((bundle.system_config, all_sols))

    # reconstruct parameters
    rec_a, rec_b = reconstruct_coupling_params(
        DW({
            'A': repr_bundle.system_config.A,
            'B': repr_bundle.system_config.B,
            'Phi': repr_bundle.system_config.Phi,
            'dt': repr_bundle.system_config.dt,
            'ts': ts
        }), data, verbose=False)

    return DW({
        'A': DW({'orig': repr_bundle.system_config.A, 'rec': rec_a}),
        'B': DW({'orig': repr_bundle.system_config.B, 'rec': rec_b}),
    })

def compute_error(result):
    """ Compute error of reconstruction
    """
    diff_A = abs(result.A.orig - result.A.rec)
    diff_B = abs(result.B.orig - result.B.rec)

    err_A = np.sum(diff_A)/result.A.orig.size
    err_B = np.sum(diff_B)/result.B.orig.size

    return err_A, err_B

def plot_errors(prange, errors_A, errors_B):
    """ Plot error development of parameter reconstruction
    """
    def transform(dat):
        y_mean = []
        y_err = []
        for e in dat:
            y_mean.append(np.mean(e))
            y_err.append(np.std(e))
        return y_mean, y_err

    def plot(data, ax, title):
        y_mean, y_err = transform(data)
        ax.errorbar(prange, y_mean, yerr=y_err, fmt='o')

        ax.set_title(title)
        ax.set_xlabel('network size')
        ax.set_ylabel('reconstruction error')

    fig = plt.figure(figsize=(10,4))
    gs = mpl.gridspec.GridSpec(1, 2)

    plot(errors_A, plt.subplot(gs[0]), 'A')
    plot(errors_B, plt.subplot(gs[1]), 'B')

    plt.tight_layout()
    save(fig, 'reconstruction_error')

def main(reps_per_config=5):
    """ General interface
    """
    systems, prange = generate_systems()

    errors_A, errors_B = [], []
    for bundle_pack in tqdm(systems):
        tmp_A, tmp_B = [], []
        for _ in trange(reps_per_config, nested=True):
            res = process(bundle_pack)
            err_A, err_B = compute_error(res)

            tmp_A.append(err_A)
            tmp_B.append(err_B)
        errors_A.append(tmp_A)
        errors_B.append(tmp_B)

    plot_errors(prange, errors_A, errors_B)


if __name__ == '__main__':
    main()

"""
Try to reconstruct system parameters from observed solutions
"""

import sys

import numpy as np
import pandas as pd
import networkx as nx

import seaborn as sns
import matplotlib as mpl
import matplotlib.pylab as plt

from tqdm import tqdm, trange

from utils import DictWrapper as DW, save, solve_system
from investigations import reconstruct_coupling_params
from plotter import plot_graph
from generators import *


def get_base_config(A, B, omega, OMEGA=3):
    """ Get base of system config
    """
    return DW({
        'A': A,
        'B': B,
        'o_vec': np.ones(A.shape[0]) * omega,
        'Phi': lambda t: OMEGA * t,
        'OMEGA': OMEGA,
        'dt': 0.01,
        'tmax': 10
    })

def generate_systems(prop_step=5, o_size=20):
    """ Generate population of systems
    """
    para_range = np.linspace(0, 1, prop_step)
    o_range = np.random.uniform(0, 3, o_size)

    systs = []
    for p in para_range:
        systs.append([])

        # setup network
        graph = nx.gnp_random_graph(20, p)

        dim = len(graph.nodes())
        Bvec = np.random.uniform(0, 5, size=dim)

        for omega in o_range:
            # setup dynamical system
            system_config = get_base_config(
                nx.to_numpy_matrix(graph), Bvec, omega)

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
            sols, ts = solve_system(bundle.system_config, force_mod=False)
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
    """ Compute relative error of reconstruction
    """
    inds = np.nonzero(result.A.orig)
    rel_err_A = abs(result.A.rec[inds] - result.A.orig[inds]) / result.A.orig[inds]
    rel_err_B = abs(result.B.rec - result.B.orig) / result.B.orig

    return np.mean(rel_err_A), np.mean(rel_err_B)

def plot_errors(df):
    """ Plot error development of parameter reconstruction
    """
    fig = plt.figure()
    sns.boxplot(x='graph_property', y='relative_error', hue='parameter', data=df)
    save(fig, 'reconstruction_error')

def compute_solutions(res):
    """ Compute solution from original and reconstructed parameters
    """
    # configure comparison
    omega = 0.5

    orig_conf = get_base_config(res.A.orig, res.B.orig, omega)
    rec_conf = get_base_config(res.A.rec, res.B.rec, omega)

    init = np.random.uniform(0, 2*np.pi, size=orig_conf.o_vec.shape)

    # solve systems
    orig_sols, ots = solve_system(orig_conf, init=init)
    rec_sols, rts = solve_system(rec_conf, init=init)
    assert (ots == rts).all()

    # aggregate data
    df = pd.DataFrame()
    for orig_slice, rec_slice, t in zip(orig_sols.T, rec_sols.T, rts):
        for i, (orig_val, rec_val) in enumerate(zip(orig_slice, rec_slice)):
            df = df.append([
                {'time': t, 'theta': orig_val, 'oscillator': i, 'source': 'orig'},
                {'time': t, 'theta': rec_val, 'oscillator': i, 'source': 'rec'}
            ], ignore_index=True)

    return df

def plot_reconstruction_result(res):
    """ Plot original and reconstructed graph plus time series
    """
    fig = plt.figure(figsize=(32, 8))
    gs = mpl.gridspec.GridSpec(1, 4)

    # original graph
    orig_ax = plt.subplot(gs[0])
    plot_graph(nx.from_numpy_matrix(res.A.orig), orig_ax)
    orig_ax.set_title('Original graph')

    # time series
    ax = plt.subplot(gs[1:3])
    sns.tsplot(
        time='time', value='theta',
        unit='source', condition='oscillator',
        estimator=np.mean, legend=False,
        data=compute_solutions(res),
        ax=ax)
    ax.set_title(r'$A_{{err}} = {:.2}, B_{{err}} = {:.2}$'.format(*compute_error(res)))

    # reconstructed graph
    rec_ax = plt.subplot(gs[3])
    tmp = res.A.rec
    tmp[abs(tmp) < 1e-1] = 0
    plot_graph(nx.from_numpy_matrix(tmp), rec_ax)
    rec_ax.set_title('Reconstructed graph')

    plt.tight_layout()
    save(fig, 'reconstruction_overview')

def main(reps_per_config=50):
    """ General interface
    """
    systems, prange = generate_systems()

    df = pd.DataFrame(columns=['graph_property', 'parameter', 'relative_error'])
    for i, bundle_pack in enumerate(tqdm(systems)):
        tmp_A, tmp_B = [], []
        for _ in trange(reps_per_config, nested=True):
            res = process(bundle_pack)

            #plot_reconstruction_result(res)
            err_A, err_B = compute_error(res)

            df = df.append([
                {'graph_property': prange[i], 'parameter': 'A', 'relative_error': err_A},
                {'graph_property': prange[i], 'parameter': 'B', 'relative_error': err_B}
            ], ignore_index=True)
        df.to_pickle('df.bak')

    plot_errors(df)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    else:
        plot_errors(pd.read_pickle(sys.argv[1]))

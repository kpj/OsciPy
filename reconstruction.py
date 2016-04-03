"""
Try to reconstruct system parameters from observed solutions
"""

import numpy as np
import pandas as pd
import networkx as nx

import seaborn as sns
import matplotlib.pylab as plt

from tqdm import tqdm, trange

from utils import DictWrapper as DW, save, solve_system
from investigations import reconstruct_coupling_params
from generators import *


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

def plot_errors(df):
    """ Plot error development of parameter reconstruction
    """
    fig = plt.figure()
    sns.boxplot(x='graph_property', y='err_B', data=df)
    save(fig, 'reconstruction_error')

def main(reps_per_config=5):
    """ General interface
    """
    systems, prange = generate_systems()

    df = pd.DataFrame(columns=['graph_property', 'err_A', 'err_B'])
    for i, bundle_pack in enumerate(tqdm(systems)):
        tmp_A, tmp_B = [], []
        for _ in trange(reps_per_config, nested=True):
            res = process(bundle_pack)

            plot_comparison(bundle_pack, res)
            err_A, err_B = compute_error(res)

            df = df.append({
                'graph_property': prange[i],
                'err_A': err_A,
                'err_B': err_B
            }, ignore_index=True)
    df.graph_property = pd.factorize(df.graph_property)[0]

    plot_errors(df)


if __name__ == '__main__':
    main()

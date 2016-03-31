"""
Try to reconstruct system parameters from observed solutions
"""

import numpy as np
import networkx as nx

import matplotlib as mpl
import matplotlib.pylab as plt

from tqdm import tqdm

from utils import DictWrapper as DW, save
from main import simulate_system
from investigations import reconstruct_coupling_params


def generate_systems(size=20):
    """ Generate population of systems
    """
    systs = []
    for p in np.linspace(0, 1, 5):
        # setup network
        graph = nx.gnp_random_graph(size, p)

        # setup dynamical system
        omega = 0.2
        OMEGA = 3
        dim = len(graph.nodes())

        system_config = DW({
            'A': nx.to_numpy_matrix(graph),
            'B': np.random.uniform(0, 5, size=dim),
            'o_vec': np.ones(dim) * omega,
            'Phi': lambda t: OMEGA * t,
            'OMEGA': OMEGA,
            'dt': 0.01,
            'tmax': 1
        })

        systs.append(DW({
            'graph': graph,
            'system_config': system_config
        }))
    return systs

def process(bundle, reps=1):
    """ Solve system bundle and return data
    """
    rec_a, rec_b = reconstruct_coupling_params(
        simulate_system(
            bundle,
            reps=reps, check_laplacian=False,
            nested=True),
        verbose=False)

    return DW({
        'A': DW({'orig': bundle.system_config.A, 'rec': rec_a}),
        'B': DW({'orig': bundle.system_config.B, 'rec': rec_b}),
    })

def compute_error(result):
    """ Compute error of reconstruction
    """
    diff_A = abs(result.A.orig - result.A.rec)
    diff_B = abs(result.B.orig - result.B.rec)

    err_A = np.sum(diff_A)/result.A.orig.size
    err_B = np.sum(diff_B)/result.B.orig.size

    return err_A, err_B

def plot_errors(errors_A, errors_B):
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
        ax.errorbar(range(len(data)), y_mean, yerr=y_err, fmt='o')

        ax.set_title(title)
        ax.set_ylabel('reconstruction error')

    fig = plt.figure(figsize=(10,5))
    gs = mpl.gridspec.GridSpec(1, 2)

    plot(errors_A, plt.subplot(gs[0]), 'A')
    plot(errors_B, plt.subplot(gs[1]), 'B')

    plt.tight_layout()
    save(fig, 'reconstruction_error')

def main(reps=10):
    """ General interface
    """
    systems = generate_systems()

    errors_A, errors_B = [], []
    for bundle in tqdm(systems):
        tmp_A, tmp_B = [], []
        for _ in range(reps):
            res = process(bundle)
            err_A, err_B = compute_error(res)

            tmp_A.append(err_A)
            tmp_B.append(err_B)
        errors_A.append(tmp_A)
        errors_B.append(tmp_B)

    plot_errors(errors_A, errors_B)


if __name__ == '__main__':
    main()

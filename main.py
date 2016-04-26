"""
Investigate how to reconstruct parameters from scarce information
"""

import numpy as np
import networkx as nx

from tqdm import tqdm

from system import System
from reconstruction import Reconstructor
from visualization import show_reconstruction_overview


class DictWrapper(dict):
    """
    Dict with dot-notation access functionality
    """
    def __getattr__(self, attr):
        if not attr in self:
            raise KeyError('{} not in {}'.format(attr, self.keys()))

        return self.get(attr)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def main():
    """
    Main interface
    """
    # generate basis of system
    graph = nx.cycle_graph(4)
    dim = len(graph.nodes())

    orig_A = nx.to_numpy_matrix(graph)
    orig_B = np.random.uniform(0, 5, size=dim)

    print('Original A:\n', orig_A)
    print('Original B:', orig_B)

    omega = 2
    OMEGA_list = np.arange(2.2, 3, 0.2)

    # generate solutions
    data = []
    for OMEGA in tqdm(OMEGA_list):
        syst = System(orig_A, orig_B, omega, OMEGA)
        sols, ts = syst.solve(0.01, 20)

        pdiffs = Reconstructor.extract_phase_differences(sols, ts, syst.Phi)
        #System.plot_solution(syst.Phi(ts), sols, ts)

        data.append(((OMEGA, omega), pdiffs))

    # reconstruct parameters
    recon = Reconstructor(ts, data)
    rec_A, rec_B = recon.reconstruct()

    print('Reconstructed A:\n', rec_A)
    print('Reconstructed B:', rec_B)

    # plot result
    bundle = DictWrapper({
        'orig_A': orig_A,
        'orig_B': orig_B,
        'rec_A': rec_A,
        'rec_B': rec_B
    })
    show_reconstruction_overview(syst, bundle)

if __name__ == '__main__':
    main()

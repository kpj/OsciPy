"""
Investigate how to reconstruct parameters from scarce information
"""

import numpy as np
import networkx as nx

from tqdm import tqdm

from system import System
from reconstruction import Reconstructor


def main():
    """
    Main interface
    """
    # generate basis of system
    graph = nx.cycle_graph(4)
    dim = len(graph.nodes())

    A = nx.to_numpy_matrix(graph)
    B = np.random.uniform(0, 5, size=dim)

    print('Original A:\n', A)
    print('Original B:', B)

    omega = 2
    OMEGA_list = np.arange(2.2, 3, 0.2)

    # generate solutions
    data = []
    for OMEGA in tqdm(OMEGA_list):
        syst = System(A, B, omega, OMEGA)
        sols, ts = syst.solve(0.01, 20)

        pdiffs = Reconstructor.extract_phase_differences(sols, ts, syst.Phi)
        #System.plot_solution(syst.Phi(ts), sols, ts)

        data.append(((OMEGA, omega), pdiffs))

    # reconstruct parameters
    recon = Reconstructor(ts, data)
    rec_A, rec_B = recon.reconstruct()

    print('Reconstructed A:\n', rec_A)
    print('Reconstructed B:', rec_B)

if __name__ == '__main__':
    main()

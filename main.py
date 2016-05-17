"""
Investigate how to reconstruct parameters from scarce information
"""

import sys
from datetime import datetime

import numpy as np
import networkx as nx

from tqdm import tqdm, trange

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

def main(inp):
    """
    Main interface
    """
    if inp is None:
        # generate basis of system
        graph = nx.gnp_random_graph(10, 0.6)
        dim = len(graph.nodes())

        orig_A = nx.to_numpy_matrix(graph)
        orig_B = np.random.uniform(10, 20, size=dim)

        nz = np.nonzero(orig_A)
        orig_A[nz] = np.random.uniform(2, 5, size=len(nz[0]))

        print('Original A:\n', orig_A)
        print('Original B:', orig_B)

        omega = 3
        OMEGA_list = [2.9,3.05,3.1,3.2]#np.arange(3.7, 4.3, 0.05)

        # generate solutions
        data = []
        for OMEGA in tqdm(OMEGA_list):
            runs = []
            for i in trange(dim):
                mask = np.ones(dim, dtype=bool)
                mask[i] = 0
                Bvec = orig_B.copy()
                Bvec[mask] = 0

                syst = System(orig_A, Bvec, omega, OMEGA)
                sols, ts = syst.solve(0.01, 100)

                pdiffs = Reconstructor.extract_phase_differences(sols, ts, syst.Phi)
                #print(pdiffs)
                #System.plot_solution(syst.Phi(ts), sols, ts)

                runs.append(pdiffs)

            data.append(((OMEGA, omega), runs))

        # cache results
        fname = '{}_{}'.format(datetime.now().strftime('%Y%m%d%H%M%S'), dim)
        np.save('cache/{}'.format(fname), {
            'data': data,
            'ts': ts,
            'orig_A': orig_A,
            'orig_B': orig_B
        })
    else:
        data, ts = inp.item()['data'], inp.item()['ts']
        orig_A, orig_B = inp.item()['orig_A'], inp.item()['orig_B']

        print('Original A:\n', orig_A)
        print('Original B:', orig_B)

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
    show_reconstruction_overview(bundle, verbose=True)

if __name__ == '__main__':
    inp = None

    if len(sys.argv) == 2:
        inp = np.load(sys.argv[1])

    main(inp)

"""
Investigate how to reconstruct parameters from scarce information
"""

import numpy as np
import pandas as pd
import networkx as nx

from scipy.integrate import odeint

import seaborn as sns
import matplotlib.pylab as plt

from tqdm import tqdm


def find_tpi_crossings(series):
    """
    Find indices where time series wraps from 2pi back to 0.

    Arguments:
        series
            List of scalar values
    """
    idx = []
    for i in range(len(series)-1):
        if series[i] > 3/2*np.pi and series[i+1] < 1/2*np.pi:
            idx.append(i)
    return idx

class System(object):
    """
    Represent system of oscillators
    """
    def __init__(self, A, B, omega, OMEGA):
        """
        Create a system of Kuramoto oscillators.

        Arguments:
            A
                Interal coupling matrix of the system
            B
                Coupling of the external force to each oscillator
            omega
                Internal oscillator frequency
            OMEGA
                External driving force frequency
        """
        self._A = A
        self._B = B

        self._OMEGA = OMEGA
        self._omegas = omega * np.ones(A.shape[0])

    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B

    @property
    def OMEGA(self):
        return self._OMEGA

    @property
    def omegas(self):
        return self._omegas

    @property
    def Phi(self):
        return lambda t: self.OMEGA * t

    def _get_equation(self):
        """
        Generate ODE system
        """
        def func(theta, t=0):
            ode = []
            for i, omega in enumerate(self.omegas):
                ode.append(
                    omega \
                    + np.sum([self.A[i,j] * np.sin(theta[j] - theta[i])
                        for j in range(len(self.omegas))]) \
                    + self.B[i] * np.sin(self.Phi(t) - theta[i])
                )
            return np.array(ode)
        return func

    def solve(self, dt, T, init=None):
        """
        Solve system of ODEs.

        Arguments:
            dt
                Step size of the simulation
            T
                Maximal time to run the simulation to
            init
                Initial condition of the system
        """
        ts = np.arange(0, T, dt)
        if init is None:
            init = np.random.uniform(0, 2*np.pi, size=self.omegas.shape)

        sol = odeint(self._get_equation(), init, ts).T
        return sol, ts

    @staticmethod
    def extract_phase_differences(sols, ts, driver):
        """
        Extract phase difference of each oscillator to driving force in locked state.

        Arguments:
            sols
                List of system solutions
            ts
                List of time points of the simulation
            driver
                Function of driver of system
        """
        sols %= 2*np.pi

        # find driver's phase
        driver_theta = driver(ts) % (2*np.pi)
        driver_idx = find_tpi_crossings(driver_theta)

        # compute phase differences
        driver_ref = driver_theta[-1]

        pdiffs = []
        for sol in sols:
            pdiffs.append(driver_ref - sol[-1])

        return pdiffs

    @staticmethod
    def plot_solution(driver_sol, sols, ts):
        """
        Plot solution of oscillator system.

        Arguments:
            driver_sol
                Solution of external driver
            sols
                List of system solutions
            ts
                List of time points of the simulation
        """
        # confine to circular region
        sols %= 2*np.pi
        driver_sol %= 2*np.pi

        # convert to DataFrame
        df = pd.DataFrame.from_dict([
            {
                'theta': theta,
                'time': ts[i],
                'oscillator': osci,
                'source': 'raw'
            }
            for osci, sol in enumerate(sols)
                for i, theta in enumerate(sol)
        ])

        df = df.append(pd.DataFrame.from_dict([
            {
                'theta': theta,
                'time': ts[i],
                'oscillator': 'driver',
                'source': 'raw'
            }
            for i, theta in enumerate(driver_sol)
        ]))

        # plot result
        plt.figure()

        sns.tsplot(
            time='time', value='theta',
            condition='oscillator', unit='source',
            data=df)

        plt.show()

class Reconstructor(object):
    """
    Reconstruct system parameters from evolution observations
    """
    def __init__(self, ts, data):
        """
        Initialize reconstruction process.

        Arguments
            ts
                List of time points of the simulation
            data
                List of (OMEGA, [phase differences]) pairs
        """
        self._ts = ts
        self._data = data

        pdiffs = data[0][1]
        self._graph_shape = (len(pdiffs), len(pdiffs))

    @property
    def ts(self):
        return self._ts

    @property
    def data(self):
        return self._data

    def _mat_to_flat(self, mat):
        """
        Convert matrix to flat version without diagonal entries.

        Arguments:
            mat
                Matrix to be flattened
        """
        # find indices of entries which are not on the diagonal
        gsize = self._graph_shape[0]**2
        nondiag_inds = np.where(
            np.arange(gsize) % (self._graph_shape[0]+1) != 0)

        return mat.reshape(gsize)[nondiag_inds]

    def _flat_to_mat(self, flat, repl=0):
        """
        Convert flat version to matrix and reinsert diagonal elements.

        Arguments:
            flat
                Flat list to be converted
            repl
                Scalar to be inserted into diagonal
        """
        # find indices of entries which are on the diagonal
        gsize = self._graph_shape[0]**2
        diag_inds = np.where(
            np.arange(gsize) % (self._graph_shape[0]+1) == 0)

        # insert
        for i in diag_inds[0]:
            flat = np.insert(flat, i, repl)

        return flat.reshape(self._graph_shape)

    def _compute_A(self, i, phase_diffs):
        """
        Compute flat representation of coupling terms
        """
        coeffs_A = np.zeros(self._graph_shape)

        tmp = np.reshape(phase_diffs, (len(phase_diffs), 1))
        coeffs_A[i,:] = np.sin(tmp - tmp.transpose())[i,:]

        return self._mat_to_flat(coeffs_A)

    def _compute_B(self, i, phase_diffs):
        """
        Compute terms of external force coupling
        """
        coeffs_B = np.zeros(self._graph_shape[0])
        coeffs_B[i] = np.sin(phase_diffs[i])

        return coeffs_B

    def reconstruct(self):
        """
        Reconstruct parameters from provided data
        """
        # assemble linear system
        rhs = []
        lhs = []
        for (OMEGA, omega), phase_diffs in tqdm(self.data):
            for i in range(len(phase_diffs)):
                # coefficient matrix
                flat_A = self._compute_A(i, phase_diffs)
                flat_B = self._compute_B(i, phase_diffs)

                row = flat_A.tolist() + flat_B.tolist()
                rhs.append(row)

                # solution vector
                theta_dot = OMEGA - omega
                lhs.append(theta_dot)

        print('Using', len(rhs), 'data points to solve system of', self._graph_shape[0]**2, 'variables')

        # solve system
        x = np.linalg.lstsq(rhs, lhs)[0]

        # extract reconstructed parameters from solution
        rec_a = self._flat_to_mat(x[:-self._graph_shape[0]])
        rec_b = x[-self._graph_shape[0]:]

        print('Reconstructed A:\n', rec_a)
        print('Reconstructed B:', rec_b)

        return rec_a, rec_b


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

        pdiffs = System.extract_phase_differences(sols, ts, syst.Phi)
        #System.plot_solution(syst.Phi(ts), sols, ts)

        data.append(((OMEGA, omega), pdiffs))

    # reconstruct parameters
    recon = Reconstructor(ts, data)
    res = recon.reconstruct()

if __name__ == '__main__':
    main()

"""
Investigate how to reconstruct parameters from scarce information
"""

import numpy as np
import pandas as pd
import networkx as nx

from scipy.integrate import odeint

import seaborn as sns
import matplotlib.pylab as plt


def find_tpi_crossings(series):
    """
    Find indices where time series wraps from 2pi back to 0

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
    def extract_phase_differences(sols, ts, driver, t_threshold):
        """
        Extract phase difference of each oscillator to driving force in locked state

        Arguments:
            sols
                List of system solutions
            ts
                List of time points of the simulation
            driver
                Function of driver of system
            t_threshold
                Time point after which oscillators are assumed to be locked
        """
        # cut off transient
        cutoff_idx = (np.abs(ts-t_threshold)).argmin()
        sols = sols.T[cutoff_idx:].T

        # find driver's phase
        driver_theta = driver(ts[cutoff_idx:])
        driver_idx = find_tpi_crossings(driver_theta%(2*np.pi))

        # find oscillators phases
        osci_idx = []
        for sol in sols:
            flashes = find_tpi_crossings(sol%(2*np.pi))
            osci_idx.append(flashes)

        # compute smallest oscillator phase difference to last external flash
        ext_flash_idx = driver_idx[-1]

        phase_diffs = []
        for idxs in osci_idx:
            diffs = []
            for i in idxs:
                if ext_flash_idx < i: continue
                diffs.append(ext_flash_idx - i)
            phase_diffs.append(min(diffs))

        # convert index difference to actual time differences
        ref_ind = cutoff_idx + ext_flash_idx
        t_diffs = [ts[ref_ind] - ts[ref_ind-pdiff] for pdiff in phase_diffs]

        return t_diffs

    @staticmethod
    def plot_solution(sols, ts):
        """
        Plot solution of oscillator system.

        Arguments:
            sols
                List of system solutions
            ts
                List of time points of the simulation
        """
        # confine to circular region
        sols %= 2*np.pi

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

        # plot result
        plt.figure()
        sns.tsplot(
            time='time', value='theta',
            condition='oscillator', unit='source',
            data=df)
        plt.show()


def main():
    """
    Main interface
    """
    graph = nx.cycle_graph(4)
    dim = len(graph.nodes())

    A = nx.to_numpy_matrix(graph)
    B = np.random.uniform(0, 5, size=dim)

    omega = np.random.uniform(0, 3)
    OMEGA = 3

    syst = System(A, B, omega, OMEGA)
    sols, ts = syst.solve(0.01, 20)

    pdiffs = System.extract_phase_differences(sols, ts, syst.Phi, 15)
    print(pdiffs)
    System.plot_solution(sols, ts)

if __name__ == '__main__':
    main()

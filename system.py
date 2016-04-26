"""
Class which stores coupled collection of Kuramoto oscillators
"""

import numpy as np
import pandas as pd

from scipy.integrate import odeint

import seaborn as sns
import matplotlib.pylab as plt


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

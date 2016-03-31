"""
Commonly used functions
"""

import numpy as np
import numpy.random as npr

from scipy.integrate import odeint


class DictWrapper(dict):
    """ Dict with dot-notation access functionality
    """
    def __getattr__(self, attr):
        return self.get(attr)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def generate_system(omega_vec, A, B, Phi):
    """ Generate generalized Kuramoto model
    """
    def func(theta, t=0):
        ode = []
        for i, omega in enumerate(omega_vec):
            ode.append(
                omega \
                + np.sum([A[i,j] * np.sin(theta[j] - theta[i])
                    for j in range(len(omega_vec))]) \
                + B[i] * np.sin(Phi(t) - theta[i])
            )
        return np.array(ode)
    return func

def solve_system(conf, tmax=0.05):
    """ Solve particular configuration
    """
    func = generate_system(conf.o_vec, conf.A, conf.B, conf.Phi)

    ts = np.arange(0, tmax, conf.dt)
    init = npr.uniform(0, 2*np.pi, size=conf.o_vec.shape)

    sol = odeint(func, init, ts).T
    sol %= 2*np.pi

    return sol, ts

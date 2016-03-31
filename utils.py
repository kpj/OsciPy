"""
Commonly used functions
"""

import os

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

def solve_system(conf):
    """ Solve particular configuration
    """
    func = generate_system(conf.o_vec, conf.A, conf.B, conf.Phi)

    ts = np.arange(0, conf.tmax, conf.dt)
    init = npr.uniform(0, 2*np.pi, size=conf.o_vec.shape)

    sol = odeint(func, init, ts).T
    sol %= 2*np.pi

    return sol, ts

def save(fig, fname, **kwargs):
    """ Save figure in multiple formats and make sure that path exists
    """
    # save images in right directory
    sname = 'images/{}'.format(fname)

    # make sure directory exists
    fdir = os.path.dirname(sname)
    if len(fdir) > 0 and not os.path.isdir(fdir):
        os.makedirs(fdir)

    # save figure
    fig.savefig('{}.pdf'.format(sname), **kwargs)
    fig.savefig('{}.png'.format(sname), **kwargs)

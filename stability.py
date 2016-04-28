"""
Analytically investigate stability of oscillator system
"""

import numpy as np
from scipy.integrate import odeint

import matplotlib.pylab as plt

from reconstruction import find_tpi_crossings


class StabilityInvestigator(object):
    """
    Investigate stability of provided system configuration
    """
    def __init__(self, func):
        """
        Initialize system.

        Arguments
            func
                Definition of ODE
        """
        self.func = func

        self.max_val = 2 * np.pi

    def _get_ode_values(self, resolution):
        """
        Compute ODE at mesh of given resolution.

        Arguments
            resolution
                Resolution of plot
        """
        x_dom = np.linspace(0, self.max_val, resolution)
        y_dom = np.linspace(0, self.max_val, resolution)

        x_mesh, y_mesh = np.meshgrid(x_dom, y_dom)
        ode_x, ode_y = self.func((x_mesh, y_mesh))

        return x_mesh, y_mesh, ode_x, ode_y

    def _plot_vector_field(self, resolution):
        """
        Plot vector field.

        Arguments
            resolution
                Resolution of plot
        """
        x_mesh, y_mesh, ode_x, ode_y = self._get_ode_values(resolution)

        # scale vector field
        hyp = np.hypot(ode_x, ode_y)
        hyp[hyp==0] = 1.
        ode_x /= hyp
        ode_y /= hyp

        plt.quiver(
            x_mesh, y_mesh,
            ode_x, ode_y,
            pivot='mid', color='lightgray')

    def _plot_nullclines(self, resolution):
        """
        Plot nullclines.

        Arguments
            resolution
                Resolution of plot
        """
        x_mesh, y_mesh, ode_x, ode_y = self._get_ode_values(resolution)

        plt.contour(
            x_mesh, y_mesh, ode_x,
            levels=[0], linewidths=2, colors='black')
        plt.contour(
            x_mesh, y_mesh, ode_y,
            levels=[0], linewidths=2, colors='black')

    def _plot_trajectories(self, initial_conds):
        """
        Plot trajectories.

        Arguments
            initial_conds
                List of initial conditions for trajectories
        """
        def fix_wrapping(series):
            idx = find_tpi_crossings(series)
            series[idx] = None
            return series

        ts = np.linspace(0, 10, 500)

        lbl = r'$\varphi_i$ ODE trajectory'
        for init in initial_conds:
            sol = odeint(self.func, init, ts)
            phi_0, phi_1 = sol.T % self.max_val

            # fix wrapping
            phi_0 = fix_wrapping(phi_0)
            phi_1 = fix_wrapping(phi_1)

            plt.plot(phi_0, phi_1, linewidth=3, label=lbl)
            lbl = None


        #p0, p1 = np.load('pdiffs.npy') % self.max_val
        #p0 = fix_wrapping(p0)
        #p1 = fix_wrapping(p1)
        #plt.plot(p0, p1, linewidth=3, label=r'$\Theta$ ODE trajectory')

    def phase_space(self, resolution=200, initial_conds=[]):
        """
        Plot phase space of system.

        Arguments
            resolution
                Resolution of plot
            initial_conds
                List of initial conditions for trajectories
        """
        fig = plt.figure()

        self._plot_vector_field(resolution/10)
        self._plot_nullclines(resolution)
        self._plot_trajectories(initial_conds)

        plt.legend(loc='best')
        plt.xlabel(r'$\varphi_0$')
        plt.ylabel(r'$\varphi_1$')
        plt.title('Phase plane')

        plt.savefig('images/phase_space.pdf')

def generate_ode(OMEGA=4, omega=2, A=1, B=2):
    """
    Generate ODE system
    """
    def func(state, t=0):
        phi_0, phi_1 = state

        return np.array([
            OMEGA - omega - A * np.sin(phi_0 - phi_1) - B * np.sin(phi_0),
            OMEGA - omega - A * np.sin(phi_1 - phi_0) - B * np.sin(phi_1),
        ])

    return func

def main():
    """
    Main interface
    """
    func = generate_ode()
    stabi = StabilityInvestigator(func)

    stabi.phase_space(initial_conds=[[2,1], [2,4], [4,1]])

if __name__ == '__main__':
    main()

"""
Analytically investigate stability of oscillator system.

Convert images to gif: convert -delay 100 -loop 0 *.png animation.gif
"""

import numpy as np
from scipy.integrate import odeint

from sympy import Symbol, symbols, sin, Matrix
from sympy.utilities.lambdify import lambdify
from sympy.solvers.solvers import nsolve

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
            levels=[0], linewidths=2, colors='black',
            linestyles='dashed')

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


        p0, p1 = np.load('pdiffs.npy') % self.max_val
        p0 = fix_wrapping(p0)
        p1 = fix_wrapping(p1)
        plt.plot(p0, p1, linewidth=3, label=r'$\Theta$ ODE trajectory')

    def phase_space(self, resolution=200, initial_conds=[], fname_app=None):
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
        #self._plot_trajectories(initial_conds)

        #plt.legend(loc='best')
        plt.xlabel(r'$\varphi_0$')
        plt.ylabel(r'$\varphi_1$')
        plt.title(
            'Phase plane{}'.format(
                '' if fname_app is None else ' ({})'.format(fname_app)))

        plt.savefig(
            'images/phase_space{}.png'.format(
                '' if fname_app is None else '_{:04}'.format(fname_app)))
        plt.close()


class Functions(object):
    def __init__(self, N):
        self.N = N

        self.O = Symbol('Ω', real=True)
        self.o = Symbol('ω', real=True)
        self.A = Symbol('A', real=True)
        self.B = Symbol('B', real=True)

        self.phis = symbols('ϕ0:{}'.format(self.N), real=True)

    def _gen_system(self):
        syst = []
        for i in range(self.N):
            eq = self.O - self.o - sum([self.A * sin(self.phis[i] - self.phis[j]) for j in range(self.N) if i != j]) - self.B * sin(self.phis[i])
            yield eq

    def get_equations(self, substitutions):
        eqs = []
        for eq in self._gen_system():
            for sym, val in substitutions.items():
                eq = eq.subs(sym, val)
            eqs.append(eq)
        return eqs

    def get_roots(self, eqs):
        res = nsolve(eqs, self.phis, [1]*len(eqs), verify=False)#, modules=['numpy'])
        return res

    def get_jacobian(self, eqs, at=None):
        jac = Matrix(eqs).jacobian(Matrix(self.phis))

        if not at is None:
            assert len(self.phis) == len(at)
            for p, v in zip(self.phis, at):
                jac = jac.subs(p, v)

        return jac

    def get_ode(self, eqs):
        eqs = [lambdify(self.phis, e, 'numpy') for e in eqs]
        def func(state, t=0):
            return [e(*state) for e in eqs]
        return func

    @staticmethod
    def is_stable(jac):
        #print([N(k) for k in jac.eigenvals().keys()])
        return (np.array([k for k in jac.eigenvals().keys()]) <= 0).all()

    @staticmethod
    def fixroot(root):
        r = root.tolist()

        out = []
        for ele in r:
            cur = ele[0]
            if cur > np.pi:
                out.append([cur-np.pi])
            else:
                out.append([cur])

        return Matrix(out)

    @staticmethod
    def main():
        f = Functions(1)
        eqs = f.get_equations({f.O: 3.3, f.o: 2, f.B: 2, f.A: 1})
        ode = f.get_ode(eqs)

        root = Functions.fixroot(f.get_roots(eqs))
        jac = f.get_jacobian(eqs, at=root)

        print(root, jac)
        print(Functions.is_stable(jac))


def generate_ode(OMEGA=4, omega=2, A=1, B=2):
    """
    Generate ODE system
    """
    f = Functions(2)
    eqs = f.get_equations()
    return f.get_ode(eqs)

def main():
    """
    Main interface
    """
    func = generate_ode()
    stabi = StabilityInvestigator(func)

    stabi.phase_space(
        initial_conds=[[2,1], [2,4], [4,1]])

if __name__ == '__main__':
    #main()

    Functions.main()

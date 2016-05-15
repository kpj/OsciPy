"""
Analytically investigate stability of oscillator system.

Convert images to gif: convert -delay 100 -loop 0 *.png animation.gif
"""

import itertools

import numpy as np
from scipy.integrate import odeint

from sympy import Symbol, symbols, sin, Matrix, N, re
from sympy.utilities.lambdify import lambdify
from sympy.solvers.solvers import nsolve

import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.lines as mlines

from tqdm import tqdm

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

        lblx = mlines.Line2D(
            [], [],
            color='black',
            marker='', markersize=15,
            label=r'$\dot\varphi_0=0$')
        lbly = mlines.Line2D(
            [], [],
            color='black', linestyle='dashed',
            marker='', markersize=15,
            label=r'$\dot\varphi_1=0$')
        plt.legend(handles=[lblx, lbly], loc='best')

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
        self._plot_trajectories(initial_conds)

        plt.xlabel(r'$\varphi_0$')
        plt.ylabel(r'$\varphi_1$')
        plt.title(
            'Phase plane{}'.format(
                '' if fname_app is None else ' ({})'.format(fname_app)))

        plt.savefig(
            'images/phase_space{}.pdf'.format(
                '' if fname_app is None else '_{:04}'.format(fname_app)))
        plt.close()


class Functions(object):
    def __init__(self, N):
        self.N = N

        self.O = Symbol('Ω', real=True)
        self.o = Symbol('ω', real=True)

        self.As = []
        for i in range(self.N):
            self.As.append([])
            for j in range(self.N):
                sym = Symbol('A{}{}'.format(i,j), real=True)
                self.As[-1].append(sym)
        self.As = tuple(tuple(x) for x in self.As)

        self.Bs = symbols('B0:{}'.format(self.N), real=True)
        self.phis = symbols('ϕ0:{}'.format(self.N), real=True)

    def _gen_system(self):
        syst = []
        for i in range(self.N):
            eq = self.O - self.o - \
                sum([self.As[i][j] * sin(self.phis[i] - self.phis[j])
                    for j in range(self.N) if i != j]) - \
                self.Bs[i] * sin(self.phis[i])
            yield eq

    def get_equations(self, substitutions):
        # scale up if necessary
        val = substitutions[self.As]
        if np.isscalar(val):
            substitutions[self.As] = val * np.ones((self.N,self.N))
        else:
            assert np.asarray(val).shape == (self.N, self.N), 'Invalid dimension of A: {}'.format(np.asarray(val).shape)

        val = substitutions[self.Bs]
        if np.isscalar(val):
            substitutions[self.Bs] = val * np.ones(self.N)
        else:
            assert np.asarray(val).shape == (self.N,), 'Invalid dimension of B: {}'.format(np.asarray(val).shape)

        # apply substitutions (in an ugly way)
        eqs = []
        for eq in self._gen_system():
            for sym, val in substitutions.items():
                if isinstance(sym, tuple):
                    for ss, vv in zip(sym, val):
                        if isinstance(ss, tuple):
                            for s, v in zip(ss, vv):
                                eq = eq.subs(s, v)
                        else:
                            eq = eq.subs(ss, vv)
                else:
                    eq = eq.subs(sym, val)
            eqs.append(eq)
        return eqs

    def get_roots(self, eqs, steps=2):
        fix = lambda r: (r+2*np.pi) if r < 0 else r

        domain = np.linspace(0, np.pi, steps)
        space = np.meshgrid(*([domain]*self.N))
        points = np.vstack(map(np.ravel, space)).T

        roots = []
        for init in points:
            try:
                res = nsolve(eqs, self.phis, init.tolist(), verify=False)
                root = [round(fix(r), 5) for r in res]

                if root not in roots:
                    roots.append(root)
            except:# ZeroDivisonError:
                pass
        return roots

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

    def get_roots_plus(self, eqs):
        roots = self.get_roots(eqs)

        data = []
        for root in roots:
            jac = self.get_jacobian(eqs, at=root)
            eigvals = Functions.get_eigenvalues(jac)

            data.append((root, [re(e) for e in eigvals]))

        return data

    @staticmethod
    def get_eigenvalues(jac):
        # expect only eigenvalues of multiplicity one
        assert set(jac.eigenvals().values()) == set([1])

        return sorted([N(k) for k in jac.eigenvals().keys()])

    @staticmethod
    def get_stable_root(rp_data):
        return [(root, eigvals) for root, eigvals in rp_data
            if all([re(e) < 0 for e in eigvals])]

    @staticmethod
    def main(dim=2, reps=50):
        def gen(o=5, B=2):
            data = [[] for _ in range(dim)]
            for O in tqdm(np.linspace(0, 10, reps)):
                f = Functions(dim)
                eqs = f.get_equations({
                    f.O: O, f.o: o,
                    f.Bs: B,
                    f.As: 1})
                ode = f.get_ode(eqs)

                roots_p = f.get_roots_plus(eqs)
                stab = Functions.get_stable_root(roots_p)

                if len(stab) > 0:
                    for i, ev in enumerate(stab[0][1]):
                        data[i].append((O, ev))
            return data
        # compute data
        data = {
            r'$\omega=5, B=2$': gen(),
            r'$\omega=5, B=5$': gen(B=5)
        }

        # plot result
        plt.figure()

        used_lbls = set()
        colors = itertools.cycle(['blue', 'green', 'red'])
        for lbl, dat in data.items():
            col = next(colors)
            markers = itertools.cycle(['o', '*', 'D'])
            for e_list in dat:
                plt.plot(
                    *zip(*e_list),
                    marker=next(markers), linestyle='None', markersize=5,
                    color=col,
                    label=lbl if not lbl in used_lbls else None)
                used_lbls.add(lbl)

        plt.xlabel(r'$\Omega$')
        plt.ylabel(r'$\lambda_i$')
        plt.legend(loc='best')

        plt.savefig('images/stability_overview.pdf')


def generate_ode(OMEGA=4, omega=3, A=1, B=2):
    """
    Generate ODE system
    """
    f = Functions(2)
    eqs = f.get_equations({
        f.O: OMEGA,
        f.o: omega,
        f.As: A,
        f.Bs: B
    })
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

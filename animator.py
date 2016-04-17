"""
Animate evolution of system of oscillators
"""

import numpy as np
import networkx as nx

import matplotlib as mpl
mpl.use('Agg', force=True)

import matplotlib.pylab as plt
import matplotlib.animation as animation

from tqdm import tqdm

from scarce_information import System

class Animator(object):
    """
    Generate animated GIF of oscillator behavior
    """
    def __init__(self, sols, ts):
        """
        Initialize animator.

        Arguments:
            sols
                Solutions of individual oscillators
            ts
                List of time points
        """
        self.sols = sols
        self.ts = ts

        self.graph = None
        self.pos = None

        self.pbar = None

    def _init_graph(self):
        """
        Prepare graph
        """
        self.graph = nx.Graph()
        self.graph.add_nodes_from(i for i in range(len(self.sols)))

        self.pos = nx.nx_pydot.graphviz_layout(self.graph, prog='neato')

    def _clear_axis(self, ax):
        """
        Prepare axis for drawing.

        Arguments
            ax
                Axis to be prepared
        """
        # networkx turns this off by default
        plt.axis('on')

        # make it fancy
        ax.set_axis_bgcolor('black')
        ax.grid(b=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    def _get_alpha_mapping(self, t):
        """
        Generate alpha channel mapping for current point in time.

        Arguments
            t
                Index of current time point
        """
        amap = {}
        for node in self.graph.nodes():
            sol = self.sols[node]
            val = sol[t]

            amap[node] = val / (2*np.pi)
        return amap

    def _update(self, t, ax, pbar):
        """
        Draw each frame of the animation.

        Arguments
            t
                Frame index in animation
            ax
                Axis to draw on
            pbar
                Progressbar handle
        """
        pbar.update(1)
        plt.cla()

        alpha_map = self._get_alpha_mapping(t)
        for node, alpha in alpha_map.items():
            nx.draw_networkx_nodes(
                self.graph, self.pos,
                nodelist=[node],
                node_color='yellow', node_size=800,
                font_size=20, alpha=alpha,
                ax=ax)

        self._clear_axis(ax)
        ax.set_title(r'$t={:.2}$'.format(self.ts[t]))

    def _animate(self, fname):
        """
        Do actual animation work.

        Arguments
            fname
                Destination filename of GIF
        """
        if self.graph is None:
            raise RuntimeError('Graph not yet created')

        fig = plt.figure()
        ax = plt.gca()

        with tqdm(total=len(self.ts)) as pbar:
            ani = animation.FuncAnimation(
                fig, self._update,
                frames=len(self.ts),
                fargs=(ax, pbar))

        ani.save(fname, writer='imagemagick', fps=10, dpi=50) # 200

    def create(self, fname):
        """
        Create GIF animation.

        Arguments
            fname
                Destination filename of GIF
        """
        self._init_graph()
        self._animate(fname)


def get_basic_system():
    """
    Generate simple exemplary system
    """
    graph = nx.cycle_graph(4)
    dim = len(graph.nodes())

    A = nx.to_numpy_matrix(graph)
    B = np.random.uniform(0, 5, size=dim)

    omega = np.random.uniform(0, 3)
    OMEGA = 3

    return System(A, B, omega, OMEGA)

def main():
    """
    Exemplary usage
    """
    syst = get_basic_system()
    sols, ts = syst.solve(0.01, 20)

    step = 10
    sols, ts = sols.T[::step].T, ts[::step]

    anim = Animator(sols % (2*np.pi), ts)
    anim.create('network.gif')

if __name__ == '__main__':
    main()

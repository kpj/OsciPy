"""
All functions related to plotting
"""

import numpy as np
import networkx as nx

import matplotlib as mpl
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import save


def plot_matrix(mat, ax, title=None, allow_negative=True):
    """ Plot system evolution
    """
    cmap = plt.cm.coolwarm
    cmap.set_under('white')

    im = ax.imshow(
        mat, vmin=np.min(mat) if allow_negative else 0,
        interpolation='nearest', cmap=cmap)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.2)
    plt.colorbar(im, cax=cax)

    ax.set_xlabel(r'$i$')
    ax.set_ylabel(r'$j$')

    if not title is None:
        ax.set_title(title)

def plot_graph(graph, ax):
    """ Plot graph
    """
    # generate some node/edge properties
    node_labels = {}
    for n in graph.nodes():
        node_labels[n] = n

    edge_labels = {}
    for source, sink, data in graph.edges(data=True):
        edge_labels[(source, sink)] = round(data['weight'], 2)

    # compute layout
    pos = nx.nx_pydot.graphviz_layout(graph, prog='neato')

    # draw graph
    nx.draw(
        graph, pos,
        node_color='lightskyblue', node_size=800,
        font_size=20,
        ax=ax)
    nx.draw_networkx_labels(graph, pos, node_labels)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels)

def plot_evolutions(sols, ts, ax):
    """ Plot system evolution
    """
    ax.imshow(
        sols, aspect='auto',
        cmap=plt.cm.gray, interpolation='nearest')

    # let x-labels correspond to actual time steps
    ts = np.append(ts, ts[-1]+(ts[-1]-ts[-2]))
    formatter = plt.FuncFormatter(lambda x, pos: int(ts[x]))
    ax.xaxis.set_major_formatter(formatter)

    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\Theta_i$')

def plot_correlation_matrix(cmat, ts, ax):
    """ Plot individual correlation matrix
    """
    for i, row in enumerate(cmat):
        for j, sol in enumerate(row):
            ax.plot(ts, sol, label='{},{}'.format(i, j))

    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\langle \cos \left(\Theta_i(t) - \Theta_j(t)\right)\rangle$')
    ax.set_ylim((-1, 1.1))

def plot_series(series, ts, ax):
    """ Plot single time series
    """
    ax.plot(ts, series)

    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\langle\frac{N^2}{N^2 - 2 \sum_{ij} \sin\left( \theta_i - \theta_j \right)^2}\rangle$')

def plot_result(data):
    """ Plot final result
    """
    # main overview
    fig = plt.figure(figsize=(30, 10))
    gs = mpl.gridspec.GridSpec(2, 3, width_ratios=[1, 1, 2])

    plot_graph(
        data.graph, plt.subplot(gs[:, 0]))
    plot_matrix(
        data.syncs, plt.subplot(gs[:, 1]),
        title='Sign-switch of dynamic connectivity matrix',
        allow_negative=False)
    plot_evolutions(
        data.sols[0], data.ts, plt.subplot(gs[0, 2]))
    plot_correlation_matrix(
        np.mean(data.cmats, axis=0), data.ts, plt.subplot(gs[1, 2]))

    plt.tight_layout()
    save(fig, 'overview_plot')

    # variance investigation
    fig = plt.figure(figsize=(20, 10))
    gs = mpl.gridspec.GridSpec(2, 1)

    plot_evolutions(data.sols[0], data.ts, plt.subplot(gs[0]))
    plot_series(data.vser, data.ts, plt.subplot(gs[1]))

    plt.tight_layout()
    save(fig, 'cluster_development')

    # correlation matrix heatmap
    fig = plt.figure()

    cmat_sum = np.sum(np.mean(data.cmats, axis=0), axis=2)
    plot_matrix(
        cmat_sum, plt.gca(),
        title='Summed correlation matrix')

    plt.tight_layout()
    save(fig, 'correlation_matrix')

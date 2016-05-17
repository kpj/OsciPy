"""
Aggregate functions which handle plotting
"""

import numpy as np
import networkx as nx

import matplotlib as mpl
import matplotlib.pylab as plt

from reconstruction import compute_error


def plot_graph(A, B, ax, bundle=None, verbose=True):
    """
    Plot graph on given axis.

    Arguments
        A
            Adjacency matrix of graph to plot
        B
            Vector of coupling to external force
        ax
            Axis to plot on
        bundle (optional)
            If given, label edges with reconstruction error
    """
    graph = nx.from_numpy_matrix(A)

    # add external force
    if verbose:
        graph.add_node('F', color='red')

        for i, b in enumerate(B):
            if abs(b) > 1e-1:
                graph.add_edge('F', i, weight=b, style='dashed')

    # generate some node/edge properties
    node_labels = {}
    for n in graph.nodes():
        node_labels[n] = n

    node_colors = []
    for n, data in graph.nodes(data=True):
        node_colors.append(data.get('color', 'lightskyblue'))

    if not bundle is None:
        err_A, err_B = compute_error(bundle)

    edge_labels = {}
    for source, sink, data in graph.edges(data=True):
        if bundle is None:
            edge_labels[(source, sink)] = round(data['weight'], 2)
        else:
            if sink == 'F':
                edge_labels[(source, sink)] = round(err_B[source], 2)
            else:
                edge_labels[(source, sink)] = round(np.mean([
                    err_A[source, sink],
                    err_A[sink, source]
                ]), 2)

    edge_style = []
    for source, sink, data in graph.edges(data=True):
        edge_style.append(data.get('style', 'solid'))

    # compute layout
    pos = nx.nx_pydot.graphviz_layout(graph, prog='neato')

    # draw graph
    plt.axis('off')

    nx.draw_networkx_nodes(
        graph, pos,
        node_color=node_colors, node_size=800 if verbose else 400,
        font_size=20,
        ax=ax)
    if verbose:
        nx.draw_networkx_labels(
            graph, pos,
            node_labels,
            ax=ax)
    nx.draw_networkx_edges(
        graph, pos,
        style=edge_style,
        ax=ax)
    if verbose:
        nx.draw_networkx_edge_labels(
            graph, pos,
            edge_labels,
            ax=ax)

def show_reconstruction_overview(syst, bundle):
    """
    Plot representation of reconstruction results.

    Arguments
        syst
            Exemplary system used for reconstruction
        orig_A
            Original matrix A
        orig_B
            Original vector B
        rec_A
            Reconstructed matrix A
        rec_B
            Reconstructed vector B
    """
    fig = plt.figure(figsize=(20, 8))
    gs = mpl.gridspec.GridSpec(1, 2)

    err_A, err_B = compute_error(bundle)
    plt.suptitle(
        r'$A_{{err}} = {:.2}, B_{{err}} = {:.2}$'.format(
            np.mean(err_A), np.mean(err_B)),
        fontsize=24)

    # original graph
    orig_ax = plt.subplot(gs[0])
    orig_ax.set_title('Original graph', fontsize=24)
    orig_ax.set_aspect('equal')

    plot_graph(bundle.orig_A, bundle.orig_B, orig_ax)

    # reconstructed graph
    rec_ax = plt.subplot(gs[1])
    rec_ax.set_title('Reconstructed graph', fontsize=24)
    rec_ax.set_aspect('equal')

    tmp = bundle.rec_A
    tmp[abs(tmp) < 1e-1] = 0
    plot_graph(tmp, bundle.rec_B, rec_ax, bundle=bundle)

    # save plot
    plt.tight_layout()
    plt.savefig('images/reconstruction_overview.pdf', bbox_inches='tight')

    return err_A, err_B

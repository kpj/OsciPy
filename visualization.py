"""
Aggregate functions which handle plotting
"""

import networkx as nx

import matplotlib as mpl
import matplotlib.pylab as plt


def plot_graph(A, B, ax):
    """
    Plot graph on given axis.

    Arguments
        A
            Adjacency matrix of graph to plot
        B
            Vector of coupling to external force
        ax
            Axis to plot on
    """
    graph = nx.from_numpy_matrix(A)

    # add external force
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

    edge_labels = {}
    for source, sink, data in graph.edges(data=True):
        edge_labels[(source, sink)] = round(data['weight'], 2)

    edge_style = []
    for source, sink, data in graph.edges(data=True):
        edge_style.append(data.get('style', 'solid'))

    # compute layout
    pos = nx.nx_pydot.graphviz_layout(graph, prog='neato')

    # draw graph
    plt.axis('off')

    nx.draw_networkx_nodes(
        graph, pos,
        node_color=node_colors, node_size=800,
        font_size=20,
        ax=ax)
    nx.draw_networkx_labels(
        graph, pos,
        node_labels,
        ax=ax)
    nx.draw_networkx_edges(
        graph, pos,
        style=edge_style,
        ax=ax)
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
    fig = plt.figure(figsize=(32, 8))
    gs = mpl.gridspec.GridSpec(1, 2)

    # original graph
    orig_ax = plt.subplot(gs[0])
    orig_ax.set_title('Original graph')
    orig_ax.set_aspect('equal')

    plot_graph(bundle.orig_A, bundle.orig_B, orig_ax)

    # reconstructed graph
    rec_ax = plt.subplot(gs[1])
    rec_ax.set_title('Reconstructed graph')
    rec_ax.set_aspect('equal')

    tmp = bundle.rec_A
    tmp[abs(tmp) < 1e-1] = 0
    plot_graph(tmp, bundle.orig_B, rec_ax)

    # save plot
    plt.tight_layout()
    plt.savefig('images/reconstruction_overview.pdf')

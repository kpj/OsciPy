"""
All graph generators
"""

import networkx as nx
import numpy as np


def generate_graph(size):
    """ Generate graph with clusters
    """
    clus_num = 4
    graph = nx.complete_graph(size)

    for j in range(1, clus_num):
        g = nx.complete_graph(size)
        nl_map = {}
        for i in range(size):
            nl_map[i] = i + size * j
        g = nx.relabel_nodes(g, nl_map)

        graph = nx.compose(graph, g)

    graph.add_edges_from([
        (0,4), (1,5),
        (8,12), (9,13),
        (0,12)
    ])

    return graph

def generate_ring_graph(size):
    """ Generate graph in rign structure
    """
    graph = nx.Graph()

    for i in range(size-1):
        graph.add_edge(i, i+1, weight=np.random.uniform(0.5, 3))
    graph.add_edge(size-1, 0, weight=np.random.uniform(0.5, 3))

    return graph

def generate_paper_graph(_):
    """ Generate graph from figure 3 of A. Arenas et al. / Physica D 224 (2006)
    """
    graph = nx.Graph()
    graph.add_edges_from([
        (1,5),(5,6),(6,1),
        (0,3),(3,4),(4,0),
        (2,7),(7,8),(8,2),
        (1,0),(5,0),(6,0),
        (2,0),(7,0),(8,0)
    ])
    return graph

def generate_simple_graph(_):
    graph = nx.Graph()

    graph.add_edge(0, 1)

    return graph

def generate_snake_graph(size):
    graph = nx.Graph()

    for i in range(size-1):
        graph.add_edge(i, i+1)

    return graph

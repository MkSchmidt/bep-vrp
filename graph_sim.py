import networkx as nx
import pandas as pd
from itertools import pairwise
from typing import Optional

def graph_from_data(edges: pd.DataFrame, nodes: Optional[pd.DataFrame] = None) -> nx.DiGraph:
    edge_list = [
        (edge["init_node"], edge["term_node"], edge)
        for edge in edges.to_dict(orient="records")
    ]
    node_list = [
        (node["node"], {"coordinates": (node["x"], node["y"])})
        for node in nodes.to_dict(orient="records")
    ]
    graph = nx.DiGraph()
    graph.add_nodes_from(node_list)
    graph.add_edges_from(edge_list)

    return graph

def get_path_length(graph: nx.DiGraph, path: list):
    accumulator = 0
    for source, dest in pairwise(path):
        accumulator += graph.edges[source, dest]["free_flow_time"]
    return accumulator

def to_dense_graph(graph: nx.DiGraph) -> nx.DiGraph:
    paths = nx.all_pairs_shortest_path(graph)
    edges = [
        (source, dest, {"free_flow_time": get_path_length(graph, path)})
        for source, source_paths in dict(paths).items()
        for dest, path in source_paths.items()
    ]

    return nx.DiGraph(edges)
    
if __name__ == "__main__":
    import os
    from read_files import load_edgefile, load_nodefile, project_root
    import random
    from networkx.drawing import nx_pylab
    import matplotlib.pyplot as plt

    edges: pd.DataFrame =  load_edgefile(os.path.join(project_root, "TransportationNetworks", "SiouxFalls", "SiouxFalls_net.tntp"))
    nodes = load_nodefile(os.path.join(project_root, "TransportationNetworks", "SiouxFalls", "SiouxFalls_node.tntp"))
    G = graph_from_data(edges, nodes=nodes)
    dense_directed = to_dense_graph(G)
    dense = nx.Graph(dense_directed)
    
    s = dense.subgraph([ random.randrange(1, len(dense.nodes)+1) for i in range(5) ])
    c_solution = nx.algorithms.approximation.traveling_salesman.christofides(s)
    
    for source, dest in pairwise(c_solution):
        dense.edges[source, dest]["visited"] = True
    
    node_positions = dict([ (i, G.nodes[i]["coordinates"]) for i in dense.nodes ])
    nx.draw_networkx_nodes(G, pos=node_positions)
    edges_to_draw = s.edges()
    edge_color = [ "red" if "visited" in dense.edges[source, dest] else "black" for source, dest in edges_to_draw ]

    nx.draw_networkx_edges(s, node_positions, edgelist=edges_to_draw, edge_color=edge_color)
    plt.show()

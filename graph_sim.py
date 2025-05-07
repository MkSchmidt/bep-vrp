import networkx as nx
import pandas as pd
from itertools import pairwise

def edges_to_graph(edges: pd.DataFrame) -> nx.DiGraph:
    edge_list = [
        (edge["init_node"], edge["term_node"], edge)
        for edge in edges.to_dict(orient="records")
    ]
    G = nx.DiGraph(edge_list)

    return G

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
    from read_files import load_edgefile, project_root

    edges: pd.DataFrame =  load_edgefile(os.path.join(project_root, "TransportationNetworks", "SiouxFalls", "SiouxFalls_net.tntp"))
    G = edges_to_graph(edges)

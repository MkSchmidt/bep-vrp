import networkx as nx
import pandas as pd

def edges_to_graph(edges: pd.DataFrame) -> nx.MultiGraph:
    edge_list = [
        (edge["init_node"], edge["term_node"], edge)
        for edge in edges.to_dict(orient="records")
    ]
    G = nx.MultiGraph(edge_list)

    return G
    
if __name__ == "__main__":
    import os
    from read_files import load_edgefile, project_root

    edges: pd.DataFrame =  load_edgefile(os.path.join(project_root, "TransportationNetworks", "SiouxFalls", "SiouxFalls_net.tntp"))
    G = edges_to_graph(edges)

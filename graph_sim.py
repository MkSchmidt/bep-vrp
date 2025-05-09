import networkx as nx
import pandas as pd
from itertools import pairwise
from typing import Optional

def graph_from_data(edges: pd.DataFrame, nodes: Optional[pd.DataFrame] = None) -> nx.DiGraph:
    graph = nx.DiGraph()
    if nodes is not None:
        node_list = [
            (node["node"], {"coordinates": (node["x"], node["y"])})
            for node in nodes.to_dict(orient="records")
        ]
        graph.add_nodes_from(node_list)

    edge_list = [
        (edge["init_node"], edge["term_node"], edge)
        for edge in edges.to_dict(orient="records")
    ]
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

# give current time as minutes since midnight
def get_added_travel_time(edge, t):
    t_tc = ((t % (24*60)) - 18*60) # assume peak hour at 18:00
    h = edge["volume"] / edge["capacity"] * 10
    hump = h + min(-0.5*t_tc, h/90*t_tc)

    return max(0, hump)

if __name__ == "__main__":
    import os
    from read_files import load_edgefile, load_flowfile, load_nodefile, load_nodefile_geojson, project_root
    import random
    from networkx.drawing import nx_pylab
    import matplotlib.pyplot as plt
    from matplotlib import animation

    edges: pd.DataFrame =  load_edgefile(os.path.join(project_root, "TransportationNetworks", "Chicago-Sketch", "ChicagoSketch_net.tntp"))
    nodes = load_nodefile(os.path.join(project_root, "TransportationNetworks", "Chicago-Sketch", "ChicagoSketch_node.tntp"))
    flow = load_flowfile(os.path.join(project_root, "TransportationNetworks", "Chicago-Sketch", "ChicagoSketch_flow.tntp"))
    G = graph_from_data(edges, nodes=nodes)
    
    undirected = nx.Graph(G)
    for flow_row in flow.to_dict(orient="records"):
        undirected.edges[flow_row["from"], flow_row["to"]]["volume"] = flow_row["volume"]

    node_positions = dict([ (i, undirected.nodes[i]["coordinates"]) for i in undirected.nodes ])
    edges_to_draw = undirected.edges()
    
    fig, ax = plt.subplots()
    drawn_edges = nx.draw_networkx_edges(undirected, node_positions, edgelist=edges_to_draw, edge_color="0.8", ax=ax)
    title = plt.title("t=0")

    def update_colors(frame):
        t = 16*60 + (frame % (4*60))
        edge_times = [ get_added_travel_time(undirected.edges[edge], t) for edge in edges_to_draw ]
        colors = [ str(max(0.08 * (10 - travel_time), 0)) for travel_time in edge_times ]
        
        title.set_text(f"t={t//60:02d}:{t%60:02d}")
        drawn_edges.set_color(colors)

        return [drawn_edges, title]

    anim = animation.FuncAnimation(fig=fig, func=update_colors, frames=3*60, interval=40)
    plt.show()

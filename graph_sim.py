import networkx as nx
import pandas as pd
from itertools import pairwise
from typing import Optional
import math

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
    h = edge["volume"] / edge["capacity"] * edge["free_flow_time"] * 3
    hump = h + min(-0.2*t_tc, h/90*t_tc)

    return max(0, hump)

def get_travel_time(edge, t):
    added_time = get_added_travel_time(edge, t)
    return edge["free_flow_time"] + added_time

def get_node_sequence(graph, end):
    previous = graph.nodes[end]["previous"]
    if previous is None:
        return [end]
    return get_node_sequence(graph, previous) + [end]

def dynamic_dijkstra(graph, start, end, start_t):
    nx.set_node_attributes(graph, math.inf, "arrival_time")
    nx.set_node_attributes(graph, None, "previous")
    graph.nodes[start]["arrival_time"] = start_t
    
    Q = set(graph.nodes)

    while len(Q) > 0:
        minimum_distance_node = list(Q)[0]
        minimum_distance = graph.nodes[minimum_distance_node]["arrival_time"]
        for node_index in Q:
            node_distance = graph.nodes[node_index]["arrival_time"]
            if node_distance < minimum_distance:
                minimum_distance = node_distance
                minimum_distance_node = node_index
        if minimum_distance_node == end:
            return get_node_sequence(graph, end)
        Q.remove(minimum_distance_node)
        t = graph.nodes[minimum_distance_node]["arrival_time"]
        for successor in graph.neighbors(minimum_distance_node):
            if successor not in Q: continue
            travel_time = get_travel_time(graph.edges[minimum_distance_node, successor], t)
            potential_time = minimum_distance + travel_time
            if potential_time < graph.nodes[successor]["arrival_time"]:
                graph.nodes[successor]["arrival_time"] = potential_time
                graph.nodes[successor]["previous"] = minimum_distance_node

def arrival_times_for_path(graph, path, start_t):
    edges = dict()
    t = start_t
    for od in pairwise(path):
        edges[od] = t
        edges[od[1], od[0]] = t
        t += get_travel_time(graph.edges[od], t)

    return edges

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
    edges_to_draw = list(undirected.edges())
    
    fig, ax = plt.subplots()
    drawn_edges = nx.draw_networkx_edges(undirected, node_positions, edgelist=edges_to_draw, edge_color="0.8", ax=ax)
    title = plt.title("t=0")
    nx.draw_networkx_nodes(undirected, node_positions, nodelist=[911, 918])
    
    route_start_t = 15.5*60
    route = dynamic_dijkstra(undirected, 918, 911, route_start_t)
    visited_nodes = set(route)
    
    static_path = nx.shortest_path(undirected, source=918, target=911, weight="free_flow_time")
    static_edges = arrival_times_for_path(undirected, static_path, route_start_t)

    def update_colors(frame):
        t = 15.5*60 + (frame % (4*60))

        edge_times = [ get_added_travel_time(undirected.edges[edge], t) for edge in edges_to_draw ]
        tt_colors = [ str(max(0.8 / 15 * (15 - travel_time), 0)) for travel_time in edge_times ]
        d_colors = [ "blue" if undirected.nodes[edges_to_draw[i][0]]["arrival_time"] <= t and len(visited_nodes & set(edges_to_draw[i])) == 2 else tt_colors[i] for i in range(len(edges_to_draw)) ]
        
        colors = [ "red" if edges_to_draw[i] in static_edges and static_edges[edges_to_draw[i]] <= t else d_colors[i] for i in range(len(d_colors)) ]

        title.set_text(f"t={int(t)//60:02d}:{int(t)%60:02d}")
        drawn_edges.set_color(colors)

        return [drawn_edges, title]

    anim = animation.FuncAnimation(fig=fig, func=update_colors, frames=range(0, 4*60, 10), interval=200)
    
    plt.show()

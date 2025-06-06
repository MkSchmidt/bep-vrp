import os
import math
import random
from typing import Optional
import numpy as np
import pandas as pd
import networkx as nx
from itertools import pairwise
from matplotlib import pyplot as plt, animation
# Define VRP
depot = 918
customers = [300, 400, 911]
stops = [depot] + [customers] + [depot]


#Reading out files and Creating Directed Graph in Python

def graph_from_data(edges: pd.DataFrame, nodes: Optional[pd.DataFrame] = None) -> nx.DiGraph:
    graph = nx.DiGraph()
    if nodes is not None:
        node_list = [
            (node["node"], {"coordinates": (node["x"], node["y"])})
            for node in nodes.to_dict(orient="records")
        ]
        graph.add_nodes_from(node_list)

from read_files import (
    load_edgefile, load_flowfile,
    load_nodefile, project_root
)
from BsoLns_imp import BSOLNS


def graph_from_data(edges: pd.DataFrame, nodes: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from([
        (n["node"], {"coordinates": (n["x"], n["y"])})
        for n in nodes.to_dict("records")
    ])
    G.add_edges_from([
        (e["init_node"], e["term_node"], e)
        for e in edges.to_dict("records")
    ])
    return G


def get_path_length(graph: nx.DiGraph, path: list) -> float:
    return sum(
        graph.edges[u, v]["free_flow_time"]
        for u, v in pairwise(path)
    )


def get_added_travel_time(edge, t: float) -> float:
    # simple rush‐hour bump
    t_tc = ((t % (24*60)) - 18*60)
#Calculate Total free flow time along any path

def get_path_length(graph: nx.DiGraph, path: list):
    accumulator = 0
    for source, dest in pairwise(path):
        accumulator += graph.edges[source, dest]["free_flow_time"]
    return accumulator

# Calculate all shortest path for all posible source -> destination combo's ??

def to_dense_graph_with_congestion(graph: nx.DiGraph, start_time=0) -> nx.DiGraph:
    # Compute all-pairs shortest paths (by free-flow time)
    shortest_paths = dict(nx.all_pairs_dijkstra_path(graph, weight="free_flow_time"))
    
    edges = []

    # For each source -> destinatio pair
    for source, dest_paths in shortest_paths.items():
        for dest, path in dest_paths.items():
            # Calculate the ideal (no traffic) travel time
            free_time = get_path_length(graph, path)
            # Figure out when each edge is entered along the path
            actual_times = arrival_times_for_path(graph, path, start_time)
            # Calcute the total trip duration
            total_time = (
                actual_times[(path[-2], path[-1])] + 
                get_travel_time(graph.edges[path[-2], path[-1]], actual_times[(path[-2], path[-1])])
                - start_time
            )

            # Build a edge record database
            edges.append((source, dest, {
                "free_flow_time": free_time,
                "total_travel_time": total_time,
                "delay": total_time - free_time
            }))
    
    return nx.DiGraph(edges)



# Give current time as minutes since midnight
def get_added_travel_time(edge, t):
    t_tc = ((t % (24*60)) - 18*60) # assume peak hour at 18:00
    # Determines the congestion based on flow and capacity
    h = edge["volume"] / edge["capacity"] * edge["free_flow_time"] * 3
    return max(0, h + min(-0.2 * t_tc, h/90 * t_tc))


# Effective Travel time for specific edge at time of entering t

def get_travel_time(edge, t):
    added_time = get_added_travel_time(edge, t)
    return edge["free_flow_time"] + added_time

# Path-reconstruction

def get_node_sequence(graph, end):
    previous = graph.nodes[end]["previous"]
    if previous is None:
        return [end]
    return get_node_sequence(graph, previous) + [end]

# Run a dynamic Dijsktra algorithm

def dynamic_dijkstra(graph, start, end, start_t):
    nx.set_node_attributes(graph, math.inf, "arrival_time")
    nx.set_node_attributes(graph, None, "previous")
    graph.nodes[start]["arrival_time"] = start_t
    
    Q = set(graph.nodes) # Keeps track of unvisited nodes
# Keep processing nodes until all have been visited or end is reached
    while len(Q) > 0:
        minimum_distance_node = list(Q)[0]
        minimum_distance = graph.nodes[minimum_distance_node]["arrival_time"]
        # Scans all nodes in Q to find next best one
        for node_index in Q:
            node_distance = graph.nodes[node_index]["arrival_time"]
            if node_distance < minimum_distance:
                minimum_distance = node_distance
                minimum_distance_node = node_index
        # If end node is reached return the reconstructed path
        if minimum_distance_node == end:
            return get_node_sequence(graph, end)
        Q.remove(minimum_distance_node)
        t = graph.nodes[minimum_distance_node]["arrival_time"]
        # Compute and possibly reroute due to traffic
        for successor in graph.neighbors(minimum_distance_node):
            if successor not in Q: continue
            #Compute dynamic travel time for each edge to neighbor of current node
            travel_time = get_travel_time(graph.edges[minimum_distance_node, successor], t)
            # Time to gete to neighbor via current node
            potential_time = minimum_distance + travel_time

            # If path is faster -> reroute
            if potential_time < graph.nodes[successor]["arrival_time"]:
                graph.nodes[successor]["arrival_time"] = potential_time
                graph.nodes[successor]["previous"] = minimum_distance_node

# Compute the arrival time at each edge along given path 

def arrival_times_for_path(graph, path, start_t):
    edges = dict() # Create emoty stor for edge-entry times
    t = start_t # Intial start time

    # Store every entry time in both directions
    for od in pairwise(path):
        edges[od] = t
        edges[od[1], od[0]] = t
        t += get_travel_time(graph.edges[od], t) # Advance the clock by travel time


if __name__ == "__main__":
    # ── load network ───────────────────────────────────────────────
    edges = load_edgefile(os.path.join(project_root,"TransportationNetworks","Chicago-Sketch", "ChicagoSketch_net.tntp"))
    nodes = load_nodefile(os.path.join(project_root,"TransportationNetworks","Chicago-Sketch", "ChicagoSketch_node.tntp"))
    flow  = load_flowfile(os.path.join(project_root,"TransportationNetworks","Chicago-Sketch", "ChicagoSketch_flow.tntp"))

# Build the Graph in Python

    edges: pd.DataFrame =  load_edgefile(os.path.join(project_root, "TransportationNetworks", "Chicago-Sketch", "ChicagoSketch_net.tntp"))
    nodes = load_nodefile(os.path.join(project_root, "TransportationNetworks", "Chicago-Sketch", "ChicagoSketch_node.tntp"))
    flow = load_flowfile(os.path.join(project_root, "TransportationNetworks", "Chicago-Sketch", "ChicagoSketch_flow.tntp"))
    G = graph_from_data(edges, nodes=nodes)
    
    undirected = nx.Graph(G)
    for row in flow.to_dict("records"):
        undirected.edges[row["from"], row["to"]]["volume"] = row["volume"]

    node_positions = {
        n: undirected.nodes[n]["coordinates"]
        for n in undirected.nodes
    }
    edges_to_draw = list(undirected.edges())

    fig, ax = plt.subplots()
    drawn_edges = nx.draw_networkx_edges(
        undirected, node_positions,
        edgelist=edges_to_draw,
        edge_color="0.8", ax=ax
    )
    nx.draw_networkx_nodes(
        undirected, node_positions,
        nodelist=[911, 918], node_size=50, node_color="k"
    )
    title = ax.set_title("t=00:00")

    # ── compute static & dynamic Dijkstra ─────────────────────────
    route_start_t = 15.5 * 60  # 15:30 in minutes

    # dynamic
    dyn_path = dynamic_dijkstra(undirected, 918, 911, route_start_t)
    dynamic_edges = arrival_times_for_path(undirected, dyn_path, route_start_t)

    # static
    static_path = nx.shortest_path(
        undirected, source=918, target=911,
        weight="free_flow_time"
    )
    static_edges = arrival_times_for_path(undirected, static_path, route_start_t)

    # ── dynamic BSO‐LNS integration ────────────────────────────────
    # We’ll treat nodes [918→210→911] as depot+two customers
    bso_nodes = [918, 911]

    def td_travel_time(u_idx, v_idx, depart_t):
        """
        Returns the time‐dependent shortest‐path travel time from bso_nodes[u_idx]
        to bso_nodes[v_idx], departing at time depart_t.
        """
        u_node = bso_nodes[u_idx]
        v_node = bso_nodes[v_idx]
        # run dynamic Dijkstra on the full graph
        path = dynamic_dijkstra(undirected, u_node, v_node, depart_t)
        # dynamic_dijkstra writes the arrival_time into graph.nodes[v_node]
        arrival = undirected.nodes[v_node]["arrival_time"]
        return arrival - depart_t

    # instantiate BSO-LNS to visit that one customer and return
    bso = BSOLNS(
        travel_time_fn=td_travel_time,
        demands=[1],            # single customer
        vehicle_capacity=1,     # capacity must cover that single demand
        start_time=route_start_t,
        pop_size=5,             # you can tune these
        n_clusters=1,
        ideas_per_cluster=1,
        max_iter=20,
        remove_rate=0.2
    )

    best = bso.run()
    print("BSO-LNS dynamic best cost:", best["cost"], "routes:", best["sol"])

    # ── Reconstruct the actual two‐leg path for coloring ────────────────
    # 1) forward leg: depot → customer
    forward_path = dynamic_dijkstra(undirected, 918, 911, route_start_t)
    t_arrival = undirected.nodes[911]["arrival_time"]

    # 2) return leg: customer → depot
    return_path  = dynamic_dijkstra(undirected, 911, 918, t_arrival)

    # 3) stitch them together (drop duplicate 911)
    bso_full_path = forward_path + return_path[1:]

    # 4) compute the time each edge is entered
    bso_edges = arrival_times_for_path(undirected, bso_full_path, route_start_t)
    # ── animation update ──────────────────────────────────────────
    total_minutes = 6*60
    time_step = 10
    def update_colors(frame):
        # 'frame' here is already in minutes (because we use range(0, total_minutes, time_step))
        t = route_start_t + frame

        # recompute base intensities
        add_times = [ get_added_travel_time(undirected.edges[e], t)
                    for e in edges_to_draw ]
        base_int  = [ str(max(0.8/15 * (15 - tau), 0)) for tau in add_times ]

        default_colors = []
        for i, e in enumerate(edges_to_draw):
            if e in dynamic_edges and dynamic_edges[e] <= t:
                default_colors.append("blue")
            else:
                default_colors.append(base_int[i])

        # now overlay static (red), dynamic (blue), BSO (green)
        colors = []
        for e in edges_to_draw:
            if e in bso_edges and bso_edges[e] <= t:
                colors.append("green")
            elif e in static_edges and static_edges[e] <= t:
                colors.append("red")
            elif e in dynamic_edges and dynamic_edges[e] <= t:
                colors.append("blue")
            else:
                colors.append(default_colors[edges_to_draw.index(e)])

        title.set_text(f"t={int(t)//60:02d}:{int(t)%60:02d}")
        drawn_edges.set_color(colors)
        return [drawn_edges, title]
    

# Launch Animation

    # 3) animate with frames=actual minutes:
    anim = animation.FuncAnimation(
        fig,
        update_colors,
        frames=range(0, total_minutes, time_step),
        interval=200,  # ms between frames
        blit=True
    )

    plt.show()

import os
import math
import random
import numpy as np
import pandas as pd
import networkx as nx
from itertools import pairwise
from matplotlib import pyplot as plt, animation
from read_files import (
    load_edgefile, load_flowfile,
    load_nodefile, project_root, read_folder)
from BsoLns_imp import BSOLNS 
import mplcursors

# Define BSO-LNS Problem: Depot and Customers
depot_node_id = 1  # 918 
customer_node_ids = [2, 4, 5, 123, 210, 350, 123, 300]
time_step_minutes = 10  # mins
sim_start = 0 * 60  # 6:00
route_start_t = 8 * 60 + 30  # 15:30 (in minutes)
num_vehicles = 4
n_demand = [1] * len(customer_node_ids)  #Demand per customer
total_demand = sum(n_demand)
vehicle_capacity = math.ceil(total_demand / num_vehicles)
B =0.15
edge_example = 12 ,275 


# Time-breakpoints demand function
t1, t2, t3, t4 = 6.5 * 60, 8.5 * 60, 10 * 60, 12 * 60
t5, t6, t7, t8 = 16.5 * 60, 18 * 60, 20 * 60, 22 * 60

# Parameters for BSO
pop_size = 10
n_clusters = 3
ideas_per_cluster = 5
max_iter = 1
remove_rate = 0.3

# Function to build graph
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

# Compute path length (not used directly below but kept for reference)
def get_path_length(graph: nx.DiGraph, path: list) -> float:
    return sum(
        graph.edges[u, v]["free_flow_time"]
        for u, v in pairwise(path)
    )

# Demand function
def demand(t: float) -> float:
    low, medium, high = 0.1, 0.5, 1.1
    if t <= t1:
        return low
    elif t < t2:
        return low + (high - low) * (t - t1) / (t2 - t1)
    elif t < t3:
        return high
    elif t <= t4:
        return high - (high - medium) * (t - t3) / (t4 - t3)
    elif t <= t5:
        return medium
    elif t < t6:
        return medium + (high - medium) * (t - t5) / (t6 - t5)
    elif t <= t7:
        return high
    elif t < t8:
        return low + (high - low) * (1 - (t - t7) / (t8 - t7))
    else:
        return low

# --- Congestion Model Based on Demand ---
def get_flow(attrs: dict, t_min: float) -> float:
    volume = attrs.get("volume", 0)
    return volume * demand(t_min)
def get_capacity(attrs: dict, t_min: float) -> float:
    capacity = attrs.get("capacity")
    return capacity

# Determine Critical Density for each edge
def get_critical_density(attrs: dict):
    capacity = attrs.get("capacity")                    #[veh/h]
    free_time = attrs.get("free_flow_time")/60          #[h]
    free_time = 1.667 if free_time == 0 else free_time # Neighboorhoodroads, dont have Free_flow_time in Data ---> ~30 km/h
    length = attrs.get("length")* 0.3048/1000           #[km]
    ff_speed = length / free_time                       #[km/h]
    return capacity / ff_speed                          #[veh/km]

# Determine Density of each edge
def get_density(attrs: dict, t_min):
    flow = get_flow(attrs, t_min)                        #[veh/h]
    free_time = attrs.get("free_flow_time")/60           #[h]
    free_time = 1.6667 if free_time ==0 else free_time   # Neighboorhoodroads, dont have Free_flow_time in Data ---> ~30 km/h
    length = attrs.get("length") * 0.3048/1000           #[km]
    free_speed = (length / free_time)                    #[km/h] 
    density = flow / free_speed                          #[veh/h]
    return density                                       #[veh/h]

# Travel time bepalen
def get_travel_time(attrs: dict, t_min):
    capacity = attrs.get("capacity")                       #[veh/h]
    free_time_min = attrs.get("free_flow_time")            #[min]
    critical_density = get_critical_density(attrs)         #[veh/km]
    density = get_density(attrs,t_min)                     #[veh/km]
    flow = get_flow(attrs, t_min)                          #[veh/h]
    beta = 4
    B = 0.15
    travel_time = free_time_min * (1 + B* (flow/capacity)**beta)   #[min]
    return travel_time

# Determine the speed over each edge
def get_speed(attrs,t_min):
    length = attrs.get("length")* 0.3048                     #[m]
    travel_time = get_travel_time(attrs,t_min)  *60   #[s]
    return (length / travel_time)                            #[m/s]


# Congestiontime ---- Needed for Visualization
def congestion_time(attrs: dict, t_min):
    capacity = attrs.get("capacity")                       #[veh/h]
    free_time_min = attrs.get("free_flow_time")            #[min]
    critical_density = get_critical_density(attrs)         #[veh/km]
    density = get_density(attrs,t_min)                     #[veh/km]
    flow = get_flow(attrs, t_min)                          #[veh/h]
    beta = 4            #[veh/h]
    B = 0.15
    if density <= critical_density:
        return 0
    else:
        return free_time_min * B* (flow/capacity)**beta     #[min]


def get_node_sequence(graph: nx.DiGraph, end: int) -> list:
    path = []
    curr = end
    while curr is not None:
        path.append(curr)
        if "previous" not in graph.nodes[curr] or graph.nodes[curr]["previous"] is None:
            if curr != graph.nodes[curr].get("_dijkstra_source"):
                break
            else:
                break
        if curr == graph.nodes[curr]["previous"]:
            break
        curr = graph.nodes[curr]["previous"]
    if not path:
        return [end] if graph.nodes[end].get("_dijkstra_source") == end else []
    return path[::-1]

def dynamic_dijkstra(graph: nx.DiGraph, start: int, end: int, start_t: float) -> list:
    nx.set_node_attributes(graph, {"_dijkstra_source": start})
    for node in graph.nodes:
        graph.nodes[node]["arrival_time"] = math.inf
        graph.nodes[node]["previous"] = None
    if start not in graph or end not in graph:
        return []
    graph.nodes[start]["arrival_time"] = start_t
    Q = set(graph.nodes)
    while Q:
        u = min(Q, key=lambda n: graph.nodes[n]["arrival_time"])
        if graph.nodes[u]["arrival_time"] == math.inf:
            break
        if u == end:
            return get_node_sequence(graph, end)
        Q.remove(u)
        t_u = graph.nodes[u]["arrival_time"]
        for v in graph.neighbors(u):
            if v not in Q:
                continue
            edge_data = graph.edges[u, v]
            tt = get_travel_time(edge_data, t_u)
            alt = t_u + tt
            if alt < graph.nodes[v]["arrival_time"]:
                graph.nodes[v]["arrival_time"] = alt
                graph.nodes[v]["previous"] = u
    return []

def arrival_times_for_path(graph: nx.DiGraph, path: list, start_t: float) -> dict:
    times = {}
    t = start_t
    for u, v in pairwise(path):
        if not graph.has_edge(u, v):
            continue
        times[(u, v)] = t
        if graph.has_edge(v, u):
            times[(v, u)] = t
        t += get_travel_time(graph.edges[u, v], t)
    return times

# Function to assign distinct colors per route
def get_route_colors(num_routes):
    colormap = plt.cm.get_cmap('tab10', num_routes)
    return [colormap(i) for i in range(num_routes)]


if __name__ == "__main__":
    # Load Network data
    edges_df, nodes_df, trips_df, flow_df = read_folder(
        os.path.join(project_root, "TransportationNetworks", "Anaheim")
    )

    # Build directed and undirected graphs
    G_directed = graph_from_data(edges_df, nodes_df)
    undirected_graph = nx.Graph()
    for node, data in G_directed.nodes(data=True):
        undirected_graph.add_node(node, **data)
    for u, v, data in G_directed.edges(data=True):
        if not undirected_graph.has_edge(u, v):
            undirected_graph.add_edge(u, v, **data)

    # Populate volume from flow_df onto undirected_graph
    for row in flow_df.to_dict("records"):
        if undirected_graph.has_node(row["from"]) and undirected_graph.has_node(row["to"]):
            if undirected_graph.has_edge(row["from"], row["to"]):
                undirected_graph.edges[row["from"], row["to"]]["volume"] = row["volume"]

    # Coordinates for plotting
    pos = {n: data["coordinates"] for n, data in undirected_graph.nodes(data=True)}
    edges = list(undirected_graph.edges())

    # Filter out any customer IDs not present
    customer_node_ids = [nid for nid in customer_node_ids if undirected_graph.has_node(nid)]
    if not undirected_graph.has_node(depot_node_id):
        raise ValueError(f"Depot node {depot_node_id} not in graph.")
    if len(customer_node_ids) < 1:
        raise ValueError("No customer nodes defined or found in graph.")

    # BSO-LNS setup: map from BSO indices → actual node IDs
    bso_nodes_map = [depot_node_id] + customer_node_ids
    customer_demands = n_demand

    # Plot setup: draw nodes & edges once
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_nodes(
        undirected_graph, pos,
        node_size=1, ax=ax, node_color='gray'
    )
    nx.draw_networkx_nodes(
        G_directed, pos,
        nodelist=customer_node_ids,
        node_size=20, ax=ax, node_color='blue'
    )
    nx.draw_networkx_nodes(
        G_directed, pos,
        nodelist=[depot_node_id],
        node_size=20, ax=ax, node_color='red'
    )
    drawn = nx.draw_networkx_edges(
        undirected_graph, pos,
        edgelist=edges, edge_color="0.8", ax=ax
    )

    title = ax.set_title("t=00:00")
    timer_text = ax.text(
        0.02, 0.95,
        "Elapsed: 00:00:00",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    # Travel time wrapper for BSO using dynamic Dijkstra + memoization
    memoized_travel_times = {}
    def td_travel_time_wrapper(u_bso_idx, v_bso_idx, depart_t):
        u_actual = bso_nodes_map[u_bso_idx]
        v_actual = bso_nodes_map[v_bso_idx]
        if u_actual == v_actual:
            return 0.0
        cache_key = (u_actual, v_actual, depart_t)
        if cache_key in memoized_travel_times:
            return memoized_travel_times[cache_key]
        path_nodes = dynamic_dijkstra(undirected_graph, u_actual, v_actual, depart_t)
        if not path_nodes or path_nodes[-1] != v_actual:
            return float('inf')
        arrival_at_v = undirected_graph.nodes[v_actual]["arrival_time"]
        duration = arrival_at_v - depart_t
        memoized_travel_times[cache_key] = duration
        return duration

    # Run BSO-LNS solver
    bso_solver = BSOLNS(
        travel_time_fn=td_travel_time_wrapper,
        demands=customer_demands,
        vehicle_capacity=vehicle_capacity,
        start_time=route_start_t,
        pop_size=pop_size,
        n_clusters=n_clusters,
        ideas_per_cluster=ideas_per_cluster,
        max_iter=max_iter,
        remove_rate=remove_rate
    )
    best_solution = bso_solver.run()
    print(f"BSO-LNS final best cost: {best_solution['cost']:.2f}, Routes: {best_solution['sol']}")

    # Reconstruct the BSO solution path for animation
    bso_solution_routes = best_solution['sol']
    bso_edges_for_animation = {}

    # FIXED: use a fresh t = route_start_t per route, store both (u,v) and (v,u)
    for route in bso_solution_routes:
        t = route_start_t  # each vehicle leaves at 15:30
        current_node = depot_node_id

        # From depot → each customer in route
        for cust_bso_idx in route:
            dest_node = bso_nodes_map[cust_bso_idx]
            path_nodes = dynamic_dijkstra(undirected_graph, current_node, dest_node, t)
            if not path_nodes:
                break
            seg_times = arrival_times_for_path(undirected_graph, path_nodes, t)
            for (u, v), entry in seg_times.items():
                bso_edges_for_animation[(u, v)] = entry
                bso_edges_for_animation[(v, u)] = entry
            t = undirected_graph.nodes[dest_node]["arrival_time"]
            current_node = dest_node

        # Final leg: last customer → depot
        if current_node != depot_node_id:
            path_back = dynamic_dijkstra(undirected_graph, current_node, depot_node_id, t)
            if path_back:
                seg_times = arrival_times_for_path(undirected_graph, path_back, t)
                for (u, v), entry in seg_times.items():
                    bso_edges_for_animation[(u, v)] = entry
                    bso_edges_for_animation[(v, u)] = entry

    # Update function for animation
    def update_frame(frame_minutes_offset):
        current_sim_time = sim_start + frame_minutes_offset

        # Update title & timer text
        h, m = divmod(current_sim_time, 60)
        title.set_text(f"Time: {h:02}:{m:02}")
        hrs = frame_minutes_offset // 60
        mins = frame_minutes_offset % 60
        timer_text.set_text(f"Elapsed: {hrs:02}:{mins:02}:00")

        # Compute congestion-based grayscale for every undirected edge
        added_travel_times = [
            congestion_time(undirected_graph.edges[e], current_sim_time)
            for e in edges
        ]
        base_intensities = [
            str(max(0.9 / 15 * (15 - tau), 0.0))
            for tau in added_travel_times
        ]

        # Assign distinct colors per route
        num_routes = len(bso_solution_routes)
        route_colors = get_route_colors(num_routes)

        # Build edge_colors (one color per edge)
        edge_colors = []
        for i, (u, v) in enumerate(edges):
            t_uv = bso_edges_for_animation.get((u, v), float("inf"))
            t_vu = bso_edges_for_animation.get((v, u), float("inf"))
            route_color = None
            for route_idx in range(num_routes):
                if t_uv <= current_sim_time or t_vu <= current_sim_time:
                    route_color = route_colors[route_idx]
                    break
            if route_color is not None:
                edge_colors.append(route_color)
            else:
                edge_colors.append(base_intensities[i])

        # Update edge colors
        drawn.set_edgecolor(edge_colors)
        return [drawn, title, timer_text]

    # Create and show animation
    ani = animation.FuncAnimation(
        fig,
        update_frame,
        frames=range(0, 24 * 60 - sim_start, 15),
        interval=200,
        blit=True
    )

    ax.set_aspect('equal')
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.tight_layout()
    plt.show()

u, v = edge_example

# Tabel with sample times
if undirected_graph.has_edge(u, v):
    edge_attrs = undirected_graph.edges[u, v]
    records = []

    for t in [t1, t2, t3, t4, t5, t6, t7, t8]:
        records.append({
            "Time (min)": t,
            "Flow (veh/h)": get_flow(edge_attrs, t),
            "Critical Density (veh/km)": get_critical_density(edge_attrs),
            "Travel Time (min)": get_travel_time(edge_attrs, t),
            "Speed in m/s" : get_speed(edge_attrs, t),
            "Density(veh/km)" : get_density(edge_attrs,t),
            "Capacity" : get_capacity(edge_attrs,t)
        })

    df = pd.DataFrame(records)
    print(df)

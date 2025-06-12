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
from GA_imp import GA_DP 
import mplcursors
import time

start_script = time.time()

# Define Problem: Depot and Customers
depot_node_id = 406   
customer_node_ids = [386 ,370 , 17, 267 ,303, 321 ,305 ,308, 342, 400, 6, 372, 358,  300, 404, 333, 390, 369, 325, 388]
#[386,370, 17, 303 ,305 ,342,400,372,358, 404, 333 ,390 ,369]
time_step_minutes = 10  # mins
sim_start = 0 * 60      # (0 means midnight)
route_start_t = 12 * 60   # 00:30 (in minutes)
num_vehicles = 4
n_demand = [1] * len(customer_node_ids)
demands_dict = {customer_node_ids[i]: n_demand[i] for i in range(len(customer_node_ids))}
total_demand = sum(n_demand)
vehicle_capacity = math.ceil(total_demand / num_vehicles)
edge_example = (12, 275)

# Time-breakpoints demand function
t1, t2, t3, t4 = 6.5 * 60, 8.5 * 60, 10 * 60, 12 * 60
t5, t6, t7, t8 = 16.5 * 60, 18 * 60, 20 * 60, 22 * 60
period_breaks = sorted([0, t1, t2, t3, t4, t5, t6, t7, t8, 24*60])

# Parameters for GA
pop_size = 2000
max_gens = 500
tournament_size = 2    
crossover_rate = 0.9
mutation_rate = 0.2
elite_count = 2
start_time = route_start_t  # Make DP depart at 00:30

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

# Travel-distance function (shortest-path in km)
def travel_distance_fn(u, v):
    path_nodes = nx.shortest_path(undirected_graph, source=u, target=v, weight="length")
    dist_m = sum(
        undirected_graph.edges[path_nodes[i], path_nodes[i+1]]["length"] * 0.3048
        for i in range(len(path_nodes)-1)
    )
    return dist_m / 1000.0  # kilometers

# --- Congestion Model Based on Demand ---
def get_flow(attrs: dict, t_min: float) -> float:
    volume = attrs.get("volume", 0)
    return volume * demand(t_min) * 10 

def get_capacity(attrs: dict, t_min: float) -> float:
    return attrs.get("capacity") 

def get_critical_density(attrs: dict):
    capacity = attrs.get("capacity")                    # [veh/h]
    free_time = attrs.get("free_flow_time") / 60        # [h]
    free_time = 1.667 if free_time == 0 else free_time   # default ~30 km/h
    length = attrs.get("length") * 0.3048 / 1000         # [km]
    ff_speed = length / free_time                        # [km/h]
    return capacity / ff_speed                           # [veh/km]

def get_density(attrs: dict, t_min):
    flow = get_flow(attrs, t_min)                        # [veh/h]
    free_time = attrs.get("free_flow_time") / 60         # [h]
    free_time = 1.6667 if free_time == 0 else free_time   # default ~30 km/h
    length = attrs.get("length") * 0.3048 / 1000         # [km]
    free_speed = length / free_time                      # [km/h]
    return flow / free_speed                             # [veh/km]

def get_travel_time(attrs: dict, t_min):
    capacity = attrs.get("capacity")                    # [veh/h]
    free_time_min = attrs.get("free_flow_time")         # [min]
    density = get_density(attrs, t_min)                 # [veh/km]
    flow = get_flow(attrs, t_min)                       # [veh/h]
    beta = 4
    B = 0.15
    return free_time_min * (1 + B * (flow / capacity) ** beta)  # [min]

def get_speed(attrs: dict, t_min):
    length_m = attrs.get("length") * 0.3048               # [m]
    travel_time_s = get_travel_time(attrs, t_min) * 60   # [s]
    return length_m / travel_time_s                      # [m/s]

def congestion_time(attrs: dict, t_min):
    capacity = attrs.get("capacity")                       # [veh/h]
    free_time_min = attrs.get("free_flow_time")            # [min]
    flow = get_flow(attrs, t_min)                          # [veh/h]
    beta = 4
    B = 0.15
    return free_time_min * (B * (flow / capacity) ** beta)  # [min]

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

    # Travel time wrapper for GA using dynamic Dijkstra + memoization
    memoized_travel_times = {}
    def td_travel_time_wrapper(u, v, depart_t):
        if u == v:
            return 0.0
        cache_key = (u, v, depart_t)
        if cache_key in memoized_travel_times:
            return memoized_travel_times[cache_key]
        path_nodes = dynamic_dijkstra(undirected_graph, u, v, depart_t)
        if not path_nodes or path_nodes[-1] != v:
            return float("inf")
        arrival_at_v = undirected_graph.nodes[v]["arrival_time"]
        duration = arrival_at_v - depart_t
        memoized_travel_times[cache_key] = duration
        return duration

    # Instantiate GA_DP, passing depot_node_id
    ga_solver = GA_DP(
        travel_time_fn=td_travel_time_wrapper,
        demands_dict=demands_dict,
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity,
        period_breaks=period_breaks,
        time_windows={},
        pop_size=pop_size,
        max_gens=max_gens,
        tournament_size=tournament_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        elite_count=elite_count,
        start_time=route_start_t,
        depot_node_id=depot_node_id
    )

    # Run the GA_DP solver
    (giant_tour, split_indices), ga_cost = ga_solver.run()
    print(f"GA_DP final cost: {ga_cost:.2f}, giant_tour: {giant_tour}, splits: {split_indices}")

    # Immediately after GA finishes, stop the timer:
    end_time = time.time()
    runtime_seconds = end_time - start_script

    # Split the giant_tour into per-vehicle routes
    routes = []
    prev = 0
    for split in split_indices:
        routes.append(giant_tour[prev:split])
        prev = split
    routes.append(giant_tour[prev:])
    while len(routes) < num_vehicles:
        routes.append([])

    print("giant_tour:", giant_tour)
    print("split_indices:", split_indices)
    print("routes:", routes)

    # Reconstruct each route’s edge-entry times and store (time, route_idx)
    ga_edges_for_animation = {}
    for route_idx, route in enumerate(routes):
        t = route_start_t
        current_node = depot_node_id

        for cust_node in route:
            path_nodes = dynamic_dijkstra(undirected_graph, current_node, cust_node, t)
            if not path_nodes:
                break
            seg_times = arrival_times_for_path(undirected_graph, path_nodes, t)
            for (u, v), entry in seg_times.items():
                ga_edges_for_animation[(u, v)] = (entry, route_idx)
                ga_edges_for_animation[(v, u)] = (entry, route_idx)
            t = undirected_graph.nodes[cust_node]["arrival_time"]
            current_node = cust_node

        if current_node != depot_node_id:
            path_back = dynamic_dijkstra(undirected_graph, current_node, depot_node_id, t)
            if path_back:
                seg_times = arrival_times_for_path(undirected_graph, path_back, t)
                for (u, v), entry in seg_times.items():
                    ga_edges_for_animation[(u, v)] = (entry, route_idx)
                    ga_edges_for_animation[(v, u)] = (entry, route_idx)

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
        num_routes = len(routes)
        route_colors = get_route_colors(num_routes)

        # Build edge_colors (one color per edge)
        edge_colors = []
        for i, (u, v) in enumerate(edges):
            entry_route_uv = ga_edges_for_animation.get((u, v), None)
            entry_route_vu = ga_edges_for_animation.get((v, u), None)

            chosen_color = None
            if entry_route_uv is not None:
                entry_time, route_idx = entry_route_uv
                if entry_time <= current_sim_time:
                    chosen_color = route_colors[route_idx]
            if chosen_color is None and entry_route_vu is not None:
                entry_time, route_idx = entry_route_vu
                if entry_time <= current_sim_time:
                    chosen_color = route_colors[route_idx]

            if chosen_color is not None:
                edge_colors.append(chosen_color)
            else:
                edge_colors.append(base_intensities[i])

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



# --- Save results to Excel in the specified format ---
from openpyxl import load_workbook, Workbook
from openpyxl.utils.dataframe import dataframe_to_rows


def save_results(ga_cost, runtime_seconds, route_start_t, num_vehicles):
    """Save results to Excel in the specified format"""
    results_path = os.path.join(os.getcwd(), "results.xlsx")
    
    # Convert sim_start to hours:minutes format
    hours = route_start_t // 60
    minutes = route_start_t % 60
    route_start_t_label = f"route_start_t = {hours}:{minutes:02d}"

    
    try:
        if os.path.exists(results_path):
            df = pd.read_excel(results_path)
        else:
            df = pd.DataFrame()
        
        # Create new row
        new_data = {
            'route_start_t': route_start_t_label,
            'num_vehicles':num_vehicles,
            'test': len(df) + 1,
            'traveltime': ga_cost,
            'runtime': runtime_seconds
        }
        
        new_row = pd.DataFrame([new_data])
        df = pd.concat([df, new_row], ignore_index=True)
        
        df.to_excel(results_path, index=False)
        print(f"✅ Results saved: {route_start_t_label}, Test {len(df)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

# FIXED: Only call the function once
try:
    save_results(ga_cost, runtime_seconds, route_start_t,num_vehicles)
except Exception as e:
    print(f"Error saving results: {e}")
    # Simple fallback - just print the results
    print(f"Results (not saved): Travel Time: {ga_cost:.2f}, Runtime: {runtime_seconds:.2f}")

''' 
    # Sample times table for edge_example
    u, v = edge_example
    if undirected_graph.has_edge(u, v):
        edge_attrs = undirected_graph.edges[u, v]
        records = []
        for t in [t1, t2, t3, t4, t5, t6, t7, t8]:
            records.append({
                "Time (min)": t,
                "Flow (veh/h)": get_flow(edge_attrs, t),
                "Critical Density (veh/km)": get_critical_density(edge_attrs),
                "Travel Time (min)": get_travel_time(edge_attrs, t),
                "Speed in m/s": get_speed(edge_attrs, t),
                "Density(veh/km)": get_density(edge_attrs, t),
                "Capacity": get_capacity(edge_attrs, t)
            })
        df = pd.DataFrame(records)
        print(df)
'''


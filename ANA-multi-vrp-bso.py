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
import vrp_sim as vs
from read_cities import read_anaheim

# Define BSO-LNS Problem: Depot and Customers
depot_node_id = 406   
customer_node_ids = [386 ,370 , 17 ,267 ,303, 321,305]
sim_start = 6 * 60 * 60  # 6:00
route_start_t = (15 * 60 + 30)*60  # 15:30 (in seconds)
num_vehicles = 2
n_demand = [1] * len(customer_node_ids)  #Demand per customer
total_demand = sum(n_demand)
vehicle_capacity = math.ceil(total_demand / num_vehicles)
edge_example = 12 ,275 


# Time-breakpoints demand function
t1, t2, t3, t4 = 6.5 * 60, 8.5 * 60, 10 * 60, 12 * 60
t5, t6, t7, t8 = 16.5 * 60, 18 * 60, 20 * 60, 22 * 60

# Parameters for BSO
pop_size = 20
n_clusters = 3
ideas_per_cluster = 2
max_iter = 1
remove_rate = 0.5

def arrival_times_for_path(graph: nx.DiGraph, path: list, start_t: float) -> dict:
    times = {}
    t = start_t
    for u, v in pairwise(path):
        if not graph.has_edge(u, v):
            continue
        times[(u, v)] = t
        if graph.has_edge(v, u):
            times[(v, u)] = t
        t += sim._get_edge_travel_time(u, v, t)
    return times

# Function to assign distinct colors per route
def get_route_colors(num_routes):
    colormap = plt.cm.get_cmap('tab10', num_routes)
    return [colormap(i) for i in range(num_routes)]


if __name__ == "__main__":
    # Load Network data
    edges_df, nodes_df, trips_df, flow_df = read_anaheim()

    sim = vs.TrafficSim(edges_df, flow_df, nodes=nodes_df)

    # Coordinates for plotting
    pos = {n: data["coordinates"] for n, data in sim.G.nodes(data=True)}
    edges = list(sim.G.edges())

    # Filter out any customer IDs not present
    customer_node_ids = [nid for nid in customer_node_ids if sim.G.has_node(nid)]
    if not sim.G.has_node(depot_node_id):
        raise ValueError(f"Depot node {depot_node_id} not in graph.")
    if len(customer_node_ids) < 1:
        raise ValueError("No customer nodes defined or found in graph.")

    # BSO-LNS setup: map from BSO indices → actual node IDs
    bso_nodes_map = [depot_node_id] + customer_node_ids
    customer_demands = n_demand

    # Plot setup: draw nodes & edges once
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_nodes(
        sim.G, pos,
        node_size=1, ax=ax, node_color='gray'
    )
    nx.draw_networkx_nodes(
        sim.G, pos,
        nodelist=customer_node_ids,
        node_size=20, ax=ax, node_color='blue'
    )
    nx.draw_networkx_nodes(
        sim.G, pos,
        nodelist=[depot_node_id],
        node_size=20, ax=ax, node_color='red'
    )
    drawn = nx.draw_networkx_edges(
        sim.G, pos,
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
        path_nodes, duration = sim._dynamic_dijkstra(u_actual, v_actual, depart_t)
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
            path_nodes, duration = sim._dynamic_dijkstra(current_node, dest_node, t)
            if not path_nodes:
                break
            seg_times = arrival_times_for_path(sim.G, path_nodes, t)
            for (u, v), entry in seg_times.items():
                bso_edges_for_animation[(u, v)] = entry
                bso_edges_for_animation[(v, u)] = entry
            t += duration
            current_node = dest_node

        # Final leg: last customer → depot
        if current_node != depot_node_id:
            path_back, _ = sim._dynamic_dijkstra(current_node, depot_node_id, t)
            if path_back:
                seg_times = arrival_times_for_path(sim.G, path_back, t)
                for (u, v), entry in seg_times.items():
                    bso_edges_for_animation[(u, v)] = entry
                    bso_edges_for_animation[(v, u)] = entry

    # Update function for animation
    def update_frame(frame_minutes_offset):
        current_sim_time = (sim_start + frame_minutes_offset)*60

        # Update title & timer text
        h, m = divmod(current_sim_time, 60)
        title.set_text(f"Time: {h:02}:{m:02}")
        hrs = frame_minutes_offset // 60
        mins = frame_minutes_offset % 60
        timer_text.set_text(f"Elapsed: {hrs:02}:{mins:02}:00")

        # Compute congestion-based grayscale for every undirected edge
        added_travel_times = [
            sim.get_edge_congestion_time(source, dest, current_sim_time) / sim.G.edges[source, dest]["free_flow_time"]
            for source, dest in edges
        ]
        base_intensities = [
            str(max(0.9 * (1 - tau), 0.0))
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
        frames=range(0, 24 * 60 - sim_start // 60, 15),
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
if sim.G.has_edge(u, v):
    edge_attrs = sim.G.edges[u, v]
    records = []

    for t in [t1, t2, t3, t4, t5, t6, t7, t8]:
        records.append({
            "Time (min)": t,
            "Flow (veh/s)": sim._get_flow(u,v, t*60),
            "Congestion time": sim.get_edge_congestion_time(u, v, t*60),
            "Capacity" : edge_attrs["capacity"],
            "demand factor" : sim._demand(t*60),
            "Free flow" : edge_attrs["free_flow_time"]
        })

    df = pd.DataFrame(records)
    print(df)
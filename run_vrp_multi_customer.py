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
depot_node_id = 1 #918 
customer_node_ids = [1,4,5,50,60,70,80,90,100, 911, 210, 350, 123, 456,300] #,300, 400, 500, 200, 100]
time_step_minutes = 10 #mins
sim_start = 6 * 60 #6:00
route_start_t = 12 * 60  #15:30
num_vehicles = 4
n_demand = [1] * len(customer_node_ids)# Demand per customer
total_demand = sum(n_demand)
vehicle_capacity = math.ceil(total_demand/num_vehicles)

# Time-breakpoints demand function
t1, t2, t3, t4 = 6.5*60, 8.5*60, 10*60, 12*60
t5, t6, t7, t8 = 16.5*60, 18*60, 20*60, 22*60



# Parameters for BSO
pop_size= 10           
n_clusters = 3
ideas_per_cluster = 5
max_iter = 1            
remove_rate=0.3

# Demands for each customer (must match the order in customer_node_ids)
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
        for u, v in pairwise(path))

# Demand function
def demand(t: float) -> float:
    low, medium, high = 0.1, 0.5, 5
    if t <= t1:
        return low
    elif t < t2:
        return low + (high-low)*(t-t1)/(t2-t1)
    elif t < t3:
        return high
    elif t <= t4:
        return high - (high-medium)*(t-t3)/(t4-t3)
    elif t <= t5:
        return medium
    elif t < t6:
        return medium + (high-medium)*(t-t5)/(t6-t5)
    elif t <= t7:
        return high
    elif t < t8:
        return low + (high-low)*(1 - (t-t7)/(t8-t7))
    else:
        return low


# --- Congestion Model Based on Demand ---
def get_flow(attrs: dict, t_min: float) -> float:
    volume = attrs.get("volume")
    return volume * demand(t_min)

# Determine Critical Density for each edge
def get_critical_density(attrs: dict):
    capacity = attrs.get("capacity")
    free_time = attrs.get("free_flow_time")
    free_time = 5 if free_time == 0 else free_time
    length = attrs.get("length")
    ff_speed = length / free_time
    return capacity / ff_speed #klopt dit??

# Determine Density
'''
def get_density(attrs: dict):
    length = attrs.get("length")
    capacity = attrs.get("capacity")
    density = capacity / length
    return density 
'''

# Determine Congestion speed 
def congestion_speed(attrs: dict):
    capacity  = attrs.get("capacity")
    length = attrs.get("length")
    pc = get_critical_density(attrs) 
    pj = 5 * pc # ???? hoe bereken/bepaal je dit ???
    w = capacity / (pj - pc) # Congestion speed
    return w

# Needed foir Visualization
def congestion_time(attrs: dict, t_min):
    flow = get_flow(attrs, t_min)
    return flow / 1500


# Travel time bepalen
def get_travel_time(attrs: dict, t_min):
    freetime = attrs.get("free_flow_time")
    capacity  = attrs.get("capacity")
    flow = get_flow(attrs, t_min)
    if flow <= capacity:
        travel_time = freetime
    else:
        travel_time = freetime + flow / 1500    #length / w
    return travel_time

def get_node_sequence(graph: nx.DiGraph, end: int) -> list:
    path = []
    curr = end
    while curr is not None:
        path.append(curr)
        if "previous" not in graph.nodes[curr] or graph.nodes[curr]["previous"] is None:
            if curr != graph.nodes[curr].get("_dijkstra_source"): # check if it's the source
                 break
            else: # Reached source
                break
        if curr == graph.nodes[curr]["previous"]: # Stuck in a loop
            break
        curr = graph.nodes[curr]["previous"]
    if not path: return [end] if graph.nodes[end].get("_dijkstra_source") == end else []

    return path[::-1]


def dynamic_dijkstra(graph: nx.DiGraph, start: int, end: int, start_t: float) -> list:
    nx.set_node_attributes(graph, {"_dijkstra_source": start})
    for node in graph.nodes:
        graph.nodes[node]["arrival_time"] = math.inf
        graph.nodes[node]["previous"] = None
    if start not in graph:
        print(f"Error: Start node {start} not in graph for Dijkstra.")
        return []
    if end not in graph:
        print(f"Error: End node {end} not in graph for Dijkstra.")
        return []

    graph.nodes[start]["arrival_time"] = start_t

    Q = set(graph.nodes) # Priority queue elements
    
    # For actual priority queue behavior, typically a min-heap is used.
    # This simplified version rrates to find min, which is less efficient for large graphs
    # but fine for this context if graph isn't excessively large or Dijkstra isn't called too often.
    
    while Q:
        # Find u in Q with smallest arrival_time
        u = min(Q, key=lambda n: graph.nodes[n]["arrival_time"])
        if graph.nodes[u]["arrival_time"] == math.inf:
            break
        if u == end:
            return get_node_sequence(graph, end)
        Q.remove(u)
        t_u = graph.nodes[u]["arrival_time"]
        # Neighbors of u that are still in Q
        for v in graph.neighbors(u): 
            if v not in Q: # Already processed or not in graph
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
        if not graph.has_edge(u,v):
            print(f"Warning: Edge ({u}, {v}) not found in graph during arrival_times_for_path.")
            continue 
        times[(u, v)] = t
        if graph.has_edge(v,u):
            times[(v, u)] = t 
        t += get_travel_time(graph.edges[u, v], t)
    return times


# -------Animation Update Functions---------
if __name__ == "__main__":
    # Load Network
    edges_df, nodes_df, trips_df, flow_df = read_folder(os.path.join(project_root, "TransportationNetworks", "Chicago-Sketch")) #Anaheim
    
    G_directed = graph_from_data(edges_df, nodes_df)
    undirected_graph = nx.Graph() # Start with an empty graph

    for node, data in G_directed.nodes(data=True):
        undirected_graph.add_node(node, **data)
    for u, v, data in G_directed.edges(data=True):
        if not undirected_graph.has_edge(u,v): 
             undirected_graph.add_edge(u, v, **data)


    for row in flow_df.to_dict("records"):
        # Ensure from/to nodes exist and edge exists before adding volume
        if undirected_graph.has_node(row["from"]) and undirected_graph.has_node(row["to"]):
            if undirected_graph.has_edge(row["from"], row["to"]):
                 undirected_graph.edges[row["from"], row["to"]]["volume"] = row["volume"]
            # else: print(f"Warning: Edge {row['from']}-{row['to']} not in graph for flow data.")

    pos = {n: data["coordinates"] for n, data in undirected_graph.nodes(data=True)}
    edges = list(undirected_graph.edges())

    # Plot setup
    fig, ax = plt.subplots(figsize=(10,8))
    nx.draw_networkx_nodes(undirected_graph, pos, node_size=1, ax=ax, node_color='gray')
    nx.draw_networkx_nodes(G_directed, pos, nodelist= customer_node_ids, node_size=20, ax=ax, node_color='blue')
    nx.draw_networkx_nodes(G_directed, pos, nodelist= [depot_node_id], node_size=20, ax=ax, node_color='red')
    drawn = nx.draw_networkx_edges(undirected_graph, pos, edgelist=edges, edge_color="0.8", ax=ax)
    
    

    # Ensure these nodes exist in your graph:
    customer_node_ids = [nid for nid in customer_node_ids if undirected_graph.has_node(nid)]
    if not undirected_graph.has_node(depot_node_id):
        raise ValueError(f"Depot node {depot_node_id} not in graph.")
    if len(customer_node_ids) < 1:
        raise ValueError("No customer nodes defined or found in graph.")

    # `bso_nodes` maps BSOLNS internal indices (0 for depot, 1..N for customers) to actual graph node IDs
    bso_nodes_map = [depot_node_id] + customer_node_ids 
    num_bso_customers = len(customer_node_ids)
    customer_demands = n_demand


    # Title and timer text
    title = ax.set_title("Time: 00:00")
    timer_text = ax.text(
        0.02, 0.95,             # x, y in axes coords
        "Elapsed: 00:00:00",   # initial text
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    start_min = int(1*60)

    title = ax.set_title("t=00:00")

    # ── Travel time function for BSOLNS ───────────────────────────────
    memoized_travel_times = {} # Cache for (u_node, v_node, depart_t) -> travel_duration
    
    def td_travel_time_wrapper(u_bso_idx, v_bso_idx, depart_t):
        """
        BSOLNS internal indices (u_bso_idx, v_bso_idx) to actual graph node IDs.
        u_bso_idx=0 is depot, u_bso_idx=1 is first customer, etc.
        v_bso_idx=0 is depot, v_bso_idx=1 is first customer, etc.
        """
        u_actual_node = bso_nodes_map[u_bso_idx]
        v_actual_node = bso_nodes_map[v_bso_idx]

        if u_actual_node == v_actual_node: # Travel to self is 0
            return 0.0

        # Memoization key
        # Discretize depart_t to reduce cache size if times are very granular.
        # E.g., round depart_t to nearest minute or second. For now, use exact.
        cache_key = (u_actual_node, v_actual_node, depart_t)
        if cache_key in memoized_travel_times:
            return memoized_travel_times[cache_key]

        # Run dynamic Dijkstra on the full graph (undirected_graph in this setup)
        path_nodes = dynamic_dijkstra(undirected_graph, u_actual_node, v_actual_node, depart_t)
        
        if not path_nodes or path_nodes[-1] != v_actual_node:
            # print(f"Warning: No path found from {u_actual_node} to {v_actual_node} at {depart_t}.")
            return float('inf') # No path found or error

        arrival_at_v = undirected_graph.nodes[v_actual_node]["arrival_time"]
        duration = arrival_at_v - depart_t
        
        memoized_travel_times[cache_key] = duration
        return duration

    # Run BSO-LNS
    bso_solver = BSOLNS(
        travel_time_fn=td_travel_time_wrapper,
        demands=customer_demands,
        vehicle_capacity=vehicle_capacity,
        start_time= route_start_t,
        pop_size= pop_size,           
        n_clusters= n_clusters, 
        ideas_per_cluster = ideas_per_cluster,
        max_iter = max_iter,
        remove_rate = remove_rate)

    best_solution = bso_solver.run()
    print(f"BSO-LNS final best cost: {best_solution['cost']:.2f}, Routes: {best_solution['sol']}")

    # Reconstruct the BSO solution path for animation
    bso_solution_routes = best_solution['sol'] # List of routes
    bso_edges_for_animation = {} # Stores {(u,v): entry_time} for all segments in BSO solution

    if bso_solution_routes: 
        for route_of_customer_indices in bso_solution_routes:
            if not route_of_customer_indices:
                continue

            # Start from depot for this route
            current_leg_start_node_id = depot_node_id
            
            # Leg 1: Depot to first customer in current route
            first_cust_bso_idx = route_of_customer_indices[0]
            first_cust_actual_node_id = bso_nodes_map[first_cust_bso_idx]
            
            path_segment_nodes = dynamic_dijkstra(undirected_graph, current_leg_start_node_id, first_cust_actual_node_id, route_start_t)
            if path_segment_nodes:
                segment_edge_times = arrival_times_for_path(undirected_graph, path_segment_nodes, route_start_t)
                bso_edges_for_animation.update(segment_edge_times)
                route_start_t = undirected_graph.nodes[first_cust_actual_node_id]["arrival_time"]
                current_leg_start_node_id = first_cust_actual_node_id

            # Legs between customers in the current route
            for i in range(len(route_of_customer_indices) - 1):
                next_cust_bso_idx = route_of_customer_indices[i+1]
                next_cust_actual_node_id = bso_nodes_map[next_cust_bso_idx]
                
                path_segment_nodes = dynamic_dijkstra(undirected_graph, current_leg_start_node_id, next_cust_actual_node_id, route_start_t)
                if path_segment_nodes:
                    segment_edge_times = arrival_times_for_path(undirected_graph, path_segment_nodes, route_start_t)
                    bso_edges_for_animation.update(segment_edge_times)
                    route_start_t = undirected_graph.nodes[next_cust_actual_node_id]["arrival_time"]
                    current_leg_start_node_id = next_cust_actual_node_id
            
            # Final Leg: Last customer in route back to Depot
            if path_segment_nodes: # Ensure previous leg was successful
                path_segment_nodes = dynamic_dijkstra(undirected_graph, current_leg_start_node_id, depot_node_id, route_start_t)
                if path_segment_nodes:
                    segment_edge_times = arrival_times_for_path(undirected_graph, path_segment_nodes, route_start_t)
                    bso_edges_for_animation.update(segment_edge_times)
                    route_start_t = undirected_graph.nodes[depot_node_id]["arrival_time"]

     # Assign routes a color
def get_route_colors(num_routes):
    colormap = plt.cm.get_cmap('tab10', num_routes)  # 'tab10' colormap with distinct colors
    colors = [colormap(i) for i in range(num_routes)]  
    return colors

# Update animation function
def update_frame(frame_minutes_offset):
    current_sim_time = sim_start + frame_minutes_offset
    
    # Add Timer
    h, m = divmod(current_sim_time, 60)
    title.set_text(f"Time: {h:02}:{m:02}")
    hrs = frame_minutes_offset // 60
    mins = frame_minutes_offset % 60
    secs = 0  # since frame steps in minutes
    timer_text.set_text(f"Elapsed: {hrs:02}:{mins:02}:{secs:02}")

    # Get the number of routes and assign unique colors **(Added)**
    num_routes = len(bso_solution_routes)
    route_colors = get_route_colors(num_routes)  # **(Added)** Get unique colors for each route

    # Congestion coloring
    added_travel_times = [congestion_time(undirected_graph.edges[e], current_sim_time) for e in edges]
    base_intensities = [str(max(0.9/15 * (15 - tau), 0)) for tau in added_travel_times] 

    edge_colors = []  # This will store the color for each edge in the plot

    for i, edge_tuple in enumerate(edges):
        u, v = edge_tuple 
        edge_key_forward = (u, v)
        edge_key_backward = (v, u)

        route_color = None  # **(Added)** Initialize route color as None

        # Check if the edge is part of any route and assign a unique color **(Modified)**
        for route_idx, route_of_customer_indices in enumerate(bso_solution_routes):
            if (edge_key_forward in bso_edges_for_animation and bso_edges_for_animation[edge_key_forward] <= current_sim_time) or \
               (edge_key_backward in bso_edges_for_animation and bso_edges_for_animation[edge_key_backward] <= current_sim_time):
                route_color = route_colors[route_idx]  # **(Modified)** Assign color for the route
                break

        if route_color:  # **(Modified)** Check if the route_color is set
            edge_colors.append(route_color)  # Add the unique route color
        else:
            edge_colors.append(base_intensities[i])  # Default congestion color

    # Update the edge colors in the plot
    title.set_text(f"t={int(current_sim_time)//60:02d}:{int(current_sim_time)%60:02d}")
    drawn.set_color(edge_colors)  # Set the colors for the edges
    return [drawn, title, timer_text]

# Create and show the animation
ani = animation.FuncAnimation(
    fig,
    update_frame,
    frames=range(sim_start, 24*60, 15),  # Simulate for a full day (in minutes)
    interval=200,  # Milliseconds between frames
    blit=True
)

ax.set_aspect('equal')
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.tight_layout()
plt.show()

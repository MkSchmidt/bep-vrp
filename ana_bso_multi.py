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
import vrp_sim as vs
from read_cities import read_anaheim
from plot_solution import plot_solution
import time
from export_excel import save_results

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

#Default Parameters for BSO
default_params = {
"pop_size": 100,
"max_iter": 100,
"n_clusters": 3,
"ideas_per_cluster": 2,
"remove_rate": 0.5,}

# Load Network data
edges_df, nodes_df, trips_df, flow_df = read_anaheim()

sim = vs.TrafficSim(edges_df, flow_df, nodes=nodes_df)

# Filter out any customer IDs not present
customer_node_ids = [nid for nid in customer_node_ids if sim.G.has_node(nid)]
if not sim.G.has_node(depot_node_id):
    raise ValueError(f"Depot node {depot_node_id} not in graph.")
if len(customer_node_ids) < 1:
    raise ValueError("No customer nodes defined or found in graph.")

# BSO-LNS setup: map from BSO indices → actual node IDs
customer_demands = n_demand
bso_nodes_map = [depot_node_id] + customer_node_ids

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


def run_bso(route_start_t, vehicle_capacity,customer_demands, pop_size, n_clusters,ideas_per_cluster, max_iter, remove_rate):
    # Run BSO-LNS solver
    start_time = time.time()
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

    save_results(best_cost, run_time, route_start_t, num_vehicles)
    best_solution, best_cost = bso_solver.run()
    run_time = time.time() - start_time

    return best_solution, best_cost, run_time


def main():

    best_solution, best_cost, run_time = run_bso(
        **default_params,  
        travel_time_fn=td_travel_time_wrapper,
        demands=customer_demands,
        vehicle_capacity=vehicle_capacity,
        start_time=route_start_t,
        demands=customer_demands,
        depot_node_id=depot_node_id)
    

    print(f"BSO-LNS final best cost: {best_solution['cost']:.2f}, Routes: {best_solution['sol']}")
    save_results(best_solution["cost"], run_time, route_start_t, num_vehicles)

    def test_edge_example(u=12 ,v=275):
        # Time-breakpoints demand function
        t1, t2, t3, t4 = 6.5 * 60, 8.5 * 60, 10 * 60, 12 * 60
        t5, t6, t7, t8 = 16.5 * 60, 18 * 60, 20 * 60, 22 * 60
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

            return pd.DataFrame(records)

    # plot_solution(sim, best_solution["sol"], route_start_t, customer_node_ids, depot_node_id)

if __name__ == "__main__":
    main()


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
# Small Customerbase
#[386 ,370 , 17 ,267 ,303, 321,305]
# Medium Customerbase
#[386, 370,  17,  303,  305,  342, 400,  372, 358 , 404 ,333 ,390 ,369]
# Large Customerbase
#[386 ,370 , 17, 267 ,303, 321 ,305 ,308, 342, 400, 6, 372, 358,  300, 404, 333, 390, 369, 325, 388]
depot_node_id = 406   
customer_node_ids = [386 ,370 , 17, 267 ,303, 321 ,305 ,308, 342, 400, 6, 372, 358,  300, 404, 333, 390, 369, 325, 388]
sim_start = 6 * 60 * 60  # 6:00
start_time = (7 * 60)*60  # 15:30 (in seconds)
num_vehicles = 4
n_demand = [1] * len(customer_node_ids)  #Demand per customer
demands = {customer_node_ids[i]: n_demand[i] for i in range(len(customer_node_ids))}
total_demand = sum(n_demand)
vehicle_capacity = math.ceil(total_demand / num_vehicles)
edge_example = 12 ,275 

#Default Parameters for BSO
default_params = {
"pop_size": 150,
"max_iter": 66,
"n_clusters": 2,
"ideas_per_cluster": 10,
"remove_rate": 0.6,}

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
    if u_actual == v_actual: return 0.0

    # Discretize time for the cache key to improve cache hits
    time_key = round(depart_t) 
    cache_key = (u_actual, v_actual, time_key)

    if cache_key in memoized_travel_times:
        return memoized_travel_times[cache_key]
        
    # Use the original, precise time for the actual calculation
    path_nodes, duration = sim._dynamic_dijkstra(u_actual, v_actual, depart_t) 
    
    memoized_travel_times[cache_key] = duration
    return duration


def run_bso(start_time, vehicle_capacity,demands, pop_size, n_clusters,ideas_per_cluster, max_iter, remove_rate):
    # Run BSO-LNS solver
    alg_start_time = time.time()
    bso_solver = BSOLNS(
        travel_time_fn=td_travel_time_wrapper,
        demands=demands,
        vehicle_capacity=vehicle_capacity,
        start_time=start_time,
        pop_size=pop_size,
        n_clusters=n_clusters,
        ideas_per_cluster=ideas_per_cluster,
        max_iter=max_iter,
        remove_rate=remove_rate
    )

    
    best_solution, cost_history = bso_solver.run()
    best_cost = best_solution['cost']
    run_time = time.time() - alg_start_time

    return best_solution, best_cost, run_time


def main():
    num_runs = 2
    experiment_name = "bso_40_runs_large_customers"
    
    # Let save_results function determine the full path
    output_file_path = os.path.join(os.getcwd(), "output", f"{experiment_name}.xlsx")
    if os.path.exists(output_file_path):
        os.remove(output_file_path)
        print(f"Removed old results file to start fresh: {output_file_path}")

    all_results = []

    print(f"--- Starting {num_runs} BSO-LNS runs. Results will be saved to {output_file_path} ---")


    for i in range(num_runs):
        print(f"\n--- Starting Run {i+1} of {num_runs}")
        # Run with default parameters
        best_solution, best_cost, run_time = run_bso(
        **default_params,  
        demands=customer_demands,
        vehicle_capacity=vehicle_capacity,
        start_time=start_time,)

        print(f"BSO-LNS final best cost: {best_cost / 60:.2f} minutes ({best_cost:.0f} seconds)")
        print(f"Solution (tour, splits): {best_solution}")

        # 3. Collect the key results from this run
        save_results(
                cost=best_cost, 
                runtime_seconds=run_time, 
                num_vehicles=num_vehicles,
                name=experiment_name)   
            
        all_results.append({
                'run_number': i + 1,
                'cost': best_cost,
                'solution': best_solution}) 
        print("\n ---Complete-----")
    
    if all_results:
        best_overall_run = min(all_results, key=lambda x: x['cost'])
        print("\n--- Best Overall Run Found ---")
        print(f"From Run Number: {best_overall_run['run_number']}")
        print(f"Best Cost: {best_overall_run['cost'] / 60:.2f} minutes")


    print(f"BSO-LNS final best cost: {best_solution['cost']:.2f}, Routes: {best_solution['sol']}")
    save_results(best_solution["cost"], run_time, start_time, num_vehicles)

    # plot_solution(sim, best_solution["sol"], route_start_t, customer_node_ids, depot_node_id)

if __name__ == "__main__":
    main()


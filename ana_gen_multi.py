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
import vrp_sim as vs
from read_cities import read_anaheim
from plot_solution import plot_solution
import time
from export_excel import save_results

# Define GA Problem: Depot and Customers
# Small Customerbase
#[386 ,370 , 17 ,267 ,303, 321,305]
# Medium Customerbase
#[386, 370,  17,  303,  305,  342, 400,  372, 358 , 404 ,333 ,390 ,369]
# Large Customerbase
#[386 ,370 , 17, 267 ,303, 321 ,305 ,308, 342, 400, 6, 372, 358,  300, 404, 333, 390, 369, 325, 388]
depot_node_id = 406   
customer_node_ids =  [386 ,370 , 17, 267 ,3033, 321 ,305 ,308, 342, 400, 6, 372, 358,  300, 404, 333, 390, 369, 325, 388]
time_step_minutes = 10  # mins
sim_start = 6 * 60 *60 # 6:00
route_start_t = 7 * 60 * 60
num_vehicles = 4
n_demand = [1] * len(customer_node_ids)  #Demand per customer
demands_dict = {customer_node_ids[i]: n_demand[i] for i in range(len(customer_node_ids))}
total_demand = sum(n_demand)
vehicle_capacity = math.ceil(total_demand / num_vehicles)

# Parameters for GA
default_params = {
"pop_size": 400,
"max_gens": 50,
"tournament_size": 3,
"crossover_rate": 0.9,
"mutation_rate": 0.11,
"elite_count": 57,}

# Time-breakpoints demand function
t1, t2, t3, t4 = 6.5 * 60, 8.5 * 60, 10 * 60, 12 * 60
t5, t6, t7, t8 = 16.5 * 60, 18 * 60, 20 * 60, 22 * 60
route_start_t_minutes = route_start_t /60
breaks_in_minutes = sorted([0, t1, t2, t3, t4, route_start_t_minutes, t5, t6, t7, t8, 24*60])
period_breaks = [t * 60 for t in breaks_in_minutes]

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

# Load Network data
edges_df, nodes_df, trips_df, flow_df = read_anaheim()

sim = vs.TrafficSim(edges_df, flow_df, nodes=nodes_df)

# Filter out any customer IDs not present
customer_node_ids = [nid for nid in customer_node_ids if sim.G.has_node(nid)]
if not sim.G.has_node(depot_node_id):
    raise ValueError(f"Depot node {depot_node_id} not in graph.")
if len(customer_node_ids) < 1:
    raise ValueError("No customer nodes defined or found in graph.")

memoized_travel_times = {}

def td_travel_time_wrapper(u, v, depart_t):
        # u and v are already actual node IDs, so no mapping needed
        if u == v:
            return 0.0
        cache_key = (u, v, depart_t)
        cache_key = (u, v, depart_t)
        if cache_key in memoized_travel_times:
            return memoized_travel_times[cache_key]
        path_nodes, duration = sim._dynamic_dijkstra(u, v, depart_t)
        memoized_travel_times[cache_key] = duration
        return duration


# 5) Instantiate GA_DP, passing exactly those arguments:
def run_ga(route_start_t, num_vehicles, vehicle_capacity,
           period_breaks, demands_dict, depot_node_id,pop_size,max_gens,tournament_size,crossover_rate,mutation_rate,elite_count):
    """
    Runs one GA sweep with the given parameters.
    Returns: (best_solution, best_cost, run_time_seconds)
    """
    start_time = time.time()
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

    best_solution, best_cost = ga_solver.run()
    run_time = time.time() - start_time

    # Save results for bookkeeping
    #save_results(best_cost, run_time, route_start_t, num_vehicles)

    return best_solution, best_cost, run_time

if __name__ == '__main__':
    
    num_runs = 40
    all_results=[]

    output_name = "ga_40_runs_log" 

    if os.path.exists(output_filename):
        os.remove(output_filename)

    for i in range(num_runs):
        print(f"\n--- Starting Run {i+1} of {num_runs}")
        #2.Run with default parameters
        best_solution, best_cost, run_time = run_ga(
            **default_params,  
            route_start_t=route_start_t,
            num_vehicles=num_vehicles,
            vehicle_capacity=vehicle_capacity,
            period_breaks=period_breaks,
            demands_dict=demands_dict,
            depot_node_id=depot_node_id)

    print(f"GA final best cost: {best_cost / 60:.2f} minutes ({best_cost:.0f} seconds)")
        # 3. Collect the key results from this run
    all_results.append({
            'run_number': i + 1,
            'best_cost_seconds': best_cost,
            'run_time_seconds': run_time,
            'solution_tour': best_solution[0],  # Storing the giant tour
            'solution_splits': best_solution[1] # Storing the splits
        })    
    print("\n ---Complete-----")
    result_df = pd.DataFrame(all_results)

    output_filename ="ga_multiple runs_results.csv"

    
    #print(f"Solution details (giant_tour, splits): {best_solution}")

    save_results(
            filename=output_filename,
            run_number=i + 1,
            best_cost=best_cost, 
            run_time=run_time, 
            route_start_t=route_start_t, 
            num_vehicles=num_vehicles)

    #plot_solution(sim, best_solution, route_start_t, customer_node_ids, depot_node_id)
    #plot_solution(sim, best_solution["sol"], route_start_t, customer_node_ids, depot_node_id)
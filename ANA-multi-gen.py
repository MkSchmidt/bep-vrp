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
depot_node_id = 406   
customer_node_ids = [386 ,370 , 17 ,267 ,303, 321,305]
time_step_minutes = 10  # mins
sim_start = 0 * 60 *60 # 6:00
route_start_t = (15*60 +30)*60
num_vehicles = 2
n_demand = [1] * len(customer_node_ids)  #Demand per customer
n_demand = [1] * len(customer_node_ids)
demands_dict = {customer_node_ids[i]: n_demand[i] for i in range(len(customer_node_ids))}
total_demand = sum(n_demand)
vehicle_capacity = math.ceil(total_demand / num_vehicles)
B = 0.15
edge_example = 12 ,275 

# Time-breakpoints demand function
t1, t2, t3, t4 = 6.5 * 60, 8.5 * 60, 10 * 60, 12 * 60
t5, t6, t7, t8 = 16.5 * 60, 18 * 60, 20 * 60, 22 * 60
route_start_t_minutes = route_start_t /60
breaks_in_minutes = sorted([0, t1, t2, t3, t4, route_start_t_minutes, t5, t6, t7, t8, 24*60])
period_breaks = [t * 60 for t in breaks_in_minutes]

# Parameters for GA
pop_size=50
max_gens=10
tournament_size=2    
crossover_rate=0.9
mutation_rate=0.2
elite_count=2

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

start_time = time.time()

# 5) Instantiate GA_DP, passing exactly those arguments:
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

# 2. Use the correct variables in your print and save functions
# Cost is in seconds, so convert to minutes for display
print(f"GA final best cost: {best_cost / 60:.2f} minutes ({best_cost:.0f} seconds)")
print(f"Solution details (giant_tour, splits): {best_solution}")

save_results(best_cost, run_time, route_start_t, num_vehicles)


# plot_solution(sim, best_solution["sol"], route_start_t, customer_node_ids, depot_node_id)
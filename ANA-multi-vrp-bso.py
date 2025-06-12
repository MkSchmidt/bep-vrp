import os
import math
import random
import numpy as np
import pandas as pd
import networkx as nx
from itertools import pairwise
from BsoLns_imp import BSOLNS 
import vrp_sim as vs
from read_cities import read_anaheim
from plot_solution import plot_solution
import time
from export_excel import save_results
from matplotlib import pyplot as plt

# Define BSO-LNS Problem: Depot and Customers
depot_node_id = 406
customer_node_ids = [386 ,370 , 17 ,267 ,303, 321,305, 308, 342, 400, 6, 372, 358, 300, 404, 333, 390, 369, 325, 388]
sim_start = 7 * 60 * 60  # 7:00
route_start_t = 7*60*60  # 7:00 (in seconds)

# Parameters for BSO
pop_size = 20
n_clusters = 2
ideas_per_cluster = 2
max_iter = 30
remove_rate = 0.2

# Load Network data
edges_df, nodes_df, trips_df, flow_df = read_anaheim()

sim = vs.TrafficSim(edges_df, flow_df, nodes=nodes_df)

def run(num_customers, num_vehicles, sim_number=1):
    n_demand = [1] * num_customers #Demand per customer
    total_demand = sum(n_demand)
    vehicle_capacity = math.ceil(num_customers / num_vehicles)

    # Filter out any customer IDs not present
    customer_nodes = [nid for nid in customer_node_ids if sim.G.has_node(nid)][0:num_customers]
    if not sim.G.has_node(depot_node_id):
        raise ValueError(f"Depot node {depot_node_id} not in graph.")
    if len(customer_node_ids) < 1:
        raise ValueError("No customer nodes defined or found in graph.")

    # BSO-LNS setup: map from BSO indices → actual node IDs
    customer_demands = n_demand
    bso_nodes_map = [depot_node_id] + customer_nodes

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

    start_time = time.time()
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

    best_solution, cost_history, population_history = bso_solver.run()
    run_time = time.time() - start_time
    print(f"BSO-LNS final best cost: {best_solution['cost']:.2f}, Routes: {best_solution['sol']}")

    plot_solution(sim, best_solution["sol"], route_start_t, customer_node_ids, depot_node_id)
    breakpoint()

    save_results(best_solution["cost"], run_time, num_customers, num_vehicles)

    population_scatter_x = [ i for i in range(len(population_history)) for j in population_history[i] ]
    population_scatter_y = [ cost for population in population_history for cost in population ]

    plt.scatter(population_scatter_x, population_scatter_y)
    plt.title("Brain storm optimization")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.savefig(f"output/test-result-{num_customers}c{num_vehicles}v-{sim_number:03d}.svg")
#plt.show()


for i in range(5):
    run(13,4, sim_number=i)

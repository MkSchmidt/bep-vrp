import argparse
import math
import traci
import networkx as nx
from sumolib import net as sumonet
from GA_Imp import GA_DP
path = "C:/Users/tiesv/OneDrive/Werktuigbouwkunde/BEP/bep-vrp/output/"
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cfg",
        required=True,
        help="SUMO .sumocfg (must include NPC routes in <route-files>)",
    )
    p.add_argument(
        "--net",
        required=True,
        help="SUMO network .net.xml",
    )
    p.add_argument(
        "--depot",
        required=True,
        help="Depot node ID (string) as in your SUMO net",
    )
    p.add_argument(
        "--customers",
        required=True,
        nargs="+",
        help="List of customer node IDs (space separated; strings)",
    )
    p.add_argument(
        "--vehs",
        type=int,
        default=1,
        help="Number of vehicles to use in VRP (default: 1)",
    )
    return p.parse_args()


def build_graph(netfile):
    """
    Build a static directed graph G where:
      - G.nodes are SUMO node IDs (strings).
      - Each edge (u, v) has attributes:
          'sumo_edge': the SUMO edge ID (string)
          'free_flow_time': length / speed (seconds)
          'length': length (meters)
    """
    snet = sumonet.readNet(netfile)
    G = nx.DiGraph()
    for edge in snet.getEdges():
        u = edge.getFromNode().getID()
        v = edge.getToNode().getID()
        length = edge.getLength()
        speed = edge.getSpeed()
        # free-flow travel time in seconds
        fft = length / speed if speed > 0 else float("inf")
        G.add_edge(u, v,
                   sumo_edge=edge.getID(),
                   free_flow_time=fft,
                   length=length)
    return G


def td_travel_time(u, v, depart_t, G, depot_id):
    # Map GA_DP’s “0” to the real depot
    u_str = depot_id if (u == 0 or u == "0") else u
    v_str = depot_id if (v == 0 or v == "0") else v

    # Standard Dijkstra – track travel time in seconds (we will convert to minutes)
    dist = {node: float("inf") for node in G.nodes}
    dist[u_str] = 0.0
    visited = set()
    import heapq
    heap = [(0.0, u_str)]

    while heap:
        curr_t, x = heapq.heappop(heap)
        if curr_t > dist[x]:
            continue
        if x == v_str:
            break
        for y in G.successors(x):
            edge_id = G[x][y]["sumo_edge"]
            length_m = G[x][y]["length"]
            # Query SUMO for the last-step mean speed (m/s). If 0 or None, use free-flow fallback:
            speed_ms = traci.edge.getLastStepMeanSpeed(edge_id)
            if speed_ms is None or speed_ms < 0.1:
                # fallback to free-flow speed = length / free_flow_time
                fft = G[x][y]["free_flow_time"]
                speed_ms = (length_m / fft) if (fft > 0 and fft < float("inf")) else 1.0
            travel_sec = length_m / speed_ms
            alt = curr_t + travel_sec
            if alt < dist[y]:
                dist[y] = alt
                heapq.heappush(heap, (alt, y))

    if math.isinf(dist[v_str]):
        # no path → return a very large travel time (minutes)
        return float("inf")
    return dist[v_str] / 60.0  # convert sec → min


def travel_distance(u, v, G, depot_id):
    """
    Return the static shortest-path distance (km) from u to v in G (meters → km),
    mapping GA_DP’s 0 → depot_id.
    """
    u_str = depot_id if (u == 0 or u == "0") else u
    v_str = depot_id if (v == 0 or v == "0") else v

    try:
        length_m = nx.dijkstra_path_length(G, u_str, v_str, weight="length")
    except nx.NetworkXNoPath:
        return float("inf")
    return length_m / 1000.0  # to km


def main(cfg, net, depot, customers, vehs):

    # 1. Start SUMO (headless)
    traci.start(["sumo-gui", "-c", cfg, "--start", "--no-step-log"])

    # 2. Build static graph from the network
    G = build_graph(net)
    snet = sumonet.readNet(net)
    assert G and len(G) > 0, f"Failed to build graph from {net}"

    # 3. Prepare VRP inputs
    depot_id = depot       # e.g. "344"
    customers = customers  # list of strings, e.g. ["4", "82", "74", ...]
    bso_nodes = [depot_id] + customers

    # Every customer has unit demand = 1.0
    # GA_DP expects demands: dict[node_id → demand_kg], excluding depot.
    demands = {cust: 1.0 for cust in customers}

    total_demand = sum(demands.values())
    num_vehicles = vehs
    num_customers = len(customers)
    # Evenly split total_demand across vehicles
    vehicle_capacity = math.ceil(num_customers / num_vehicles)

    # No time windows in this example
    time_windows = {}

    # GA_DP requires at least two period breaks; we only care about a single “period”:
    period_breaks = [0.0, 1440.0]  # a 24-hour window in minutes

    # Dummy emission function (we only minimize time here)
    def emission_fn(weight, speed):
        return 0.0

    # 4. Wrap travel_time_fn & travel_distance_fn so that whenever GA_DP passes u=0 or v=0,
    #    we look up the real depot_id. Otherwise, pass through the string node as is.
    travel_time_fn = lambda u, v, t: td_travel_time(u, v, t, G, depot_id)
    travel_distance_fn = lambda u, v: travel_distance(u, v, G, depot_id)

    # 5. Instantiate GA_DP
    ga = GA_DP(
        travel_time_fn=travel_time_fn,
        travel_distance_fn=travel_distance_fn,
        demands=demands,
        num_vehicles=num_vehicles,
        vehicle_capacity=vehicle_capacity,
        time_windows=time_windows,
        period_breaks=period_breaks,
        emission_fn=None,
        pop_size=50,
        max_gens=10,
        tournament_size=2,
        crossover_rate=0.9,
        mutation_rate=0.2,
        elite_count=2,
        start_time=0.0,
    )

    # 6. Run the GA
    print("Running GA_DP to optimize routes...")
    best_solution, best_cost = ga.run()
    giant_tour, split_indices = best_solution
    print(f"=== GA finished: best cost = {best_cost:.2f} ===")
    print(f"Giant tour (customer node IDs): {giant_tour}")
    print(f"Split indices: {split_indices}")

    # 7. Reconstruct per-vehicle subroutes from giant_tour + split_indices
    routes = []
    N = len(giant_tour)
    prev = 0
    for idx in split_indices:
        routes.append(giant_tour[prev:idx])
        prev = idx
    routes.append(giant_tour[prev:])  # last vehicle
    # If GA allocated fewer than num_vehicles, some sublists may be empty

    # 8. Extract DP schedule so we know each vehicle’s departure time from the depot
    full_schedule = {}
    for subroute in routes:
        if not subroute:
            continue
        _, schedule = ga._dynamic_programming(subroute, 0)
        # schedule maps each node (string or 0) → (arrival_min, depart_min)
        # In particular, schedule[0] = (arrival_at_depot, depart_from_depot)
        for node_id, times in schedule.items():
            full_schedule[node_id] = times

    # 9. Add each GA-optimized vehicle to SUMO
    veh_idx = 0
    for subroute in routes:
        if not subroute:
            continue

        # Build full node path: depot → c1 → c2 → … → depot
        hop_nodes = [depot_id] + subroute + [depot_id]

        # Convert node sequence → edge sequence:
        # For each (u, v), find a free-flow shortest path by “free_flow_time”
        # then translate each link (a, b) to G[a][b]['sumo_edge'].
        edge_seq = []
        for (u, v) in zip(hop_nodes, hop_nodes[1:]):
            try:
                node_path = nx.shortest_path(G, source=u, target=v, weight="free_flow_time")
            except nx.NetworkXNoPath:
                node_path = []
            if len(node_path) < 2:
                edge_seq = []
                break
            for (a, b) in zip(node_path, node_path[1:]):
                edge_id = G[a][b]["sumo_edge"]
                edge_seq.append(edge_id)
        if not edge_seq:
            continue

        # Determine depart time from depot (GA_DP stored schedule[0] = (arr, depart))
        depot_info = full_schedule.get(0, (None, None))
        depart_time = depot_info[1] if depot_info[1] is not None else 0.0

        route_id = f"ga_route_{veh_idx}"
        veh_id = f"ga_veh_{veh_idx}"
        traci.route.add(route_id, edge_seq)
        traci.vehicle.add(
            veh_id,
            routeID=route_id,
            typeID="DEFAULT_VEHTYPE",
            depart=f"{depart_time:.2f}"
        )
        veh_idx += 1

    # 10. Continue SUMO until all vehicles (NPC + GA) finish
    print("Starting SUMO simulation with GA-optimized routes + NPC traffic...")
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

    traci.close()
    print("Simulation complete.")


if __name__ == "__main__":
    args = parse_args()
    main(path + args.cfg, path + args.net, args.depot, args.customers, args.vehs)

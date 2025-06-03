#!/usr/bin/env python3
import argparse
import traci
import networkx as nx
from BsoLns_imp import BSOLNS
from itertools import pairwise
from sumolib import net as sumonet

# 1) ARG PARSING — depot + customers only
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default='BEP-VRP/output/anaheim.sumocfg', required=True, help="SUMO .sumocfg (with NPC routes)")
    p.add_argument("--depot",     required=True, help="Depot node ID")
    p.add_argument("--customers", required=True, nargs="+", help="Customer node IDs (space separated)")
    p.add_argument("--net", default="BEP-VRP/output/anaheim_net.xml", required=True)
    return p.parse_args()

# 2) BUILD A NETWORKX GRAPH ON THE FLY FROM TraCI

def build_graph(netfile):
    snet = sumonet.readNet(netfile)
    G = nx.DiGraph()
    for edge in snet.getEdges():
        u = edge.getFromNode().getID()
        v = edge.getToNode().getID()
        length = edge.getLength()          # in meters
        speed  = edge.getSpeed()           # in m/s
        fft    = length / speed            # free‐flow travel‐time in s

        G.add_edge(u, v,
                   sumo_edge=edge.getID(),
                   free_flow_time=fft,
                   length=length)  

# 3) TIME‐DEPENDENT TRAVEL‐TIME FUNCTION via TraCI
def td_travel_time(u, v, depart_t, G):
    """
    Run a tiny Dijkstra over G, but at each edge query 
    the *current* speed via TraCI to get time‐dependent costs.
    depart_t is in seconds.
    """
    dist = {n: float("inf") for n in G}
    prev = {}
    dist[u] = depart_t
    Q = set(G)
    while Q:
        x = min(Q, key=lambda n: dist[n])
        Q.remove(x)
        if x == v:
            break
        tx = dist[x]
        for y in G.successors(x):
            edge_id = G[x][y]["sumo_edge"]
            # last‐step mean speed (m/s). if zero, fallback to 0.1 m/s
            speed = traci.edge.getLastStepMeanSpeed(edge_id) or 0.1
            length = G[x][y]['length']
            cost   = length / speed
            alt    = tx + cost
            if alt < dist[y]:
                dist[y], prev[y] = alt, x
    return dist[v] - depart_t

# 4) MAIN: start SUMO, build G, run BSOLNS, inject vehicle
def main():
    args = parse_args()
    # launch SUMO with your existing .cfg (includes NPC traffic)
    traci.start(["sumo", "-c", args.cfg, "--start", "--no-step-log"])
    # build graph from the *running* sim
    G = build_graph(args.net)

    # map depot+customers → BSOLNS indices
    depot     = args.depot
    customers = args.customers
    bso_nodes = [depot] + customers
    demands   = [1] * len(customers)  # unit demand per customer

    # wrap our TD oracle
    def travel_time_fn(u_idx, v_idx, t):
        u = bso_nodes[u_idx]
        v = bso_nodes[v_idx]
        return td_travel_time(u, v, t, G)

    # instantiate & run BSO‐LNS for one-route VRP
    bso = BSOLNS(
        travel_time_fn=travel_time_fn,
        demands=demands,
        vehicle_capacity=sum(demands),
        start_time=0.0,     # sim seconds
        pop_size=20,
        n_clusters=2,
        ideas_per_cluster=2,
        max_iter=50,
        remove_rate=0.3
    )
    best = bso.run()
    print("Best dynamic cost:", best["cost"], "solution:", best["sol"])

    # 5) TRANSLATE BSOLNS ROUTE → SUMO EDGE LIST
    sol      = best["sol"][0]               # e.g. [1,2,3,4,5]
    hop_idxs  = [0] + sol + [0]             # include return to depot
    node_path = [bso_nodes[i] for i in hop_idxs]

    sumo_route = []
    for a, b in pairwise(node_path):
        # ask SUMO for its current shortest-edge route
        edges = traci.simulation.findRoute(a, b).edges
        sumo_route.extend(edges)

    # 6) INJECT YOUR BSO VEHICLE
    traci.route.add("bso_route", sumo_route)
    traci.vehicle.add("bso_veh", routeID="bso_route", typeID="car")

    # 7) RUN UNTIL IT FINISHES (NPC traffic keeps going)
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

    traci.close()
    print("Simulation complete.")

if __name__ == "__main__":
    main()

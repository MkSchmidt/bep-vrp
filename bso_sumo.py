import argparse
import traci
import networkx as nx
from BsoLns_imp import BSOLNS
from itertools import pairwise
from sumolib import net as sumonet

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default='BEP-VRP/output/anaheim.sumocfg', required=True, help="SUMO .sumocfg (with NPC routes)")
    p.add_argument("--depot",     required=True, help="Depot node ID")
    p.add_argument("--customers", required=True, nargs="+", help="Customer node IDs (space separated)")
    p.add_argument("--net", default="BEP-VRP/output/anaheim_net.xml", required=True)
    p.add_argument("--vehs", type=int, required=False, default=1)
    return p.parse_args()

def build_graph(netfile):
    snet = sumonet.readNet(netfile)
    G = nx.DiGraph()
    for edge in snet.getEdges():
        u = edge.getFromNode().getID()
        v = edge.getToNode().getID()
        length = edge.getLength()
        speed  = edge.getSpeed()
        fft    = length / speed

        G.add_edge(u, v,
                   sumo_edge=edge.getID(),
                   free_flow_time=fft,
                   length=length)  
    return G

def td_travel_time(u, v, depart_t, G):
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
            speed = traci.edge.getLastStepMeanSpeed(edge_id) or 0.1
            length = G[x][y]['length']
            cost   = length / speed
            alt    = tx + cost
            if alt < dist[y]:
                dist[y], prev[y] = alt, x
    return dist[v] - depart_t


def main():
    args = parse_args()
    traci.start(["sumo-gui", "-c", args.cfg, "--start", "--no-step-log"])
    G = build_graph(args.net)
    snet = sumonet.readNet(args.net)    
    assert G is not None and len(G) > 0, f"build_graph({args.net}) failed to load any edges!"

    depot     = args.depot
    customers = args.customers
    bso_nodes = [depot] + customers
    # every customer has unit demand
    demands   = [1] * len(customers)

    # split total demand evenly across args.vehs vehicles
    import math
    total_demand     = sum(demands)
    num_vehicles     = args.vehs
    vehicle_capacity = math.ceil(total_demand / num_vehicles) 

    def travel_time_fn(u_idx, v_idx, t):
        u = bso_nodes[u_idx]
        v = bso_nodes[v_idx]
        return td_travel_time(u, v, t, G)

    bso = BSOLNS(
        travel_time_fn=travel_time_fn,
        demands=demands,
        vehicle_capacity=vehicle_capacity,
        start_time=0.0,
        pop_size=20,
        n_clusters=3,
        ideas_per_cluster=1,
        max_iter=10,
        remove_rate=0.5
    )
    best = bso.run()
    print("Best dynamic cost:", best["cost"], "solution:", best["sol"])

    # now best["sol"] is a list of routes, one per vehicle
    for vidx, route in enumerate(best["sol"]):
        hop_idxs  = [0] + route + [0]
        node_path = [bso_nodes[i] for i in hop_idxs]

        sumo_route = []
        for u, v in pairwise(node_path):
            subpath = nx.shortest_path(G,source=u,target=v,weight="free_flow_time")
            for a, b in pairwise(subpath):
                edge_id = G[a][b]["sumo_edge"]
                sumo_route.append(edge_id)

        route_id = f"bso_route_{vidx}"
        veh_id   = f"bso_veh_{vidx}"

        traci.route.add(route_id, sumo_route)
        traci.vehicle.add(veh_id, routeID=route_id, typeID="DEFAULT_VEHTYPE", depart=vidx*2.0)

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

    traci.close()
    print("Simulation complete.")

if __name__ == "__main__":
    main()

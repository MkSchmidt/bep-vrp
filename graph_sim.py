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
    load_nodefile, project_root
)
from BsoLns_imp import BSOLNS


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
        for u, v in pairwise(path)
    )


def get_added_travel_time(edge, t: float) -> float:
    # simple rush‐hour bump
    t_tc = ((t % (24*60)) - 18*60)
    h = edge["volume"] / edge["capacity"] * edge["free_flow_time"] * 3
    return max(0, h + min(-0.2 * t_tc, h/90 * t_tc))


def get_travel_time(edge, t: float) -> float:
    return edge["free_flow_time"] + get_added_travel_time(edge, t)


def get_node_sequence(graph: nx.DiGraph, end: int) -> list:
    prev = graph.nodes[end]["previous"]
    if prev is None:
        return [end]
    return get_node_sequence(graph, prev) + [end]


def dynamic_dijkstra(graph: nx.DiGraph, start: int, end: int, start_t: float) -> list:
    nx.set_node_attributes(graph, math.inf, "arrival_time")
    nx.set_node_attributes(graph, None, "previous")
    graph.nodes[start]["arrival_time"] = start_t

    Q = set(graph.nodes)
    while Q:
        u = min(Q, key=lambda n: graph.nodes[n]["arrival_time"])
        if u == end:
            return get_node_sequence(graph, end)
        Q.remove(u)
        t_u = graph.nodes[u]["arrival_time"]
        for v in graph.neighbors(u):
            if v not in Q:
                continue
            τ = get_travel_time(graph.edges[u, v], t_u)
            alt = t_u + τ
            if alt < graph.nodes[v]["arrival_time"]:
                graph.nodes[v]["arrival_time"] = alt
                graph.nodes[v]["previous"] = u


def arrival_times_for_path(graph: nx.DiGraph, path: list, start_t: float) -> dict:
    times = {}
    t = start_t
    for u, v in pairwise(path):
        times[(u, v)] = t
        times[(v, u)] = t
        t += get_travel_time(graph.edges[u, v], t)
    return times


if __name__ == "__main__":
    # ── load network ───────────────────────────────────────────────
    edges = load_edgefile(os.path.join(project_root,"TransportationNetworks","Chicago-Sketch", "ChicagoSketch_net.tntp"))
    nodes = load_nodefile(os.path.join(project_root,"TransportationNetworks","Chicago-Sketch", "ChicagoSketch_node.tntp"))
    flow  = load_flowfile(os.path.join(project_root,"TransportationNetworks","Chicago-Sketch", "ChicagoSketch_flow.tntp"))

    G = graph_from_data(edges, nodes)
    undirected = nx.Graph(G)
    for row in flow.to_dict("records"):
        undirected.edges[row["from"], row["to"]]["volume"] = row["volume"]

    node_positions = {
        n: undirected.nodes[n]["coordinates"]
        for n in undirected.nodes
    }
    edges_to_draw = list(undirected.edges())

    fig, ax = plt.subplots()
    drawn_edges = nx.draw_networkx_edges(
        undirected, node_positions,
        edgelist=edges_to_draw,
        edge_color="0.8", ax=ax
    )
    nx.draw_networkx_nodes(
        undirected, node_positions,
        nodelist=[911, 918], node_size=50, node_color="k"
    )
    title = ax.set_title("t=00:00")

    # ── compute static & dynamic Dijkstra ─────────────────────────
    route_start_t = 15.5 * 60  # 15:30 in minutes

    # dynamic
    dyn_path = dynamic_dijkstra(undirected, 918, 911, route_start_t)
    dynamic_edges = arrival_times_for_path(undirected, dyn_path, route_start_t)

    # static
    static_path = nx.shortest_path(
        undirected, source=918, target=911,
        weight="free_flow_time"
    )
    static_edges = arrival_times_for_path(undirected, static_path, route_start_t)

    # ── dynamic BSO‐LNS integration ────────────────────────────────
    # We’ll treat nodes [918→210→911] as depot+two customers
    bso_nodes = [918, 911]

    def td_travel_time(u_idx, v_idx, depart_t):
        """
        Returns the time‐dependent shortest‐path travel time from bso_nodes[u_idx]
        to bso_nodes[v_idx], departing at time depart_t.
        """
        u_node = bso_nodes[u_idx]
        v_node = bso_nodes[v_idx]
        # run dynamic Dijkstra on the full graph
        path = dynamic_dijkstra(undirected, u_node, v_node, depart_t)
        # dynamic_dijkstra writes the arrival_time into graph.nodes[v_node]
        arrival = undirected.nodes[v_node]["arrival_time"]
        return arrival - depart_t

    # instantiate BSO-LNS to visit that one customer and return
    bso = BSOLNS(
        travel_time_fn=td_travel_time,
        demands=[1],            # single customer
        vehicle_capacity=1,     # capacity must cover that single demand
        start_time=route_start_t,
        pop_size=5,             # you can tune these
        n_clusters=1,
        ideas_per_cluster=1,
        max_iter=10,
        remove_rate=0.2
    )

    best = bso.run()
    print("BSO-LNS dynamic best cost:", best["cost"], "routes:", best["sol"])

    # ── Reconstruct the actual two‐leg path for coloring ────────────────
    # 1) forward leg: depot → customer
    forward_path = dynamic_dijkstra(undirected, 918, 911, route_start_t)
    t_arrival = undirected.nodes[911]["arrival_time"]

    # 2) return leg: customer → depot
    return_path  = dynamic_dijkstra(undirected, 911, 918, t_arrival)

    # 3) stitch them together (drop duplicate 911)
    bso_full_path = forward_path + return_path[1:]

    # 4) compute the time each edge is entered
    bso_edges = arrival_times_for_path(undirected, bso_full_path, route_start_t)
    # ── animation update ──────────────────────────────────────────
    total_minutes = 6*60
    time_step = 10
    def update_colors(frame):
        # 'frame' here is already in minutes (because we use range(0, total_minutes, time_step))
        t = route_start_t + frame

        # recompute base intensities
        add_times = [ get_added_travel_time(undirected.edges[e], t)
                    for e in edges_to_draw ]
        base_int  = [ str(max(0.8/15 * (15 - tau), 0)) for tau in add_times ]

        default_colors = []
        for i, e in enumerate(edges_to_draw):
            if e in dynamic_edges and dynamic_edges[e] <= t:
                default_colors.append("blue")
            else:
                default_colors.append(base_int[i])

        # now overlay static (red), dynamic (blue), BSO (green)
        colors = []
        for e in edges_to_draw:
            if e in bso_edges and bso_edges[e] <= t:
                colors.append("green")
            elif e in static_edges and static_edges[e] <= t:
                colors.append("red")
            elif e in dynamic_edges and dynamic_edges[e] <= t:
                colors.append("blue")
            else:
                colors.append(default_colors[edges_to_draw.index(e)])

        title.set_text(f"t={int(t)//60:02d}:{int(t)%60:02d}")
        drawn_edges.set_color(colors)
        return [drawn_edges, title]

    # 3) animate with frames=actual minutes:
    anim = animation.FuncAnimation(
        fig,
        update_colors,
        frames=range(0, total_minutes, time_step),
        interval=200,  # ms between frames
        blit=True
    )

    plt.show()

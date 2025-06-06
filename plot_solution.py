from matplotlib import pyplot as plt, animation
import networkx as nx
from itertools import pairwise

def arrival_times_for_path(sim, path: list, start_t: float) -> dict:
    times = {}
    t = start_t
    for u, v in pairwise(path):
        if not sim.G.has_edge(u, v):
            continue
        times[(u, v)] = t
        if sim.G.has_edge(v, u):
            times[(v, u)] = t
        t += sim._get_edge_travel_time(u, v, t)
    return times

def get_route_colors(num_routes):
    colormap = plt.cm.get_cmap('tab10', num_routes)
    return [colormap(i) for i in range(num_routes)]

def plot_solution(sim, bso_solution_routes, start_t, customer_node_ids, depot_node_id):
    # Coordinates for plotting
    pos = {n: data["coordinates"] for n, data in sim.G.nodes(data=True)}
    edges = list(sim.G.edges())
    # Plot setup: draw nodes & edges once
    fig, ax = plt.subplots(figsize=(10, 8))
    
    bso_nodes_map = [depot_node_id] + customer_node_ids

    nx.draw_networkx_nodes(
        sim.G, pos,
        node_size=1, ax=ax, node_color='gray'
    )
    nx.draw_networkx_nodes(
        sim.G, pos,
        nodelist=customer_node_ids,
        node_size=20, ax=ax, node_color='blue'
    )
    nx.draw_networkx_nodes(
        sim.G, pos,
        nodelist=[depot_node_id],
        node_size=20, ax=ax, node_color='red'
    )
    drawn = nx.draw_networkx_edges(
        sim.G, pos,
        edgelist=edges, edge_color="0.8", ax=ax
    )

    title = ax.set_title("t=00:00")
    timer_text = ax.text(
        0.02, 0.95,
        "Elapsed: 00:00:00",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    # Reconstruct the BSO solution path for animation
    bso_edges_for_animation = {}

    # FIXED: use a fresh t = route_start_t per route, store both (u,v) and (v,u)
    for route in bso_solution_routes:
        t = start_t  # each vehicle leaves at 15:30
        current_node = depot_node_id

        # From depot → each customer in route
        for cust_bso_idx in route:
            dest_node = bso_nodes_map[cust_bso_idx]
            path_nodes, duration = sim._dynamic_dijkstra(current_node, dest_node, t)
            if not path_nodes:
                break
            seg_times = arrival_times_for_path(sim, path_nodes, t)
            for (u, v), entry in seg_times.items():
                bso_edges_for_animation[(u, v)] = entry
                bso_edges_for_animation[(v, u)] = entry
            t += duration
            current_node = dest_node

        # Final leg: last customer → depot
        if current_node != depot_node_id:
            path_back, _ = sim._dynamic_dijkstra(current_node, depot_node_id, t)
            if path_back:
                seg_times = arrival_times_for_path(sim, path_back, t)
                for (u, v), entry in seg_times.items():
                    bso_edges_for_animation[(u, v)] = entry
                    bso_edges_for_animation[(v, u)] = entry

    # Update function for animation
    def update_frame(frame_minutes_offset):
        current_sim_time = (start_t + frame_minutes_offset)*60

        # Update title & timer text
        h, m = divmod(current_sim_time, 60)
        title.set_text(f"Time: {h:02}:{m:02}")
        hrs = frame_minutes_offset // 60
        mins = frame_minutes_offset % 60
        timer_text.set_text(f"Elapsed: {hrs:02}:{mins:02}:00")

        # Compute congestion-based grayscale for every undirected edge
        added_travel_times = [
            sim.get_edge_congestion_time(source, dest, current_sim_time) / sim.G.edges[source, dest]["free_flow_time"]
            for source, dest in edges
        ]
        base_intensities = [
            str(max(0.9 * (1 - tau), 0.0))
            for tau in added_travel_times
        ]

        # Assign distinct colors per route
        num_routes = len(bso_solution_routes)
        route_colors = get_route_colors(num_routes)

        # Build edge_colors (one color per edge)
        edge_colors = []
        for i, (u, v) in enumerate(edges):
            t_uv = bso_edges_for_animation.get((u, v), float("inf"))
            t_vu = bso_edges_for_animation.get((v, u), float("inf"))
            route_color = None
            for route_idx in range(num_routes):
                if t_uv <= current_sim_time or t_vu <= current_sim_time:
                    route_color = route_colors[route_idx]
                    break
            if route_color is not None:
                edge_colors.append(route_color)
            else:
                edge_colors.append(base_intensities[i])

        # Update edge colors
        drawn.set_edgecolor(edge_colors)
        return [drawn, title, timer_text]

    # Create and show animation
    ani = animation.FuncAnimation(
        fig,
        update_frame,
        frames=range(0, 24 * 60 - start_t // 60, 15),
        interval=200,
        blit=True
    )

    ax.set_aspect('equal')
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.tight_layout()
    plt.show()
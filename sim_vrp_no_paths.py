import networkx as nx
import pandas as pd
from typing import Optional
import os
from read_files import load_edgefile, load_flowfile, load_nodefile, project_root
import matplotlib.pyplot as plt
from matplotlib import animation

# --- Parameters & Profiles ---
customers = [918, 911, 500, 400, 300, 600]
deport = [10]
demand_value = 5

# Time-breakpoints in minutes since midnight
t1, t2, t3, t4 = 6.5*60, 8.5*60, 10*60, 12*60
t5, t6, t7, t8 = 16.5*60, 18*60, 20*60, 22*60

def demand(t: float) -> float:
    """
    Piecewise demand profile over the day.
    """
    low, medium, high = 0.1, 2, 5
    if t <= t1:
        return low
    elif t < t2:
        return low + (high-low)*(t-t1)/(t2-t1)
    elif t < t3:
        return high
    elif t <= t4:
        return high - (high-medium)*(t-t3)/(t4-t3)
    elif t <= t5:
        return medium
    elif t < t6:
        return medium + (high-medium)*(t-t5)/(t6-t5)
    elif t <= t7:
        return high
    elif t < t8:
        return low + (high-low)*(1 - (t-t7)/(t8-t7))
    else:
        return low

# Precompute demand profile if needed
t_values = range(24*60)
demand_profile = [demand(t) for t in t_values]

# --- Graph Construction ---
def graph_from_data(edges: pd.DataFrame, nodes: Optional[pd.DataFrame] = None) -> nx.DiGraph:
    G = nx.DiGraph()
    if nodes is not None:
        coords = {row["node"]: (row["x"], row["y"]) for row in nodes.to_dict('records')}
        for node, xy in coords.items():
            G.add_node(node, coordinates=xy)
    for row in edges.to_dict('records'):
        G.add_edge(row["init_node"], row["term_node"], **row)
    return G

'''
# --- Demand Application ---
def apply_demand(df: pd.DataFrame, times: range) -> dict[int, pd.DataFrame]:
    """
    Multiply each edge's volume by demand(t) for each time t.
    Returns a mapping {t: df_mod}.
    """
    results = {}
    for t in times:
        m = demand(t)
        df_mod = df.copy()
        df_mod['adjusted_volume'] = df_mod['volume'] * m
        results[t] = df_mod
    return results
'''

# --- Congestion Model Based on Demand ---
def get_congestion_time(attrs: dict, t_min: float) -> float:
    """
    Compute congestion delay as edge volume multiplied by demand at time t.
    """
    volume = attrs.get("volume", 0.0)
    return volume * demand(t_min) *0.001


def get_travel_time(attrs: dict, t_min: float) -> float:
    """
    Total travel time: free-flow time plus congestion delay.
    """
    free_time = attrs.get("free_flow_time", 0.0)
    return free_time + get_congestion_time(attrs, t_min)

'''
# Density bepalen
def density():
    pc = qc / uf # crit_density = max_flow / freeflowtime
    pj = # ???? hoe bereken/bepaal je dit ???
    w = qc / (pj - pc) # Congestion speed  
 
    if density =< crit_density:
        density = flow / freeflowtime
    else density > crit_density:
            density = flow / congestion speed
    return 
'''


# --- Main & Animation ---
if __name__ == "__main__":
    # Load Data
    edges_df = load_edgefile(os.path.join(project_root, "TransportationNetworks", "Chicago-Sketch", "ChicagoSketch_net.tntp"))
    nodes_df = load_nodefile(os.path.join(project_root, "TransportationNetworks", "Chicago-Sketch", "ChicagoSketch_node.tntp"))
    flow_df  = load_flowfile(os.path.join(project_root, "TransportationNetworks", "Chicago-Sketch", "ChicagoSketch_flow.tntp"))

    # Build graphs
    G_dir = graph_from_data(edges_df, nodes_df)
    G_und = nx.Graph(G_dir)
    for u, v, data in G_und.edges(data=True):
        data.setdefault('volume', 0.0)
    for r in flow_df.to_dict('records'):
        if G_und.has_edge(r["from"], r["to"]):
            G_und.edges[r["from"], r["to"]]['volume'] = r["volume"]

    pos = {n: data["coordinates"] for n, data in G_dir.nodes(data=True)}
    edges = list(G_und.edges())

    # Plot setup
    fig, ax = plt.subplots(figsize=(10,8))
    nx.draw_networkx_nodes(G_und, pos, node_size=1, ax=ax, node_color='gray')
    nx.draw_networkx_nodes(G_dir, pos, nodelist=customers, node_size=20, ax=ax, node_color='blue')
    nx.draw_networkx_nodes(G_dir, pos, nodelist=deport,    node_size=20, ax=ax, node_color='red')
    drawn = nx.draw_networkx_edges(G_und, pos, edgelist=edges, edge_color="0.8", ax=ax)

    # Title and timer text
    title = ax.set_title("Time: 00:00")
    timer_text = ax.text(
        0.02, 0.95,             # x, y in axes coords
        "Elapsed: 00:00:00",   # initial text
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    start_min = int(1*60)

    def update(frame):
        t_sim = start_min + frame
        # Update edge colors
        colors = []
        for u, v in edges:
            delay = get_congestion_time(G_und.edges[u,v], t_sim)
            val = max(0.0, 0.8 - (delay/15)*0.8)
            colors.append(str(val))
        drawn.set_color(colors)
        # Update title with clock time
        h, m = divmod(t_sim, 60)
        title.set_text(f"Time: {h:02d}:{m:02d}")
        # Update timer as elapsed real time
        hrs = frame // 60
        mins = frame % 60
        secs = 0  # since frame steps in minutes
        timer_text.set_text(f"Elapsed: {hrs:02d}:{mins:02d}:{secs:02d}")
        return drawn, title, timer_text

    anim = animation.FuncAnimation(
        fig, update,
        frames=range(120, 24*60, 10),  # frames stepping by 10 minutes
        interval=200,
        blit=True
    )
    ax.set_aspect('equal')
    plt.xlabel("X coordinate"); plt.ylabel("Y coordinate")
    plt.tight_layout()
    plt.show()

    # Demand profile plot
    fig2, ax2 = plt.subplots()
    ax2.plot(list(t_values), demand_profile)
    ax2.set(
        xlabel='Minutes since midnight',
        ylabel='Demand',
        title='Daily Demand'
    )
    plt.tight_layout()
    plt.show()

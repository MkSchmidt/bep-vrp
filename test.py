import networkx as nx
import pandas as pd
from typing import Optional
import os
from read_files import load_edgefile, load_flowfile, load_nodefile, project_root
import matplotlib.pyplot as plt
from matplotlib import animation

# --- Parameters & Profiles ---
customers = [918,782,  911, 500, 400, 300, 600]
depot = [1, 547]
demand_value = 5
edge_example = 392, 713 #388,390 #918,782 #
B = 0.015 # is this correct parameter to use


# Time-breakpoints in minutes since midnight
t1, t2, t3, t4 = 4*60, 8.5*60, 10*60, 12*60
t5, t6, t7, t8 = 16.5*60, 18*60, 20*60, 22*60


# Demand function
def demand(t: float) -> float:
    low, medium, high = 0.1, 0.5, 1.1
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

# --- Congestion Model Based on Demand ---
def get_flow(attrs: dict, t_min: float) -> float:
    volume = attrs.get("volume")
    return volume * demand(t_min)

# Determine Critical Density for each edge
def get_critical_density(attrs: dict):
    capacity = attrs.get("capacity")
    free_time = attrs.get("free_flow_time")
    free_time = 5 if free_time == 0 else free_time
    length = attrs.get("length")
    ff_speed = length / free_time
    return capacity / ff_speed #klopt dit??

# Determine Density
'''
def get_density(attrs: dict):
    length = attrs.get("length")
    capacity = attrs.get("capacity")
    density = capacity / length
    return density 
'''

# Determine Congestion speed 
def congestion_speed(attrs: dict):
    capacity  = attrs.get("capacity")
    length = attrs.get("length")
    pc = get_critical_density(attrs) 
    pj = 5 * pc # ???? hoe bereken/bepaal je dit ???
    w = capacity / (pj - pc) # Congestion speed
    return w

# Needed foir Visualization
def congestion_time(attrs: dict, t_min, B):
    flow = get_flow(attrs, t_min)
    capacity  = attrs.get("capacity")
    if flow <= capacity:
        return 0.001
    else:
        return(flow - capacity) * B

# Travel time bepalen
def get_travel_time(attrs: dict, t_min, B):
    freetime = attrs.get("free_flow_time")
    capacity  = attrs.get("capacity")
    flow = get_flow(attrs, t_min)
    if flow <= capacity:
        travel_time = freetime
    else:
        travel_time = freetime + (flow - capacity) * B    #length / w
    return travel_time

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
        data.setdefault('volume')
    for r in flow_df.to_dict('records'):
        if G_und.has_edge(r["from"], r["to"]):
            G_und.edges[r["from"], r["to"]]['volume'] = r["volume"]

    pos = {n: data["coordinates"] for n, data in G_dir.nodes(data=True)}
    edges = list(G_und.edges())

    # Plot setup
    fig, ax = plt.subplots(figsize=(10,8))
    nx.draw_networkx_nodes(G_und, pos, node_size=1, ax=ax, node_color='gray')
    nx.draw_networkx_nodes(G_dir, pos, nodelist=customers, node_size=20, ax=ax, node_color='blue')
    nx.draw_networkx_nodes(G_dir, pos, nodelist=depot,    node_size=20, ax=ax, node_color='red')
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
            delay = congestion_time(G_und.edges[u,v], t_sim, B) #get_flow(G_und.edges[u,v], t_sim)
            val = max(0.0, 0.8 - (delay/ 10)*0.8)
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
        frames=range(180, 24*60, 15),  # frames stepping by 10 minutes
        interval=200,
        blit=True
    )
    ax.set_aspect('equal')
    plt.xlabel("X coordinate"); plt.ylabel("Y coordinate")
    plt.tight_layout()
    plt.show()
'''
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
'''
u, v = edge_example

# Tabel with sample times
if G_und.has_edge(u, v):
    edge_attrs = G_und.edges[u, v]
    records = []

    for t in [t1, t2, t3, t4, t5, t6, t7, t8]:
        records.append({
            "Time (min)": t,
            "Flow (veh/h)": get_flow(edge_attrs, t),
            "Critical Density (veh/km)": get_critical_density(edge_attrs),
            "Travel Time (min)": get_travel_time(edge_attrs, t, B),
            "Congestion Speed (m/s)": congestion_speed(edge_attrs),
        })

    df = pd.DataFrame(records)
    print(df)

# Travel time profile plot for example edge (u,v)
if G_und.has_edge(u, v):
    travel_time_profile = [get_travel_time(G_und.edges[u, v], t, B) for t in t_values]

    fig3, ax3 = plt.subplots()
    ax3.plot(t_values, travel_time_profile, label=f"Travel Time on edge ({u},{v})")
    ax3.set(
        xlabel='Minutes since midnight',
        ylabel='Travel time [min]',
        title=f'Travel Time over Day for edge ({u},{v})'
    )
    ax3.legend()
    plt.tight_layout()
    plt.show()


import networkx as nx
import pandas as pd
from typing import Optional
import os
from read_files import load_edgefile, load_flowfile, load_nodefile, project_root
import matplotlib.pyplot as plt
from matplotlib import animation 
import os, json

with open("zones.json", "r", encoding="utf-8") as f:
    data = json.load(f)
categories = data["zone_classification"]

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

fig, ax1 = plt.subplots(figsize=(10,8))
# Plot setup

edges_coll = nx.draw_networkx_edges(
    G_und, pos,
    edgelist=G_und.edges(),
    edge_color='lightgray',
    width=0.5,
    ax=ax1
)
edges_coll.set_zorder(1)

  # Create 10 distinct colors (tab10 has 10 colors; we'll use only as many as needed)
cmap = plt.colormaps['tab10']

# Define your buckets: (start, end)
buckets = [(i, min(i+99, 933)) for i in range(1, 934, 100)]
# buckets = [(1,100), (101,200), …, (901,933)]

for idx, (start, end) in enumerate(buckets):
    nlist = [n for n in G_dir.nodes() if start <= n <= end]
    nodes_coll = nx.draw_networkx_nodes(
        G_dir, pos,
        nodelist=nlist,
        node_size=20,
        node_color=[cmap(idx)],
        ax=ax1,
        label=f"{start}-{end}"
    )
    nodes_coll.set_zorder(2)

ax1.set_aspect('equal')
ax1.legend(scatterpoints=1, fontsize=8, title="Node ranges")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.tight_layout()
plt.show(block =False)

fig, ax2 = plt.subplots(figsize=(10, 8))

# Draw edges beneath everything
edges_coll = nx.draw_networkx_edges(
    G_und, pos,
    edge_color="lightgray",
    width=0.5,
    ax=ax2
)
edges_coll.set_zorder(1)

# Pick a categorical colormap with enough entries
cmap = plt.colormaps["tab20"]  # up to 20 distinct colors

# Draw each zone
for idx, (zone_name, nodelist) in enumerate(categories.items()):
    # filter only nodes present in the graph
    present = [n for n in nodelist if n in G_dir]
    if not present:
        continue
    color = cmap(idx)
    coll = nx.draw_networkx_nodes(
        G_dir, pos,
        nodelist=present,
        node_size=20,
        node_color=[color],
        label=zone_name,
        ax=ax2
    )
    coll.set_zorder(2)

# Final touches
ax2.set_aspect("equal")
ax2.legend(scatterpoints=1, fontsize=8, title="Zone Classification")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.tight_layout()
plt.show()
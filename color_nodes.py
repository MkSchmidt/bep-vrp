import networkx as nx
import pandas as pd
from typing import Optional
import os
from read_files import load_edgefile, load_flowfile, load_nodefile, project_root
import matplotlib.pyplot as plt
import json

# Load grid-based node classification
with open("nodes_grid.json", "r", encoding="utf-8") as f:
    categories = json.load(f)

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
    for u, v, data_edge in G_und.edges(data=True):
        data_edge.setdefault('volume')
    for r in flow_df.to_dict('records'):
        if G_und.has_edge(r["from"], r["to"]):
            G_und.edges[r["from"], r["to"]]['volume'] = r["volume"]

    pos = {n: data["coordinates"] for n, data in G_dir.nodes(data=True)}
    edges = list(G_und.edges())

    # Compute bounding box for grid
    xs = [coord[0] for coord in pos.values()]
    ys = [coord[1] for coord in pos.values()]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    # Define grid resolution
    rows, cols = 10, 10
    x_lines = [minx + i*(maxx-minx)/cols for i in range(cols+1)]
    y_lines = [miny + i*(maxy-miny)/rows for i in range(rows+1)]

    # First plot: Node ID buckets
    fig, ax1 = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_edges(
        G_und, pos,
        edgelist=edges,
        edge_color='lightgray',
        width=0.5,
        ax=ax1
    )
    # Draw grid lines
    for x in x_lines:
        ax1.axvline(x, linestyle='--', linewidth=0.8, zorder=0)
    for y in y_lines:
        ax1.axhline(y, linestyle='--', linewidth=0.8, zorder=0)
    # Enforce axis limits
    ax1.set_xlim(minx, maxx)
    ax1.set_ylim(miny, maxy)
    # Ticks at grid lines
    ax1.set_xticks(x_lines)
    ax1.set_yticks(y_lines)
    ax1.set_xticklabels([f"{x:.1f}" for x in x_lines], rotation=45, fontsize=8)
    ax1.set_yticklabels([f"{y:.1f}" for y in y_lines], fontsize=8)

    # Color nodes by ID buckets
    cmap = plt.colormaps['tab10']
    max_node = max(G_dir.nodes()) if G_dir.nodes() else 0
    buckets = [(i, min(i+99, max_node)) for i in range(1, max_node+1, 100)]
    for idx, (start, end) in enumerate(buckets):
        nlist = [n for n in G_dir.nodes() if start <= n <= end]
        nx.draw_networkx_nodes(
            G_und, pos,
            nodelist=nlist,
            node_size=20,
            node_color=[cmap(idx)],
            ax=ax1,
            label=f"{start}-{end}"
        )
    ax1.set_aspect('equal')
    ax1.legend(scatterpoints=1, fontsize=8, title="Node ranges")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.tight_layout()
    plt.show(block=False)

    # Second plot: Grid cells with 6-color pattern
    fig, ax2 = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_edges(
        G_und, pos,
        edge_color='lightgray',
        width=0.5,
        ax=ax2
    )
    # Draw grid lines
    for x in x_lines:
        ax2.axvline(x, linestyle='--', linewidth=0.8, zorder=0)
    for y in y_lines:
        ax2.axhline(y, linestyle='--', linewidth=0.8, zorder=0)
    ax2.set_xlim(minx, maxx)
    ax2.set_ylim(miny, maxy)
    ax2.set_xticks(x_lines)
    ax2.set_yticks(y_lines)
    ax2.set_xticklabels([f"{x:.1f}" for x in x_lines], rotation=45, fontsize=8)
    ax2.set_yticklabels([f"{y:.1f}" for y in y_lines], fontsize=8)

    # 6-color pattern based on (row%2, col%3)
    num_colors = 6
    cmap2 = plt.cm.get_cmap('tab20', num_colors)
    for cell, nodelist in categories.items():
        parts = cell.split('_col')
        r = int(parts[0].replace('row',''))
        c = int(parts[1])
        color_idx = (r % 2) * 3 + (c % 3)
        color = cmap2(color_idx)
        present = [n for n in nodelist if n in G_und]
        if present:
            nx.draw_networkx_nodes(
                G_und, pos,
                nodelist=present,
                node_size=20,
                node_color=[color],
                ax=ax2,
                label=cell
            )
    ax2.set_aspect('equal')
    #ax2.legend(scatterpoints=1, fontsize=8, title='Grid Cells')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.tight_layout()
    plt.show()
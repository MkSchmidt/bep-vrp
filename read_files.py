import os
import pandas as pd
import re

project_root = os.path.dirname(__file__)

def load_nodefile(name):
    nodefile = os.path.join(project_root, "TransportationNetworks", name, "friedrichshain-center_node.tntp")
    node_df = pd.read_csv(nodefile, sep=r"\s+", index_col="Node")

    trimmed = [s.strip().lower() for s in node_df.columns]
    node_df.columns = trimmed

    if ';' in node_df.columns:
        node_df.drop([';'], axis=1, inplace=True)

    return node_df

def load_edgefile(name):
    edgefile = os.path.join(project_root, "TransportationNetworks", name, "friedrichshain-center_net.tntp")
    edge_df = pd.read_csv(edgefile, skiprows=8, sep='\t')

    trimmed = [s.strip().lower() for s in edge_df.columns]
    edge_df.columns = trimmed

    edge_df.drop(['~', ';'], axis=1, inplace=True)

    with open(edgefile, "r") as f:
        attrs = re.findall(r"(?m)^\s*<(.*)>\s*(.*\S)\s*$", f.read())
    edge_df.attrs = dict(attrs)

    print("✅ Edges:")
    print(edge_df)
    return edge_df

# Load both files
nodes = load_nodefile("Berlin-Friedrichshain")
edges = load_edgefile("Berlin-Friedrichshain")

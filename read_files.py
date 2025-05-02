import os
import pandas as pd
import re

project_root = os.path.dirname(__file__)

def load_nodefile(name):
    nodefile = os.path.join(project_root, "TransportationNetworks", name, "friedrichshain-center_node.tntp")
    node_df = pd.read_csv(nodefile, skiprows=1, sep='\t')

    trimmed = [s.strip().lower() for s in node_df.columns]
    node_df.columns = trimmed

    if ';' in node_df.columns:
        node_df.drop([';'], axis=1, inplace=True)

    with open(nodefile, "r") as f:
        attrs = re.findall(r"(?m)^\s*<(.*)>\s*(.*\S)\s*$", f.read())
    node_df.attrs = dict(attrs)

    print("✅ Nodes:")
    print(node_df)
    return node_df

def load_edgefile(name):
    edgefile = os.path.join(project_root, "TransportationNetworks", name, "friedrichshain-center_net.tntp")
    edge_df = pd.read_csv(edgefile, skiprows=8, sep='\t')

    trimmed = [s.strip().lower() for s in edge_df.columns]
    edge_df.columns = trimmed

    if ';' in edge_df.columns:
        edge_df.drop([';'], axis=1, inplace=True)

    with open(edgefile, "r") as f:
        attrs = re.findall(r"(?m)^\s*<(.*)>\s*(.*\S)\s*$", f.read())
    edge_df.attrs = dict(attrs)

    print("✅ Edges:")
    print(edge_df)
    return edge_df
def load_tripsfile(name):
    tripsfile = os.path.join(project_root, "TransportationNetworks", name, "friedrichshain-center_trips.tntp")
    trips_df = pd.read_csv(tripsfile, skiprows=8, sep='\t')

    trimmed = [s.strip().lower() for s in trips_df.columns]
    trips_df.columns = trimmed

    if ';' in trips_df.columns:
        trips_df.drop([';'], axis=1, inplace=True)

    with open(tripsfile, "r") as f:
        attrs = re.findall(r"(?m)^\s*<(.*)>\s*(.*\S)\s*$", f.read())
    trips_df.attrs = dict(attrs)

    print("✅ Trips:")
    print(trips_df)
    return trips_df


# Load both files
nodes = load_nodefile("Berlin-Friedrichshain")
edges = load_edgefile("Berlin-Friedrichshain")
trips = load_tripsfile("Berlin-Friedrichshain")

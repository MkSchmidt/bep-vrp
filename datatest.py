import os
import pandas as pd
import re
import json

project_root = os.path.dirname(__file__)

def load_edges(name):
    netfile = os.path.join(project_root, "TransportationNetworks", name, f"{name}_net.tntp")
    net = pd.read_csv(netfile, skiprows=8, sep='\t')

    trimmed= [s.strip().lower() for s in net.columns]
    net.columns = trimmed
    net.drop(['~', ';'], axis=1, inplace=True)
    
    # Metadata
    with open(netfile, "r") as f:
        attrs = re.findall("(?m)^\\s*\\<(.*)\\>\\s*(.*\\S)\\s*$", f.read())

    net.attrs = dict(attrs)
    return net

def load_anaheim_coordinates():
    nodefile = os.path.join(project_root, "TransportationNetworks", "Anaheim", "anaheim_nodes.geojson")
    with open(nodefile, "r") as f:
        node_dict = json.load(f)

    nodes = [
                {
                    "id": point["properties"]["id"],
                    "x": point["geometry"]["coordinates"][0],
                    "y": point["geometry"]["coordinates"][1]
                } for point in node_dict["features"]
            ]
    return nodes

anaheim_net = load_edges("Anaheim")
anaheim_nodes = load_anaheim_coordinates()


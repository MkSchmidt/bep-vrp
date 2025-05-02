import os
import pandas as pd
import re
import json

project_root = os.path.dirname(__file__)

def load_nodefile(path):
    node_df = pd.read_csv(path, sep=r"\s+")

    trimmed = [s.strip().lower() for s in node_df.columns]
    node_df.columns = trimmed

    if ';' in node_df.columns:
        node_df.drop([';'], axis=1, inplace=True)

    return node_df

def load_edgefile(path):
    edge_df = pd.read_csv(path, skiprows=8, sep='\t')

    trimmed = [s.strip().lower() for s in edge_df.columns]
    edge_df.columns = trimmed

    edge_df.drop(['~', ';'], axis=1, inplace=True)

    with open(path, "r") as f:
        attrs = re.findall(r"(?m)^\s*<(.*)>\s*(.*\S)\s*$", f.read())
    edge_df.attrs = dict(attrs)

    return edge_df

def load_nodefile_geojson(path):
    with open(path, "r") as f:
        node_dict = json.load(f)

    nodes = [
                {
                    "node": point["properties"]["id"],
                    "x": point["geometry"]["coordinates"][0],
                    "y": point["geometry"]["coordinates"][1]
                } for point in node_dict["features"]
            ]
    return pd.DataFrame.from_records(nodes)

def read_folder(folder_path):
    files = os.listdir(folder_path)
    edgefiles = list(filter(lambda name: re.match(r"(?i).*_net\.tntp$", name), files))
    assert len(edgefiles) > 0, "TNTP voor het net moet bestaan"
    edge_path = os.path.join(folder_path, edgefiles[0])
    edges = load_edgefile(edge_path)

    nodefiles_tntp = list(filter(lambda name: re.match(r"(?i).*_node\.tntp$", name), files))
    if len(nodefiles_tntp) > 0:
        node_path = os.path.join(folder_path, nodefiles_tntp[0])
        nodes = load_nodefile(node_path)
    else:
        nodefiles_geojson = list(filter(lambda name: re.match(r"(?i).*_nodes\.geojson$", name), files))
        assert len(nodefiles_geojson) > 0, "node file moet bestaan"
        node_path = os.path.join(folder_path, nodefiles_geojson[0])
        nodes = load_nodefile_geojson(node_path)

    return edges, nodes

def test_folder_reading():
    network_dir = os.path.join(project_root, "TransportationNetworks")
    
    for item in os.listdir(network_dir):
        item_path = os.path.join(network_dir, item)
        if os.path.isfile(item_path): continue
        
        try:
            read_folder(item_path)
        except AssertionError as e:
            print(f"Shit foutgegaan in mapje {item}: {str(e)}")
        except Exception as e:
            print(f"ANDERE shit foutgegeaan in mapje {item}: {str(e)}")


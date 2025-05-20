import os
import pandas as pd
import re
import json
import io

project_root = os.path.dirname(__file__)

def load_nodefile(path):
    node_df = pd.read_csv(path, sep=r"\s+")

    trimmed = [s.strip().lower() for s in node_df.columns]
    node_df.columns = trimmed

    if ';' in node_df.columns:
        node_df.drop([';'], axis=1, inplace=True)

    return node_df

def load_edgefile(path):
    lines = open(path, 'r').read().splitlines()
    metadata_header = ''
    start = 0
    for i, l in enumerate(lines):
        m = re.match(r'^\s*<original header>\s*(.*)$', l, re.IGNORECASE)
        if m:
            metadata_header = m.group(1).strip()
        if l.strip().lower() == '<end of metadata>':
            start = i + 1
            break
    header_idx = None
    for i in range(start, len(lines)):
        txt = lines[i].lstrip('~').strip()
        if not txt or txt.startswith('<'):
            continue
        tokens = re.split(r'\s+', txt.rstrip(';'))
        if 'length' in [t.lower() for t in tokens]:
            header_idx = i
            break
    if header_idx is None:
        for i in range(len(lines) - 1):
            txt = lines[i].lstrip('~').strip()
            if not txt or txt.startswith('<'):
                continue
            if 'length' in txt.lower() and re.search(r'\d', lines[i + 1]):
                header_idx = i
                break
    if header_idx is None:
        raise ValueError(f"Could not locate edge-file header in {path}")
    raw_header = lines[header_idx].lstrip('~').strip().rstrip(';')
    tokens = [tok for tok in re.split(r'\s+', raw_header) if tok and tok != ';']
    seen = {}
    cols = []
    for tok in tokens:
        name = re.sub(r'[^0-9A-Za-z_]', '_', tok).lower().strip('_')
        count = seen.get(name, 0)
        seen[name] = count + 1
        cols.append(name if count == 0 else f"{name}_{count}")
    data_buf = io.StringIO('\n'.join(lines[header_idx + 1:]))
    df = pd.read_csv(data_buf, delim_whitespace=True, comment=';', header=None, names=cols)
    df.attrs['original_header'] = metadata_header
    return df



def load_flowfile(p):
    L = open(p).read().splitlines()
    i = next(j for j, l in enumerate(L) if l.strip().lower().startswith('from'))
    buf = '\n'.join(L[i:])
    return (
        pd.read_csv(io.StringIO(buf),
                    delim_whitespace=True,
                    comment=';',
                    header=0)
          .rename(str.lower, axis=1)
    )
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

def load_tripsfile(file_path):
    trips = []
    origin = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # update origin when we see it
            if line.startswith('Origin'):
                _, origin = line.split(maxsplit=1)
                continue

            # only look for dest:val pairs after an origin
            if origin and ':' in line:
                for part in line.split(';'):
                    if ':' not in part:
                        continue
                    dest, val = part.split(':', 1)
                    try:
                        count = float(val.strip())
                    except ValueError:
                        # skip dates or any non-numeric entries
                        continue
                    if count > 0:
                        trips.append((origin, dest.strip(), int(count)))
    return trips

def load_flowfile(file_path):
    df = pd.read_csv(file_path, sep=r"\s+")
    
    trimmed = [s.strip().lower() for s in df.columns]
    df.columns = trimmed
    return df

def read_folder(folder_path):
    files = os.listdir(folder_path)
    edgefiles = [f for f in files if f.lower().endswith('net.tntp')]
    assert edgefiles, "TNTP voor het net moet bestaan"
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
    
    tripsfiles = list(filter(lambda name: re.match(r"(?i).*_trips\.tntp$", name), files))
    assert len(tripsfiles) > 0, "TNTP voor de trips moet bestaan"

    trips_path = os.path.join(folder_path, tripsfiles[0])
    trips = load_tripsfile(trips_path)

    flowfiles = [f for f in files if re.match(r"(?i).*_flow\.tntp$", f)]
    assert flowfiles, "Flow file must exist"
    flow_path = os.path.join(folder_path, flowfiles[0])
    flows = load_flowfile(flow_path)

    return edges, nodes, trips, flows

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


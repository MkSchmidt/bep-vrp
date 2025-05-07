import os
import pandas as pd
import re
import math
import subprocess
from read_files import read_folder

project_root = os.path.dirname(__file__)

def write_sumo_nodes(df, filename):
    df.columns = [c.strip().lower().lstrip(';') for c in df.columns]
    if ';' in df.columns:
        df.drop([';'], axis=1, inplace=True)

    ref_lat = df.iloc[0]["y"]
    ref_lon = df.iloc[0]["x"]
    earth_radius = 6371000

    def latlon_to_meters(lat, lon):
        lat_rad = math.radians(lat)
        ref_lat_rad = math.radians(ref_lat)
        delta_lat = lat - ref_lat
        delta_lon = lon - ref_lon
        x = delta_lon * (math.pi / 180) * earth_radius * math.cos(ref_lat_rad)
        y = delta_lat * (math.pi / 180) * earth_radius
        return x, y

    with open(filename, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<network version="1.0">\n')  # Adding network version
        f.write("<nodes>\n")
        for _, row in df.iterrows():
            x, y = latlon_to_meters(row["y"], row["x"])
            f.write(f'  <node id="{int(row["node"])}" x="{x:.2f}" y="{y:.2f}" />\n')
        f.write("</nodes>\n")
        f.write('</network>\n')

def write_sumo_edges(df, filename):
    df.columns = [col.strip().lower() for col in df.columns]
    with open(filename, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<network version="1.0">\n')  # Adding network version
        f.write("<edges>\n")
        for i, row in df.iterrows():
            f.write(f'  <edge id="{i}" from="{int(row["init_node"])}" to="{int(row["term_node"])}" />\n')
        f.write("</edges>\n")
        f.write('</network>\n')

def write_sumo_xml_plain(trips, output_path):
    """
    trips: list of (from_node, to_node, num_trips)
    Produces a <trips> (or <routes>) file where every individual trip
    is sorted by its 'depart' time to satisfy SUMO's requirements.
    """
    # 1) Build a flat list of every single trip
    trip_list = []
    trip_counter = 0
    for from_node, to_node, num in trips:
        if num <= 0:
            continue
        for i in range(num):
            depart_time = i * 10
            trip_id     = f"{from_node}-{to_node}-{trip_counter}"
            trip_counter += 1
            trip_list.append({
                "id":     trip_id,
                "depart": depart_time,
                "from":   from_node,
                "to":     to_node
            })

    # 2) Sort strictly by depart, then by id (to remove SUMO warnings)
    trip_list.sort(key=lambda t: (t["depart"], t["id"]))

    # 3) Write them out
    with open(output_path, 'w') as f:
        f.write('<trips>\n')
        for t in trip_list:
            f.write(
                f'  <trip '
                f'id="{t["id"]}" '
                f'depart="{t["depart"]}" '
                f'from="{t["from"]}" '
                f'to="{t["to"]}" '
                f'number="1"/>\n'
            )
        f.write('</trips>\n')

def generate_net_xml(node_file, edge_file, output_file="network.net.xml"):
    result = subprocess.run([
        "netconvert",
        f"--node-files={node_file}",
        f"--edge-files={edge_file}",
        f"-o={output_file}"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print("Error running netconvert:")
        print(result.stderr)
    else:
        print("Successfully generated network.net.xml")

def create_sumo_config(network_file, trips_file, config_filename):
    with open(config_filename, 'w') as f:
        f.write(f'<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(f'<configuration xmlns="http://www.eclipse.org/otf/1.0">\n')
        f.write(f'    <input>\n')
        f.write(f'        <net-file value="{network_file}"/>\n')
        f.write(f'        <route-files value="{trips_file}"/>\n')
        f.write(f'    </input>\n')
        f.write(f'    <simulation>\n')
        f.write(f'        <time-to-teleport value="0"/>\n')
        f.write(f'        <step-length value="1.0"/>\n')
        f.write(f'    </simulation>\n')
        f.write(f'</configuration>\n')

def convert_folder(input_folder, output_folder):
    edges, nodes, trips, flows = read_folder(input_folder)

    # 1) DEBUG: what do we really have?
    print("RAW flow columns ➜", flows.columns.tolist())

    # 2) normalize: strip whitespace, lowercase, drop leading semicolons
    flows.columns = [c.strip().lower().lstrip(';') for c in flows.columns]
    print("NORMALIZED flow columns ➜", flows.columns.tolist())

    # 3) rename to match your edges DF
    flows = flows.rename(columns={
        "from":   "init_node",
        "to":     "term_node",
        "volume": "flow"
    })
    print("RENAMED flow columns ➜", flows.columns.tolist())

    # 4) now it’s safe to pick out init_node, term_node, flow
    edges = edges.merge(
        flows[["init_node","term_node","flow"]],
        on=["init_node","term_node"],
        how="left"
    ).fillna({"flow": 0})


    os.makedirs(output_folder, exist_ok=True)
    
    common_name = os.path.basename(input_folder).lower()
    nodes_name = os.path.join(output_folder, f"{common_name}.nod.xml")
    edges_name = os.path.join(output_folder, f"{common_name}.edg.xml")
    trips_name = os.path.join(output_folder, f"{common_name}.rou.xml")
    net_name = os.path.join(output_folder, f"{common_name}.net.xml")
    write_sumo_nodes(nodes, nodes_name)
    write_sumo_edges(edges, edges_name)
    write_sumo_xml_plain(trips, trips_name)

    create_sumo_config(net_name, trips_name, os.path.join(output_folder, f"{common_name}.sumocfg"))
    generate_net_xml(nodes_name, edges_name, net_name)

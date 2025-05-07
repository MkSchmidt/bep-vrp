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
    # normalize the TNTP headers
    df.columns = [c.strip().lower().lstrip(';') for c in df.columns]

    FT_TO_M  = 0.3048   # feet → meters
    MIN_TO_S = 60.0     # minutes → seconds

    with open(filename, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<network version="1.0">\n')
        f.write("  <edges>\n")

        for _, row in df.iterrows():
            u = int(row["init_node"])
            v = int(row["term_node"])
            # convert units
            length_m  = float(row["length"]) * FT_TO_M
            speed_mps = float(row["speed"])  * FT_TO_M / MIN_TO_S
            lanes     = int(row.get("lanes", 1))

            # write out the edge with speed & lane count
            f.write(
                f'    <edge id="{u}_{v}" from="{u}" to="{v}" '
                f'speed="{speed_mps:.2f}" '
                f'numLanes="{lanes}"/>\n'
            )

        f.write("  </edges>\n")
        f.write("</network>\n")

def write_sumo_xml_plain(trips, origin_map, dest_map, output_path):
    """
    trips: list of (from_node, to_node, num_trips)
    origin_map: node_id -> edge_id, dest_map likewise
    Writes a <routes> file containing <flow> entries that DUAROUTER will read.
    """
    with open(output_path, 'w') as f:
        f.write('<routes>\n')
        for orig_node, dest_node, num in trips:
            if num <= 0:
                continue
            from_edge = origin_map.get(orig_node)
            to_edge   = dest_map.get(dest_node)
            if not from_edge or not to_edge:
                continue
            # create one flow with all trips
            flow_id = f"{orig_node}-{dest_node}"
            # here we span the entire simulation, e.g. from t=0 to t=3600
            f.write(
                f'  <flow id="{flow_id}" '
                f'begin="0" end="3600" '
                f'from="{from_edge}" to="{to_edge}" '
                f'number="{int(num)}"/>\n'
            )
        f.write('</routes>\n')

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

    # ── NEW: cast origin/dest to int so they match your maps ──
    trips = [(int(orig), int(dest), num) for orig, dest, num in trips]

    # 1) build a mapping from node → edge_id
    edges['edge_id'] = edges.apply(
        lambda r: f"{int(r['init_node'])}_{int(r['term_node'])}",
        axis=1
    )
    origin_map = edges.groupby('init_node')['edge_id'].first().to_dict()
    dest_map   = edges.groupby('term_node')['edge_id'].first().to_dict()

    # 2) prepare output folder & filenames
    os.makedirs(output_folder, exist_ok=True)
    common_name  = os.path.basename(input_folder).lower()
    nodes_name   = os.path.join(output_folder, f"{common_name}.nod.xml")
    edges_name   = os.path.join(output_folder, f"{common_name}.edg.xml")
    net_name     = os.path.join(output_folder, f"{common_name}.net.xml")
    trips_name   = os.path.join(output_folder, f"{common_name}.rou.xml")
    routed_name  = os.path.join(output_folder, f"{common_name}.routed.rou.xml")
    cfg_name     = os.path.join(output_folder, f"{common_name}.sumocfg")

    # 3) write nodes and edges (with your lane speeds)
    write_sumo_nodes(nodes, nodes_name)
    write_sumo_edges(edges, edges_name)

    # 4) build the SUMO network
    generate_net_xml(nodes_name, edges_name, net_name)

    # 5) write a <routes> file of <flow> entries that reference valid edge IDs
    write_sumo_xml_plain(trips, origin_map, dest_map, trips_name)

    # 6) invoke DUAROUTER to turn those flows into edge‐based <vehicle> routes
    subprocess.run([
        "duarouter",
        "--net-file",     net_name,
        "--route-files",  trips_name,
        "--output-file",  routed_name,
        "--ignore-errors"
    ], check=True)

    # 7) generate a sumocfg pointing at your routed file
    create_sumo_config(net_name, routed_name, cfg_name)

    # 8) (optional) launch SUMO‐GUI automatically
    subprocess.run([
        "sumo-gui", "-c", cfg_name
    ], check=True)

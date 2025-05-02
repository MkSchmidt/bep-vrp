import os
import pandas as pd
import re
import math
import subprocess

project_root = os.path.dirname(__file__)

def load_netfile(name):
    netfile = os.path.join(project_root, "TransportationNetworks", name, f"{name}_net.tntp")
    net = pd.read_csv(netfile, skiprows=8, sep='\t')
    trimmed = [s.strip().lower() for s in net.columns]
    net.columns = trimmed
    net.drop(['~', ';'], axis=1, inplace=True)
    
    with open(netfile, "r") as f:
        attrs = re.findall("(?m)^\\s*\\<(.*)\\>\\s*(.*\\S)\\s*$", f.read())
    net.attrs = dict(attrs)
    return net

def parse_tntp(file_path):
    trips = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    origin = None
    for line in lines:
        line = line.strip()
        if line.startswith('Origin'):
            origin = line.split()[1]
        elif ':' in line:
            parts = line.split(';')
            for part in parts:
                if ':' in part:
                    dest, val = part.split(':')
                    dest = dest.strip()
                    val = float(val.strip())
                    if val > 0:
                        trips.append((origin, dest, int(val)))
    return trips

def load_nodefile(name):
    netfile = os.path.join(project_root, "TransportationNetworks", name, f"{name}_node.tntp")
    net = pd.read_csv(netfile, sep='\t')
    trimmed = [s.strip().lower() for s in net.columns]
    net.columns = trimmed
    net.drop([';'], axis=1, inplace=True)
    
    with open(netfile, "r") as f:
        attrs = re.findall("(?m)^\\s*\\<(.*)\\>\\s*(.*\\S)\\s*$", f.read())
    net.attrs = dict(attrs)
    return net

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
    with open(output_path, 'w') as f:
        f.write('<trips>\n')
        trip_counter = 0
        for from_node, to_node, num in trips:
            if num <= 0:
                continue
            for i in range(num):
                trip_id = f"{from_node}-{to_node}-{trip_counter}"
                trip_counter += 1
                f.write(f'    <trip id="{trip_id}" depart="{i * 10}" from="{from_node}" to="{to_node}" number="1"/>\n')
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

# Main execution
siouxfallsnet = load_netfile("SiouxFalls")
siouxfallsnode = load_nodefile("SiouxFalls")

write_sumo_nodes(siouxfallsnode, "nodes.nod.xml")
write_sumo_edges(siouxfallsnet, "edges.edg.xml")

# Generate the compiled network
generate_net_xml("nodes.nod.xml", "edges.edg.xml")

# Generate trips
trips = parse_tntp(os.path.join(project_root, "TransportationNetworks", "SiouxFalls", "SiouxFalls_trips.tntp"))
write_sumo_xml_plain(trips, "trips.rou.xml")

# Write config for SUMO
create_sumo_config('network.net.xml', 'trips.rou.xml', 'sumo_sim.sumocfg')

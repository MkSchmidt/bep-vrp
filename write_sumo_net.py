import os
import pandas as pd
import subprocess
import math
import re
from read_files import read_folder

def write_sumo_nodes(df, filename):
    # normalize column names
    df.columns = [c.strip().lower().lstrip(';') for c in df.columns]
    xs = df['x'].astype(float)
    ys = df['y'].astype(float)

    # if coordinates are “big” they must be TNTP feet → convert to meters,
    # otherwise assume they’re lat/lon in degrees
    if xs.abs().max() > 1000 or ys.abs().max() > 1000:
        def project(x, y):
            return x * 0.3048, y * 0.3048
    else:
        ref_lat = ys.iloc[0]
        ref_lon = xs.iloc[0]
        R = 6_371_000
        def project(x, y):
            dlat = math.radians(y - ref_lat)
            dlon = math.radians(x - ref_lon)
            return (
                dlon * R * math.cos(math.radians(ref_lat)),
                dlat * R
            )

    with open(filename, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<network version="1.0">\n')
        f.write('  <nodes>\n')
        for _, row in df.iterrows():
            nid = int(row['node'])
            xm, ym = project(float(row['x']), float(row['y']))
            f.write(f'    <node id="{nid}" x="{xm:.2f}" y="{ym:.2f}" />\n')
        f.write('  </nodes>\n')
        f.write('</network>\n')


def write_sumo_edges(df, filename):
    FT_TO_M = 0.3048; MI_TO_M = 1609.34; MIN_TO_S = 60.0
    hdr = df.attrs.get('original_header', '')
    m = re.search(r'length\s*\(\s*([^)]+)\)', hdr, re.IGNORECASE)
    if m and 'mi' in m.group(1).lower(): LEN_TO_M = MI_TO_M
    elif m and 'ft' in m.group(1).lower(): LEN_TO_M = FT_TO_M
    else:
        maxval = df['length'].astype(float).max()
        LEN_TO_M = MI_TO_M if maxval < 10 else FT_TO_M
    with open(filename, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<network version="1.0">\n  <edges>\n')
        for _, row in df.iterrows():
            u = int(row['init_node']); v = int(row['term_node'])
            length_m = float(row['length']) * LEN_TO_M
            fftt = float(row.get('free_flow_time', row.get('fftt', 0)))
            speed_mps = length_m / (fftt * MIN_TO_S) if fftt > 0 else 45.0
            lanes = int(row.get('lanes', 1))
            f.write(
                f'    <edge id="{u}_{v}" from="{u}" to="{v}" '
                f'speed="{speed_mps:.2f}" numLanes="{lanes}" />\n'
            )
        f.write('  </edges>\n</network>\n')


def write_sumo_xml_plain(trips, origin_map, dest_map, output_path):
    with open(output_path, 'w') as f:
        f.write('<routes>\n')
        for o, d, n in trips:
            if n <= 0: continue
            fe = origin_map.get(o); te = dest_map.get(d)
            if not fe or not te: continue
            f.write(
                f'  <flow id="{o}-{d}" begin="0" end="3600" '
                f'from="{fe}" to="{te}" number="{n}"/>\n'
            )
        f.write('</routes>\n')


def generate_net_xml(node_file, edge_file, output_file):
    r = subprocess.run([
        'netconvert',
        f'--node-files={node_file}',
        f'--edge-files={edge_file}',
        f'-o={output_file}'
    ], capture_output=True, text=True)
    if r.returncode != 0: print('netconvert error:', r.stderr)


def create_sumo_config(network_file, trips_file, config_file):
    with open(config_file, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<configuration xmlns="http://www.eclipse.org/otf/1.0">\n  <input>\n')
        f.write(f'    <net-file value="{network_file}"/>\n')
        f.write(f'    <route-files value="{trips_file}"/>\n  </input>\n')
        f.write('  <simulation>\n    <time-to-teleport value="0"/>\n    <step-length value="1.0"/>\n  </simulation>\n</configuration>\n')


def convert_folder(input_folder, output_folder):
    edges, nodes, trips = read_folder(input_folder)
    trips = [(int(o), int(d), n) for o, d, n in trips]
    if 'tail' in edges.columns and 'head' in edges.columns:
        edges.rename(columns={'tail':'init_node','head':'term_node'}, inplace=True)
    elif 'from' in edges.columns and 'to' in edges.columns:
        edges.rename(columns={'from':'init_node','to':'term_node'}, inplace=True)
    edges['lanes'] = edges['capacity'].div(2000.0).round().astype(int).clip(1, 8)
    os.makedirs(output_folder, exist_ok=True)
    base = os.path.basename(input_folder).lower()
    nodes_file = os.path.join(output_folder, f"{base}.nod.xml")
    edges_file = os.path.join(output_folder, f"{base}.edg.xml")
    net_file   = os.path.join(output_folder, f"{base}.net.xml")
    trips_file = os.path.join(output_folder, f"{base}.rou.xml")
    routed_file= os.path.join(output_folder, f"{base}.routed.rou.xml")
    cfg_file   = os.path.join(output_folder, f"{base}.sumocfg")
    write_sumo_nodes(nodes, nodes_file)
    write_sumo_edges(edges, edges_file)
    generate_net_xml(nodes_file, edges_file, net_file)
    origin_map = {int(r['init_node']): f"{int(r['init_node'])}_{int(r['term_node'])}" for _, r in edges.iterrows()}
    dest_map   = {int(r['term_node']): f"{int(r['init_node'])}_{int(r['term_node'])}" for _, r in edges.iterrows()}
    write_sumo_xml_plain(trips, origin_map, dest_map, trips_file)
    subprocess.run([
        'duarouter',
        '--net-file', net_file,
        '--route-files', trips_file,
        '--output-file', routed_file,
        '--ignore-errors'
    ], check=True)
    create_sumo_config(net_file, routed_file, cfg_file)
    subprocess.run(['sumo-gui', '-c', cfg_file], check=True)
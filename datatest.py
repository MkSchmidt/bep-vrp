import os
import pandas as pd
import re

project_root = os.path.dirname(__file__)

def load_nodefile(name):
    nodefile = os.path.join(project_root, "TransportationNetworks", name, f"{name}_node.tntp")
    node = pd.read_csv(nodefile, sep='\t')

    trimmed= [s.strip().lower() for s in node.columns]
    node.columns = trimmed
    
    node.drop([';'], axis=1, inplace=True)
    
    # Metadata
    with open(nodefile, "r") as f:
        attrs = re.findall("(?m)^\\s*\\<(.*)\\>\\s*(.*\\S)\\s*$", f.read())

    net.attrs = dict(attrs)
    return node

def load_netfile(name):
    netfile = os.path.join(project_root, "TransportationNetworks", name, f"{name}_net.tntp")
    net = pd.read_csv(netfile, skiprows=8 , sep='\t')

    trimmed= [s.strip().lower() for s in net.columns]
    net.columns = trimmed

    # And drop the silly first andlast columns
    net.drop(['~',';'], axis=1, inplace=True)
    
    # Metadata
    with open(netfile, "r") as f:
        attrs = re.findall("(?m)^\\s*\\<(.*)\\>\\s*(.*\\S)\\s*$", f.read())

    net.attrs = dict(attrs)
    return net

net = load_netfile("SiouxFalls")
node = load_nodefile("SiouxFalls")
print(net,node)

print("<nodes>")
for index, row in node.iterrows():
    node_id = row['node']
    x = row['x']
    y = row['y']
    
    print(f'<node id="{node_id}" x="{x}" y="{y}" type="priority"/>')
print("<\nodes>")

with open("my_nodes.nod.xml", "w") as f:
    f.write(node_xml)

with open("my_edges.edg.xml", "w") as f:
    f.write(edge_xml)
    
netconvert --node-files=SiouxFalls.nod.xml --edge-files=SiouxFalls.edg.xml --output-file=SiouxFalls.net.xml


import os
import pandas as pd
import re

project_root = os.path.dirname(__file__)

def load_nodefile(name):
    nodefile = os.path.join(project_root, "TransportationNetworks", name, f"{name}_node.tntp")
    node = pd.read_csv(nodefile, sep='\t')

    trimmed= [s.strip().lower() for s in node.columns]
    node.columns = trimmed

    # And drop the silly first andlast columns
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


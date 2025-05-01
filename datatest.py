import os
import pandas as pd
import re

project_root = os.path.dirname(__file__)

def load_netfile(name):
    netfile = os.path.join(project_root, "TransportationNetworks", name, f"{name}_node.tntp")
    net = pd.read_csv(netfile, sep='\t')

    trimmed= [s.strip().lower() for s in net.columns]
    net.columns = trimmed

    # And drop the silly first andlast columns
    net.drop([';'], axis=1, inplace=True)
    
    # Metadata
    with open(netfile, "r") as f:
        attrs = re.findall("(?m)^\\s*\\<(.*)\\>\\s*(.*\\S)\\s*$", f.read())

    net.attrs = dict(attrs)
    return net

SiouxFalls = load_netfile("SiouxFalls")
print(SiouxFalls)

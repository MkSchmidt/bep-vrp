from read_files import read_folder, project_root
import os

def read_chicago():
    edges, nodes, trips, flows = read_folder(os.path.join(project_root, "TransportationNetworks", "Chicago-Sketch"))

    edges["length"] = edges["length"] * 1609.3                   #[m]
    edges["free_flow_time"] = edges["free_flow_time"] * 60       #[s]
    edges["capacity"] = edges["capacity"] / 3600                 #[veh/s]
    flows["volume"] = flows["volume"] / 3600                       #[veh/s]
    
    return edges, nodes, trips, flows

def read_anaheim():
    edges, nodes, trips, flows = read_folder(os.path.join(project_root, "TransportationNetworks", "Anaheim"))

    edges["length"] = edges["length"] * 0.3048                  #[m]
    edges["free_flow_time"] = edges["free_flow_time"] * 60      #[s]
    edges["capacity"] = edges["capacity"] / 3600                #[veh/s]
    flows["volume"] = (flows["volume"]) / 3600                    #[veh/s]

    return edges, nodes, trips, flows

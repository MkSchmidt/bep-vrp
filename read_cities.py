from read_files import read_folder, project_root
import os

def read_chicago():
    edges, nodes, trips, flows = read_folder(os.path.join(project_root, "TransportationNetworks", "Chicago-Sketch"))

    def get_parameters_chic(attrs):
        length = attrs.get("length") * 1.6093/1000              #[m]
        ff_time = attrs.get("free_flow_time") *60               #[s]
        capacity = attrs.get("capacity") /3600                  #[veh/s]
        flow = attrs.get("volume")/3600                         #[veh/s]

        return length, ff_time,capacity ,flow
    # TODO: eenheden omrekenen en edges returnen
    
    # Update edges with new parameters
    for edge in edges:
        attrs = edge.get('attributes', {}) 
        length, ff_time, capacity = get_parameters_chic(attrs)
        edge['length'] = length
        edge['free_flow_time'] = ff_time
        edge['capacity'] = capacity

    return edges, nodes, trips, flows


def read_anaheim():
    edges, nodes, trips, flows = read_folder(os.path.join(project_root, "TransportationNetworks", "Anaheim"))

    def get_parameters_ana(attrs):
        length = attrs.get("length") * 0.3048            #[m]
        ff_time = attrs.get("free_flow_time") *60        #[s]
        capacity = attrs.get("capacity") /3600           #[veh/s]
        flow = attrs.get("volume")/3600                  #[veh/s]
        return length, ff_time, capacity, flow
    
    # Update edges with new parameters
    for edge in edges:
        attrs = edge.get('attributes', {})  
        length, ff_time, capacity = get_parameters_ana(attrs)
        edge['length'] = length
        edge['free_flow_time'] = ff_time
        edge['capacity'] = capacity

    return edges, nodes, trips, flows

    pass

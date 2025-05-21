import networkx as nx
import pandas as pd
from typing import Optional
import os
from read_files import load_edgefile, load_flowfile, load_nodefile, project_root
import matplotlib.pyplot as plt
from matplotlib import animation

customers = [918, 911, 500, 400 , 300, 600] #Assign Customer location
depot = [10] #Assign Depot Location
demand_value = [5] # Assign demand

# Read Files
def graph_from_data(edges: pd.DataFrame, nodes: Optional[pd.DataFrame] = None) -> nx.DiGraph:
    graph = nx.DiGraph()
    if nodes is not None:
        node_list = [
            (node["node"], {"coordinates": (node["x"], node["y"])})
            for node in nodes.to_dict(orient="records")
        ]
        graph.add_nodes_from(node_list)

    edge_list = [
        (edge["init_node"], edge["term_node"], edge)
        for edge in edges.to_dict(orient="records")
    ]
    graph.add_edges_from(edge_list)
    return graph

# Load Data

edges_df: pd.DataFrame =  load_edgefile(os.path.join(project_root, "TransportationNetworks", "Chicago-Sketch", "ChicagoSketch_net.tntp"))
nodes_df = load_nodefile(os.path.join(project_root, "TransportationNetworks", "Chicago-Sketch", "ChicagoSketch_node.tntp"))
flow_df = load_flowfile(os.path.join(project_root, "TransportationNetworks", "Chicago-Sketch", "ChicagoSketch_flow.tntp"))


# Give current time as minutes since midnight
def get_added_travel_time(edge_attributes, t):

    volume = edge_attributes.get("volume", 0.0)
    capacity = edge_attributes.get("capacity", 1.0) 
    free_flow_time = edge_attributes.get("free_flow_time", 0.0)

    if capacity == 0: # Avoid division by zero; treat as infinitely congested or no flow
        return 999 # A large number for added time if capacity is zero and volume > 0
        
    t_tc = ((t % (24*60)) - 18*60) # Assume peak hour at 18:00
    
    if free_flow_time == 0:
        h_component = 0
    else:
        h_component = volume / capacity * free_flow_time * 3
        
    hump = h_component + min(-0.2*t_tc, h_component/90*t_tc)

    return max(0, hump)


def get_travel_time(edge_attributes, t):
    added_time = get_added_travel_time(edge_attributes, t)
    return edge_attributes.get("free_flow_time", 0) + added_time

# Assign a Demand to Customers

def assign_customer_demands(customers,demand_value):
    demands = {}
    for customer in customers:
        demands[customer] = demand_value 
    return demands

# Parameters
 # w = congestion_speed
 # qc = max_flow = capacity
 # pc = crit_density
 # uf = freeflowtime
 # tt = traveltime
 # q = flow

# Demand functie (___/----\____ vormig)
t1 = 6.5 * 60   # 7:00
t2 = 8.5 * 60   # 8:30
t3 = 10 * 60    # 10:00
t4 = 12 * 60
t5 = 16.5 * 60  # 16:30
t6 = 18 * 60  # 17:30
t7 = 20 * 60    # 20:00
t8 = 22 * 60    # 21:00

def demand(t, t1, t2, t3, t4):
    low,medium, high = 0.1, 0.5, 1.1

    if t <= t1:
         return low
    elif t < t2:
        return low + (high - low)*(t - t1)/(t2 -t1)
    elif t < t3:
        return high
    elif t <= t4:
        return high - (high - medium)*(t - t3)/(t4 - t3)
    elif t <= t5:
        return medium
    elif t < t6:
        return medium + (high - medium)*(t - t5)/(t6 - t5)
    elif t <= t7:
        return high
    elif t < t8:
        return low + (high -low)*(1- (t -t7)/(t8 -t7))
    else:
        return low

t_values = range(24*60)                   # 0…1439 minutes
demand_profile = [demand(t, t1,t2,t3,t4) for t in t_values]

'''
# Calculate the Flow of each edge 
def flow(edges_df, demand_profile, t_current):
    for edge in edges_df:
        volume = edge_df.get("volume", 0.0)
        flow_series = [volume * d for d in demand_profile]
        edge["flow_profile"] = flow_series
    return edges_df


t_current =  13*60 
edges_df = flow(edges_df, demand_profile, t_current)
third_edge = edges_df[2]
print(third_edge["flow_current"])



# Density bepalen
def density():
    pc = qc / uf # crit_density = max_flow / freeflowtime
    pj = # ???? hoe bereken/bepaal je dit ???
    w = qc / (pj - pc) # Congestion speed  
 
    if density =< crit_density:
        density = flow / freeflowtime
    else density > crit_density:
            density = flow / congestion speed
    return 
'''



if __name__ == "__main__":

    # Create directed graph from data
    G_directed = graph_from_data(edges_df, nodes=nodes_df)
    
    # Use an undirected representation for animation
    graph_for_animation = nx.Graph(G_directed)

    # Ensure 'volume' attribute exists for congestion calculation.
    for u, v, data in graph_for_animation.edges(data=True):
        if 'volume' not in data:
            data['volume'] = 0.0
    
    # Override/set 'volume' for edges that are in the flow file
    for flow_row in flow_df.to_dict(orient="records"):
        u, v = flow_row["from"], flow_row["to"]
        if graph_for_animation.has_edge(u, v):
            graph_for_animation.edges[u, v]["volume"] = flow_row["volume"]

    # Prepare for drawing
    node_positions = {node_id: data["coordinates"] for node_id, data in G_directed.nodes(data=True) if "coordinates" in data}
    edges_to_draw = list(graph_for_animation.edges())
    
    fig, ax = plt.subplots(figsize=(10, 8)) 
    
    # Draw nodes 

    nx.draw_networkx_nodes(graph_for_animation, node_positions, node_size= 1, ax=ax, node_color='gray') 
    
    # Draw Customer / Depot Nodes in Graph
    nx.draw_networkx_nodes(G_directed, node_positions, nodelist= customers, node_size = 20, ax = ax, node_color = 'blue')
    nx.draw_networkx_nodes(G_directed, node_positions, nodelist= depot, node_size = 20, ax = ax, node_color = 'red')

    # Initial drawing of edges
    drawn_edges = nx.draw_networkx_edges(graph_for_animation, node_positions, edgelist=edges_to_draw, edge_color="0.8", ax=ax)
    
    plot_title = ax.set_title("Time: 00:00") 

    animation_start_time_minutes = 15.5 * 60 

    def update_edge_colors(frame_offset_minutes):
        current_sim_time_minutes = animation_start_time_minutes + frame_offset_minutes

        edge_colors = []
        for u,v in edges_to_draw:
            edge_attr = graph_for_animation.edges[u,v]
            added_time = get_added_travel_time(edge_attr, current_sim_time_minutes)
            color_value = max(0.0, 0.8 - (added_time / 15.0) * 0.8)
            edge_colors.append(str(color_value))
        
        drawn_edges.set_color(edge_colors)
        hours = int(current_sim_time_minutes // 60) % 24
        minutes = int(current_sim_time_minutes % 60)
        plot_title.set_text(f"Time: {hours:02d}:{minutes:02d}")

        return [drawn_edges, plot_title]

    anim = animation.FuncAnimation(fig=fig, 
                                   func=update_edge_colors, 
                                   frames=range(0, 4*60, 10), 
                                   interval=200,  
                                   blit=True)
    
    ax.set_aspect('equal', adjustable='datalim') 
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    fig.tight_layout()
    plt.show()

fig, ax = plt.subplots()
ax.plot(t_values, demand_profile)
ax.set_xlabel('Time (minutes since midnight)')
ax.set_ylabel('Demand')
ax.set_title('Daily Demand Profile')
plt.tight_layout()
plt.show()
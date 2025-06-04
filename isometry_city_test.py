import torch
import networkx as nx
from isometry import place_points, reduce_dims, mse
from read_files import read_folder, project_root
from graph_sim import graph_from_data, get_path_length
import os
import random

edges, nodes, trips, flows = read_folder(os.path.join(project_root, "TransportationNetworks", "Anaheim"))

G = graph_from_data(edges, nodes)

nodelist = list(G.nodes)
n = len(nodelist)
paths = dict(nx.shortest_path(G, weight="free_flow_time"))
distances_ur = torch.zeros(n, n)

for i in range(n-1):
    for j in range(i+1, n):
        distances_ur[i, j] = get_path_length(G, paths[nodelist[i]][nodelist[j]])

distances = torch.max(distances_ur, distances_ur.T)

points = place_points(distances**2)

error = mse(points, distances**2)

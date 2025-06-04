from read_files import read_folder, project_root
import os

def read_chicago():
    edges, nodes, trips, flows = read_folder(os.path.join(project_root, "TransportationNetworks", "Chicago-Sketch"))
    # TODO: eenheden omrekenen en edges returnen

def read_anaheim():
    pass

from write_sumo_net import project_root, convert_folder
import os

city = "SiouxFalls"
citypath = (city+"path")
citypath = os.path.join(project_root, "TransportationNetworks", city)
convert_folder(citypath, os.path.join(project_root, "output"))

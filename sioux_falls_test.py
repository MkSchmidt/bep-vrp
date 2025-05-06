from write_sumo_net import project_root, convert_folder
import os

siouxfallspath = os.path.join(project_root, "TransportationNetworks", "SiouxFalls")
convert_folder(siouxfallspath, os.path.join(project_root, "output"))

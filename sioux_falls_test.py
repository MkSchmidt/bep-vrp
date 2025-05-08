from write_sumo_net import convert_folder
import os

if __name__ == "__main__":
    project_root = os.path.dirname(__file__)
    input_folder = os.path.join(project_root, "TransportationNetworks", "SiouxFalls")
    output_folder= os.path.join(project_root, "output")
    convert_folder(input_folder, output_folder)

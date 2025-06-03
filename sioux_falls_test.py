from write_sumo_net import convert_folder
import os

# Maps to choose:
# Simple :          SiouxFalls
# Intermediate :    Anaheim
# Complex :         Chicago-Sketch

if __name__ == "__main__":
    project_root = os.path.dirname(__file__)
    input_folder = os.path.join(project_root, "TransportationNetworks", "Anaheim")
    output_folder= os.path.join(project_root, "output")
    convert_folder(input_folder, output_folder)

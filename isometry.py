import numpy as np
import torch

def place_points(squared_distances):
    coordinates = torch.zeros((NUM_VERTICES, NUM_VERTICES - 1), dtype=torch.complex64)
    coordinates[1,0] = torch.sqrt(squared_distances[0,1])

    for i in range(2, NUM_VERTICES):
        d = squared_distances[0:i, i]
        l = torch.sum(coordinates[0:i]**2, dim=1)

        s = d - l
        r = s - torch.roll(s, 1)

        y = coordinates[0:i, 0:i]
        left = -2 * (y - torch.roll(y, 1, dims=0))

        r_reduced = r[:-1]
        left_reduced = left[:-1, :-1]

        # l @ x = r therefore x = l.inv @ r

        x_reduced = torch.inverse(left_reduced) @ r_reduced
        additional_distance = squared_distances[0][i] - torch.sum(x_reduced**2)
        x = torch.cat((x_reduced, torch.tensor([ np.roots([ 1, 0, -additional_distance.item() ])[0]])))
        coordinates[i,0:i] = x

    return coordinates

def distances_from_points(points):
    n = points.shape[0]
    differences = points.reshape(1, n, -1) - points.reshape(n, 1, -1)
    return torch.sum(differences**2, dim=2)

if __name__ == "__main__":
    NUM_VERTICES = 10

    squared_distances = torch.rand((NUM_VERTICES, NUM_VERTICES), dtype=torch.float32)
    squared_distances = torch.abs(squared_distances - squared_distances.T)
    
    analytical_points = place_points(squared_distances)
    
    random_points = torch.rand((NUM_VERTICES, NUM_VERTICES), dtype=torch.complex64)

    pairwise_d = lambda i, j: torch.sum((analytical_points[i] - analytical_points[j])**2)

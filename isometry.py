import numpy as np
import torch

def place_points(squared_distances):
    n = squared_distances.shape[0]
    coordinates = torch.zeros((n, n - 1), dtype=torch.complex64)
    coordinates[1,0] = torch.sqrt(squared_distances[0,1])

    for i in range(2, n):
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

def squared_distances_from_points(points):
    n = points.shape[0]
    differences = points.reshape(1, n, -1) - points.reshape(n, 1, -1)
    return torch.sum(differences**2, dim=2)

def mse(points, squared_distances):
    d = squared_distances_from_points(points) - squared_distances
    squared_error = torch.abs(d)**2
    return torch.mean(torch.triu(squared_error, diagonal=1)) * 2

def mse_absolute(points, squared_distances):
    d = torch.sqrt(squared_distances_from_points(points)) - torch.sqrt(squared_distances)
    squared_error = torch.abs(d)**2
    return torch.mean(torch.triu(squared_error, diagonal=1)) * 2

def approximate_points(squared_distances, dimensions=None, iterations=10000, lr=1e-6):
    points = torch.rand((NUM_VERTICES, dimensions or NUM_VERTICES - 1), dtype=torch.complex64, requires_grad = True)

    adam = torch.optim.Adam([points], lr=lr)
    mse_history = []
    for i in range(iterations):
        loss = mse(points, squared_distances)
        loss.backward()
        adam.step()
        mse_history.append(loss.detach().item())

    return points, mse_history

def reduce_dims(points, n=5):
    output = torch.zeros((points.shape[0], n), dtype=torch.complex64)
    covariance = torch.cov(points.T).abs()
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    return points @ eigenvectors.to(torch.complex64)[:, -n:]

if __name__ == "__main__":
    NUM_VERTICES = 10

    squared_distances = torch.rand((NUM_VERTICES, NUM_VERTICES), dtype=torch.float32)
    squared_distances = torch.abs(squared_distances - squared_distances.T)
    
    analytical_points = place_points(squared_distances)
    
    reduced_points = reduce_dims(analytical_points, n=9)

    random_points = torch.rand((NUM_VERTICES, NUM_VERTICES), dtype=torch.complex64)
    
    approx_points, mse_history = approximate_points(squared_distances, iterations=5000, lr=1e-4, dimensions = 4)

    print(f"Analytical loss: {mse_absolute(analytical_points, squared_distances)}, reduced loss: {mse_absolute(reduced_points, squared_distances)}, approximation_loss: {mse_absolute(approx_points, squared_distances)}, random loss: {mse_absolute(random_points, squared_distances)}")

    from matplotlib import pyplot as plt

    plt.plot(mse_history)
    #plt.show()

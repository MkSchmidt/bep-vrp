import numpy as np

NUM_VERTICES = 10

distances = np.random.rand(NUM_VERTICES, NUM_VERTICES)


coordinates = np.zeros((NUM_VERTICES, NUM_VERTICES - 1), dtype=complex)
coordinates[1][0] = distances[0][1]
pairwise_d = lambda i, j: np.sum((coordinates[i] - coordinates[j])**2)

for i in range(2, NUM_VERTICES):
    d = distances[0:i, i]
    l = np.sum(coordinates[0:i]**2, axis=1)

    s = d - l
    r = s - np.roll(s, 1)

    y = coordinates[0:i, 0:i]
    left = -2 * (y - np.roll(y, 1, axis=0))

    r_reduced = r[:-1]
    left_reduced = left[:-1, :-1]

    # l @ x = r therefore x = l.inv @ r

    x_reduced = np.linalg.inv(left_reduced) @ r_reduced
    additional_distance = distances[0][i] - np.sum(x_reduced**2)
    x = np.append(x_reduced, [ np.roots([ 1, 0, -additional_distance ])[0]])
    coordinates[i,0:i] = x

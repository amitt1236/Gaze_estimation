import numpy as np

relative = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]))
relativeT = lambda landmark, shape: (int(landmark.x * shape[1]), int(landmark.y * shape[0]), 0)


def distance(p1, p2, shape):
    dist = np.asarray(relative(p1, shape)) - np.asarray(relative(p2, shape))
    return np.sqrt(np.dot(dist, dist))


def rad(center, p1, p2, p3, p4, shape):
    return (distance(center, p1, shape) + distance(center, p2, shape) + distance(center, p3, shape) + distance(center,
                                                                                                               p4,
                                                                                                               shape)) / 4
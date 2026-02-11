from scipy.spatial.distance import cdist
import ot


def components_distance(triplet1, weights1, triplet2, weights2):
    distance_matrix = cdist(triplet1, triplet2, metric='euclidean')
    weights1 = weights1 / weights1.sum()
    weights2 = weights2 / weights2.sum()

    # Use POT library for Wasserstein distance
    return ot.emd2(weights1, weights2, distance_matrix)
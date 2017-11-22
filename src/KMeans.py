import numpy as np
import random

from src.Distance import EuclideanDistance

class KMeans:

    def __init__(self, _vectors, _k, _initial=None, _distance_alg=None):
        self.vectors = _vectors
        self.k = _k
        self.MAX_ITERATIONS = 30;
        self.MIN_AVERAGE_CHANGE = 0.001

        if (_distance_alg is None):
            self.d = EuclideanDistance()
        else:
            self.d = _distance_alg

        self.centroids = None

        if (_initial is not None and len(_initial) != _k):
            raise ValueError('initial centroids size does not match k')

        self.initial = _initial



    def cluster(self):
        return self.find_centers(self.vectors, self.k)

    def cluster_points(self, data, cetroids):
        clusters = {}

        for x in data:
            nearest = min([(i[0], self.d.distance(x, cetroids[i[0]])) \
                           for i in enumerate(cetroids)], key=lambda t: t[1])[0]

            try:
                clusters[nearest].append(x)
            except KeyError:
                clusters[nearest] = [x]

        return clusters



    def reevaluate_centers(self, clusters):
        newmu = []
        keys = sorted(clusters.keys())

        for k in keys:
            newmu.append(np.mean(clusters[k], axis=0))

        return newmu



    def has_converged(self, centroids, oldCentroids):
        distances = list()

        for i in range(0, len(centroids)):
            newCentroid = centroids[i]
            oldCentroid = oldCentroids[i]
            distance = self.d.distance(newCentroid, oldCentroid)
            distances.append(distance)

        print(distances)
        avgDistance = sum(distances) / float(len(distances))
        if (avgDistance < self.MIN_AVERAGE_CHANGE):
            return True
        return (set([tuple(a) for a in centroids]) == set([tuple(a) for a in oldCentroids]))



    def find_centers(self, vectors, k):
        oldCentroids = random.sample(list(vectors), k)
        if (self.initial is None):
            centroids = random.sample(list(vectors), k)
        else:
            centroids = self.initial;

        clusters = {}
        cnt = 0
        while not self.has_converged(centroids, oldCentroids) and cnt < self.MAX_ITERATIONS:
            cnt += 1
            print(cnt)
            oldCentroids = centroids
            clusters = self.cluster_points(vectors, centroids)
            centroids = self.reevaluate_centers(clusters)

        return (centroids, clusters)


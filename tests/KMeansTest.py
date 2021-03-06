import unittest
import pickle
import scipy.cluster.vq as vq
import random
import copy
import numpy
from scipy.spatial import distance

from src.KMeans import KMeans


class KMeansTest(unittest.TestCase):
    def test_descriptors(self):
        with open('allDescs.pkl', 'rb') as f:
            descriptors = pickle.load(f)

        newdescs = copy.copy(descriptors)
        initialCentroids = numpy.asarray(random.sample(list(newdescs), 500))

        centroids1, distorion = vq.kmeans2(descriptors, initialCentroids, minit='matrix')
        print(centroids1)

        kMenas = KMeans(descriptors, 500, _initial=initialCentroids)
        centroids2, clusters = kMenas.cluster()

        distances = list()
        for i in range(0, len(centroids1)):
            scipi_centroids = centroids1[i]
            my_centroids = centroids2[i]
            dist = distance.euclidean(scipi_centroids, my_centroids)
            distances.append(dist)

        print(distances)
        avgDistance = sum(distances) / float(len(distances))
        print(avgDistance)

        # not a good criteria but this should be same as kmeansn form scipy
        self.assertTrue(avgDistance < 0.1)

import random as rand
import math as math
import numpy as np

class KMeans:


    def __init__(self, _vectors, _k):
        self.vectors = _vectors
        self.k=_k
        self.MAX_ITERATIONS = 10;

    def cluster(self):
        #check dimenison
        #also needed for centroids creation
        dim = 0;



        print('do stuff')





    def compute_euclidean_distance(point, centroid):
        return np.sqrt(np.sum((point - centroid) ** 2))

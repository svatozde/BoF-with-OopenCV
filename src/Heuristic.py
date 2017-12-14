from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
import math


class AHeuristic:
    def getValue(self, word, img_descriptors):
        pass


class MinHeuristic(AHeuristic):
    def getValue(self, word, img_descriptors):
        """
        return cosine distance of two vectors in case vectors are not same size this throws exception
        :param in1:
        :param in2:
        :return:
        """
        min_distance = math.inf
        for desc in img_descriptors:
            curr_distance = distance.euclidean(word, desc)

            min_distance = min(min_distance, curr_distance)

        return min_distance


class CosineWordCountHeuristic(AHeuristic):
    def __init__(self, _value_threshold):
        self.value_threshold = _value_threshold

    def getValue(self, word, img_descriptors):
        """
        return cosine distance of two vectors in case vectors are not same size this throws exception
        :param in1:
        :param in2:
        :return:
        """
        count = 0
        for desc in img_descriptors:
            curr_similarity = 1-distance.cosine(word, desc)
            if curr_similarity >= self.value_threshold:
                count += 1

        return count

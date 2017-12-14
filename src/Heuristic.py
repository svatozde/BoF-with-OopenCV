from scipy.spatial import distance
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


class CosineHeuristic(AHeuristic):
    def getValue(self, word, img_descriptors):
        """
        return cosine distance of two vectors in case vectors are not same size this throws exception
        :param in1:
        :param in2:
        :return:
        """
        min_distance = math.inf
        for desc in img_descriptors:
            curr_distance = distance.cosine(word, desc)

            min_distance = min(min_distance, curr_distance)

        return min_distance

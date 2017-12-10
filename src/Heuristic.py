from scipy.spatial import distance
import math

class AHeuristic:
    def getValue(self, word, img_descriptors, threshold):
        pass


class MinHeuristic(AHeuristic):
    def getValue(self, word, img_descriptors, threshold):
        """
        return cosine distance of two vectors in case vectors are not same size this throws exception
        :param in1:
        :param in2:
        :return:
        """
        min_value = min(list(map(lambda desc: distance.euclidean(word, desc), img_descriptors)))

        return min_value

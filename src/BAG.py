import cv2
import scipy
from scipy.spatial import distance
import math


class BAG:
    def __init__(self, _words, _distance, _heuristic, _value_threshold,_normalize):
        self.words = _words
        # distance alg
        self.distance = _distance
        # Map image name/id and array of best matches with nth word
        self.bag = {}
        self.heuristic = _heuristic
        self.threshold = _value_threshold
        self.norm = _normalize

    def addImage(self, _id, _descriptors):
        """"
           append list of tuples (word_index, word_distance/similarity depends on heuristic)
           into bag dictionary
        """
        img_values = self._getSimilarityVector(_descriptors)
        filtered_values = []
        for i in range(len(img_values)):
            val = img_values[i]
            if val >= self.threshold:
                filtered_values.append((i, val));
        self.bag[_id] = filtered_values

    def _getSimilarityVector(self, _descriptors):
        """"
        returns normalized vector of values computed by provided heuristic
        """
        img_values = []
        for xword in self.words:
            value = self.heuristic.getValue(xword, _descriptors)
            img_values.append(value)
        if self.norm:
            return self.normalize(img_values)
        else:
            return img_values

    def magnitude(self, v):
        return math.sqrt(sum(v[i] * v[i] for i in range(len(v))))

    def normalize(self, v):
        vmag = self.magnitude(v)
        return [v[i] / vmag for i in range(len(v))]

    def getSimilar(self, _descriptors, _threshold):
        img_values = self._getSimilarityVector(_descriptors)
        ret = []
        for key, value in self.bag.items():
            #reduce vectors only to non zero values
            v1 = []
            v2 = []
            for val in value:
                v1.append(val[1])
                v2.append(img_values[val[0]])

            similarity = self.distance.distance(v1, v2)
            if (similarity >= _threshold):
                ret.append((key, similarity))
        ret.sort(key=lambda tup: tup[1], reverse=True)
        return ret

    def getSimilarImg(self, _img, _threshold):
        key_points, descriptors = self.sift.detectAndCompute(_img, None);
        return self.getSimilar(descriptors, _threshold)

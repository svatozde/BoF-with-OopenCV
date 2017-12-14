import cv2
import scipy
from scipy.spatial import distance
import numpy





class BAG:
    def __init__(self, _words, _classes, _distance, _heuristic, _value_threshold):
        self.words = _words
        self.classes = _classes
        # distance alg
        self.distance = _distance
        # Map image name/id and array of best matches with nth word
        self.bag = {}
        self.heuristic = _heuristic
        self.threshold=_value_threshold
        self.norm = True



    def addImage(self, _id, _descriptors):
        img_values = self._getSimilarityVector(_descriptors)
        self.bag[_id] = img_values



    def _getSimilarityVector(self, _descriptors):
        img_values = []
        for xword in self.words:
            value = self.heuristic.getValue(xword,_descriptors )

                img_values.append(value)
            else:
                img_values.append(0)
            

        return img_values



    def getSimilar(self, _descriptors, _threshold):
        img_values = self._getSimilarityVector(_descriptors)
        ret = []

        for key, value in self.bag.items():
            similarity = self.distance.distance(img_values, value)
            if (similarity >= _threshold):
                ret.append((key, similarity))
        ret.sort(key=lambda tup: tup[1], reverse=True)
        return ret



    def getSimilarImg(self, _img, _threshold):
        key_points, descriptors = self.sift.detectAndCompute(_img, None);
        return self.getSimilar(descriptors, _threshold)


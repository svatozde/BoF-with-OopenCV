import cv2
import scipy
from scipy.spatial import distance





class BAG:
    def __init__(self, _words, _classes, _distance):
        self.words = _words
        self.classes = _classes
        # distance alg
        self.distance = _distance
        # Map image name/id and array of best matches with nth word
        self.bag = {}
        self.sift = cv2.xfeatures2d.SURF_create()



    def addImage(self, _id, _descriptors):
        img_values = self._getSimilarityVector(_descriptors)
        self.bag[_id] = img_values



    def _getSimilarityVector(self, _descriptors):
        img_values = []
        for xword in self.words:
            min_distance = float("inf");

            for desc in _descriptors:
                curr_distance = distance.euclidean(xword, desc)
                min_distance = min(min_distance, curr_distance)

            img_values.append(min_distance)
        return img_values



    def getSimilar(self, _descriptors, _threshold):
        img_values = self._getSimilarityVector(_descriptors)
        ret = []

        for key, value in self.bag.items():
            similarity = 1 - scipy.spatial.distance.cosine(img_values, value)
            if (similarity >= _threshold):
                ret.append((key, similarity))
        ret.sort(key=lambda tup: tup[1], reverse=True)
        return ret



    def getSimilarImg(self, _img, _threshold):
        key_points, descriptors = self.sift.detectAndCompute(_img, None);
        return self.getSimilar(descriptors, _threshold)


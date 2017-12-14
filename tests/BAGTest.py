import glob
import pickle
import random
import unittest
import matplotlib.pyplot as plt

import cv2
import numpy as np
import scipy.cluster.vq as vq
from src.Distance import NPCosineDistance
from src.Heuristic import CosineWordCountHeuristic

from src.BAG import BAG


class BAGTest(unittest.TestCase):
    def get_descriptors(self, path, pickle_path):
        if glob.glob(pickle_path):
            with open(pickle_path, 'rb') as f:
                img_map = pickle.load(f)
            return img_map
        else:
            img_map = self.getDescriptors(path)
            with open(pickle_path, 'wb') as f:
                pickle.dump(img_map, f)
            return img_map

    def getDescriptors(self, path):
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=150)
        imap = {}
        count = 0
        files = glob.glob(path + "*")
        size = len(files)
        cnt = 0;
        for imagefile in files:
            img = cv2.imread(imagefile, 0)
            key_points, descriptors = sift.detectAndCompute(img, None);
            imap[imagefile] = descriptors
            cnt += 1
            print(str(size) + "|" + str(cnt))
        return imap

    def cluster(self, img_map, num_of_centroids):
        allDescriptors = None
        for descs in img_map.values():
            if (allDescriptors is None):
                allDescriptors = descs;
            else:
                allDescriptors = np.concatenate((allDescriptors, descs))

        initialCentroids = np.asarray(random.sample(list(allDescriptors), num_of_centroids))

        # TODO check what distortion does, try to eliminate not significant words
        centroids1, distorion = vq.kmeans2(allDescriptors, initialCentroids, minit='matrix')
        return centroids1

    def create_bag(self, words, img_map, pickle_path, heurristic, threshold,normalize):
        if pickle_path is not None and glob.glob(pickle_path):
            with open(pickle_path, 'rb') as f:
                unpickler = pickle.Unpickler(f)
                bag = unpickler.load()
                return bag

        #distance alg for comparing input and indexed values
        distance = NPCosineDistance()

        bag = BAG(words, distance, heurristic, threshold,normalize)

        for name, descs in img_map.items():
            print('ading: ' + name)
            bag.addImage(name, descs)

        if pickle_path is not None:
            with open(pickle_path, 'wb') as f:
                pickle.dump(bag, f)
        return bag

    def compareImages(self, img_name1, img_name2):

        img1 = cv2.imread(img_name1, 0)  # queryImage
        img2 = cv2.imread(img_name2, 0)  # trainImage

        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=150)

        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for i in range(len(matches))]

        # ratio test as per Lowe's paper
        number_of_matches = 0
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.6 * n.distance:
                matchesMask[i] = [1, 0]
                number_of_matches += 1

        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=0)

        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

        plt.imshow(img3, ), plt.show()
        return number_of_matches;

    def test_bag_with_lots_of_images(self):

        img_map = self.get_descriptors('../static/uploads/','small.pkl')

        words = self.cluster(img_map,600)

        bag = self.create_bag(words, img_map, 'bag2.pkl', CosineWordCountHeuristic(0.95), 0.1,False)


        best_10_matches = 0
        for name, descs in img_map.items():
            ret = bag.getSimilar(descs, 0.7)
            for i in range(0, min(len(ret),10)):
                best_10_matches = self.compareImages(name,ret[i][0])

        print('end')

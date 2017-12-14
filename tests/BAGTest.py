import glob
import pickle
import random
import unittest
import matplotlib.pyplot as plt

import cv2
import numpy as np
import scipy.cluster.vq as vq
from src.Distance import EuclideanDistance
from src.Heuristic import MinHeuristic

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
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=250)
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

    def create_bag(self, words, img_map, pickle_path, heurristic):
        if pickle_path is not None and glob.glob(pickle_path):
            with open(pickle_path, 'wb') as f:
                bag = pickle.load(f)
            return bag



        distance = EuclideanDistance()

        bag = BAG(words, None, distance, heurristic, 0.9)

        for name, descs in img_map.items():
            print('ading: ' + name)
            bag.addImage(name, descs)

        if pickle_path is not None:
            with open(pickle_path, 'wb') as f:
                pickle.dump(bag, f)
        return bag

    def compareImages(self, img_name1, img_name2):
        path_prefix = 'c:\\skola\\VMM\\TestOpenCV\\static\\uploads\\'

        img1 = cv2.imread(path_prefix + img_name1, 0)  # queryImage
        img2 = cv2.imread(img_name1 + img_name2, 0)  # trainImage
        # Initiate ORB detector
        orb = cv2.ORB_create()
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], flags=2)
        plt.imshow(img3), plt.show()

        return 0;

    def test_bag_with_lots_of_images(self):

        img_map = self.get_descriptors('../static/uploads/','../small.pkl')

        words = self.cluster(img_map,250)

        bag = self.create_bag(words, img_map, 'bag2.pkl', MinHeuristic())


        for name, descs in img_map.items():
            ret = bag.getSimilar(descs, 0.97)
            for i in range(0,10):
                self.compareImages(descs,ret[i][0])

        print('end')

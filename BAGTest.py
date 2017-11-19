import unittest
import pickle
import scipy.cluster.vq as vq
import random
import copy
import numpy as np
from BAG import BAG
from Distance import EuclideanDistance
import glob
import cv2


from scipy.spatial import distance
from KMeans import KMeans


class BAGTest(unittest.TestCase):


    def test_bag_creation(self):
        with open('descsPerImg.pkl', 'rb') as f:
            descriptors = pickle.load(f)

        allDescriptors = None
        for descs in descriptors.values():
            if (allDescriptors is None):
                allDescriptors = descs;
            else:
                allDescriptors = np.concatenate((allDescriptors, descs))

        initialCentroids = np.asarray(random.sample(list(allDescriptors), 50))

        centroids1, distorion = vq.kmeans2(allDescriptors, initialCentroids, minit='matrix')

        distance = EuclideanDistance();

        bag = BAG(centroids1,None,distance)

        for name, descs in descriptors.items():
            bag.addImage(name,descs)

        for name, descs in descriptors.items():
            ret = bag.getSimilar(descs, 0.2)
            print(str(name) + '|' +str(ret))

    def test_bag_with_lots_of_images(self):
        #imgMap = self.getFiles("c:\\skola\\VMM\\jpg2\\")
        #with open('jpg2.pkl', 'wb') as f:
        #   pickle.dump(imgMap, f)

        with open('jpg2.pkl', 'rb') as f:
           imgMap = pickle.load(f)

        allDescriptors = None
        for descs in imgMap.values():
            if (allDescriptors is None):
                allDescriptors = descs;
            else:
                allDescriptors = np.concatenate((allDescriptors, descs))

        initialCentroids = np.asarray(random.sample(list(allDescriptors), 1500))

        centroids1, distorion = vq.kmeans2(allDescriptors, initialCentroids, minit='matrix')

        distance = EuclideanDistance();

        bag = BAG(centroids1, None, distance)

        for name, descs in imgMap.items():
            bag.addImage(name, descs)

        for name, descs in imgMap.items():
            ret = bag.getSimilar(descs, 0.95)
            print(str(name) + '|' + str(ret))

        print('end')

    def getFiles(self,path):
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
            print(str(size)+"|"+str(cnt))
        return imap
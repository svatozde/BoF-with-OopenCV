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
import time
import os
import ntpath
from os.path import basename

import re
import scipy

from src.BAG import BAG


class BAGTest(unittest.TestCase):
    def get_descriptors(self, path, pickle_path, nfeatures, blur_kernel):
        if glob.glob(pickle_path):
            with open(pickle_path, 'rb') as f:
                img_map = pickle.load(f)
            return img_map
        else:
            img_map = self.getDescriptors(path, nfeatures, blur_kernel)
            with open(pickle_path, 'wb') as f:
                pickle.dump(img_map, f)
            return img_map

    def getDescriptors(self, path, nfeatures, blur_kernel):
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=nfeatures)
        imap = {}
        count = 0
        files = glob.glob(path + "*")
        size = len(files)
        cnt = 0;
        for imagefile in files:
            img = cv2.imread(imagefile, 0)
            if blur_kernel != 0:
                img = cv2.medianBlur(img, blur_kernel, 0)
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

    def create_bag(self, words, img_map, pickle_path, heurristic, threshold, normalize):
        if pickle_path is not None and glob.glob(pickle_path):
            with open(pickle_path, 'rb') as f:
                unpickler = pickle.Unpickler(f)
                bag = unpickler.load()
                return bag, 0, len(bag.bag)

        # distance alg for comparing input and indexed values
        distance = NPCosineDistance()

        bag = BAG(words, distance, heurristic, threshold, normalize)

        millis = int(round(time.time() * 1000))
        cnt = 0
        for name, descs in img_map.items():
            cnt += 1
            print('adding: ' + str(len(img_map)) + '|' + str(cnt))
            bag.addImage(name, descs)
        build_time = int(round(time.time() * 1000)) - millis

        if pickle_path is not None:
            with open(pickle_path, 'wb') as f:
                pickle.dump(bag, f)
        return bag, build_time, len(img_map)

    def compareImages(self, img_name1, img_name2, bag_distance, results_folder, blur_kernel):
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=2000)

        img1 = cv2.imread(img_name1, 0)  # queryImage
        img2 = cv2.imread(img_name2, 0)  # trainImage

        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        matchesMask = [[0, 0] for i in range(len(matches))]

        number_of_matches = 0
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.9 * n.distance:
                matchesMask[i] = [1, 0]
                number_of_matches += 1

        draw_params = dict(matchesMask=matchesMask,
                           flags=0)

        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

        tail1 = basename(img_name1)[:-4]
        tail2 = basename(img_name2)[:-4]

        print('comapared: ' + tail1 + ' | ' + tail2)

        cat1 = tail1[:tail1.find('_')]
        cat2 = tail2[:tail2.find('_')]

        if cat1 == cat2:
            category_match = True
        else:
            category_match = False
        imgName = 'img_' + tail1 + '_' + tail2 + '_' + str(round(bag_distance, 2)) + '_' + str(
            number_of_matches) + '.jpg'
        path = os.path.join(results_folder, imgName)
        cv2.imwrite(path, img3)

        return number_of_matches, category_match;

    def do_run(self, num_cluster, num_features, heuristic_threshold, blur_kernel):
        test_name = 'f' + str(num_features) + '_c' + str(num_cluster) + '_t' + str(
            heuristic_threshold * 100) + '_b' + str(blur_kernel)
        print('launching:' + test_name)
        img_map = self.get_descriptors('../static/uploads/',
                                       'results/test_desc_' + str(num_features) + '_k' + str(blur_kernel) + '.pkl',
                                       num_features, blur_kernel)
        words = self.cluster(img_map, num_cluster)

        bag, build_time, img_count = self.create_bag(words, img_map, 'results/' + test_name + '_bag.pkl',
                                                     CosineWordCountHeuristic(heuristic_threshold), 0.0, False)

        directory = 'results/imgs_' + test_name

        if not os.path.exists(directory):
            os.makedirs(directory)

        query_desc_name = 'results/query_desc_' + str(num_features) + '_k' + str(blur_kernel) + '.pkl'
        test_descriptors = self.get_descriptors('query/', query_desc_name, num_features, blur_kernel)

        matches = 0
        avg_search_time = 0
        count = 0
        similarity_sum = 0
        search_count = 0
        ret_size = 0;

        compare_count = 50
        all_compares = len(test_descriptors) * compare_count
        cat_matches = 0
        for name in test_descriptors.keys():
            search_start = int(round(time.time() * 1000))
            ret = bag.getSimilar(test_descriptors.get(name), 0.85)
            ret_size += len(ret)
            avg_search_time += int(round(time.time() * 1000)) - search_start
            search_count += 1
            for i in range(0, min(len(ret), compare_count)):
                count += 1
                similarity_sum += ret[i][1]
                print('comparing: ' + str(all_compares) + '|' + str(count))
                match, cat_match = self.compareImages(name, ret[i][0], ret[i][1], directory, blur_kernel)
                matches += match
                if cat_match == True:
                    cat_matches += 1

        avg_search_time = avg_search_time / search_count
        avg_ret_size = ret_size / search_count

        avg_matches = 0
        avg_similarity = 0
        if count != 0:
            avg_matches = matches / count
            avg_similarity = similarity_sum / count

        self.write_results_to_file(test_name,
                                   [num_cluster, num_features, heuristic_threshold, blur_kernel, img_count, build_time,
                                    round(avg_search_time, 2), round(avg_similarity, 2), round(avg_matches, 2),
                                    round(avg_ret_size, 2), cat_matches, count])
        return matches, build_time, img_count, avg_search_time, test_name

    def write_results_to_file(self, test_name, vals):
        for v in vals:
            test_name = test_name + ',' + str(v)
        with open('bag_test_results.csv', "a") as myfile:
            myfile.write(test_name + '\n')

    def test_100_15_95_5(self):
        self.do_run(1500, 300, 0.95, 0)



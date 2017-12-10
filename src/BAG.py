import cv2
import scipy
import numpy as np
from scipy.spatial import distance

import scipy.cluster.vq as vq
import random
import os
import pickle
import glob
import collections



from src.Heuristic import MinHeuristic
from src.Distance import EuclideanDistance


class BAG:
    def __init__(self, _words, _classes, _distance,_heuristic,_threshold):
        self.words = _words
        self.classes = _classes
        # distance alg
        self.distance = _distance
        # Map image name/id and array of best matches with nth word
        self.bag = []
        self.heuristic = _heuristic
        self.threshold =_threshold
        self.norm = True

    def add_image(self, img_name, img_descriptors):
        similarity_vector = self.get_similarity_vector(img_descriptors)
        print("Added " + img_name)
        self.bag.append({
            "img_name": img_name,
            "similarity_vector": similarity_vector
        })

    def get_similarity_vector(self, descriptors):
        similarity_vector = list(map(lambda word: self.heuristic.getValue(word, descriptors, self.threshold), self.words))
        return similarity_vector

    def find_similar_images(self, descriptors, threshold):
        similarity_vector = self.get_similarity_vector(descriptors)
        ret = []

        similarity = np.array(list(map(lambda bag_item:
                            1 - scipy.spatial.distance.cosine(similarity_vector, bag_item["similarity_vector"]),
                            self.bag)))

        # Normalization  0 - 1
        norm_similarity = (similarity - min(similarity)) / (max(similarity) - min(similarity))
        print(norm_similarity)

        result = list(filter(lambda x: x[1] >= threshold, list(zip(self.bag, norm_similarity))))
        result.sort(key=lambda tup: tup[1], reverse=True)

        return result

    def get_similar_img(self, img, threshold):
        sift = cv2.xfeatures2d.SURF_create()
        key_points, descriptors = sift.detectAndCompute(img, None)
        return self.find_similar_images(descriptors, threshold)


def get_descriptors_images(path):
    Image = collections.namedtuple('Image', 'name descriptors')

    sift = cv2.xfeatures2d.SIFT_create(nfeatures=150)
    coputed_images = []
    files = glob.glob(path + "*")
    concatenate_descriptors = None

    for file_name in files:
        img = cv2.imread(file_name, 0)
        _, descriptors = sift.detectAndCompute(img, None)
        file_name = file_name.split('/')[-1]
        coputed_images.append(Image(name=file_name, descriptors=descriptors))

        # First descriptors
        if concatenate_descriptors is None:
            concatenate_descriptors = descriptors
        else:
            concatenate_descriptors = np.concatenate((concatenate_descriptors, descriptors))

        print(".")

    print("Processed " + str(len(coputed_images)) + " images.")
    return coputed_images, concatenate_descriptors


def create_bag():

    if os.path.exists("./data/BAG.pkl"):
        with open('./data/BAG.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        coputed_images, concatenate_descriptors = get_descriptors_images("./data/samples/tmp/")

        init_centroids = np.asarray(random.sample(list(concatenate_descriptors), 1500))
        centroids1, _ = vq.kmeans2(concatenate_descriptors, init_centroids, minit='matrix')
        euc_distance = EuclideanDistance()

        bag = BAG(centroids1, None, euc_distance, MinHeuristic(), 0.9)
        for img in coputed_images:
            bag.add_image(img.name, img.descriptors)

        print("save")
        with open('./data/BAG.pkl', 'wb') as f:
            pickle.dump(bag, f)






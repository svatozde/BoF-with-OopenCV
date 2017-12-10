import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
from sklearn.cluster import KMeans
import os
import glob
import pickle


def imlist(path):
    """
    The function imlist returns all the names of the files in
    the directory path supplied as argument to the function.
    """
    return [os.path.join(path, f) for f in os.listdir(path)]


def getFiles(path):
    imlist = {}
    count = 0
    for each in glob.glob(path + "*"):
        word = each.split("/")[-1]
        print(" #### Reading image category " + word + " ##### ")
        imlist[word] = []
        for imagefile in glob.glob(word + "/*"):
            print("Reading file " + imagefile)
            im = cv2.imread(imagefile, 0)
            imlist[word].append(im)
            count += 1

    return [imlist, count]


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def cosineDistance(in1, in2):
    distance = 0


def main():
    print(cv2.__version__)

    sift = cv2.xfeatures2d.SURF_create()

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    trainer = cv2.BOWKMeansTrainer(500)

    print(dir(trainer))
    extract_bow = cv2.BOWImgDescriptorExtractor(sift, flann)

    imgs, cnt = getFiles('cliparts/')
    allDescriptors = None
    for word, imlist in imgs.items():
        # print("category" + word)
        for img in imlist:
            key_points, descriptors = sift.detectAndCompute(img, None);
            if (allDescriptors is None):
                allDescriptors = descriptors;
            else:
                allDescriptors = np.concatenate((allDescriptors, descriptors))
            trainer.add(descriptors)

    # pickle all descriptors for tests of k-means
    with open('allDescs.pkl', 'wb') as f:
        pickle.dump(allDescriptors, f)

    vocabulary = trainer.cluster();
    extract_bow.setVocabulary(vocabulary)

    # print(dir(extract_bow))
    # print(np.shape(vocabulary))

    traindata, trainlabels = [], []
    cnt = 0
    labels_map = {}
    for word, imlist in imgs.items():
        # print("category " + word)

        labels_map[cnt] = word
        for img in imlist:
            traindata.extend(extract_bow.compute(img, sift.detect(img)))
            trainlabels.append(cnt)
        cnt += 1

    print(extract_bow.descriptorSize())
    print(labels_map)

    # print(dir(cv2.ml))

    svm = cv2.ml.KNearest_create()
    svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))
    svm.save('svm1.xml')

    for word, imlist in imgs.items():
        # print("category" + word)
        for img in imlist:
            extracted = extract_bow.compute(img, sift.detect(img))
            out = svm.predict(extracted)
            print(word + " " + labels_map[out[1][0][0]])


main()

import ADistance
import math

class CosineDistance(ADistance):
    def __init__(self):
        print('construct')


    def distance(self, in1, in2):
        """
        return cosine distance of two vectors in case vectors are not same size this throws exception
        :param in1:
        :param in2:
        :return:
        """
        length = max(len(in1), len(in2))
        sumOfProducts =  0
        sumOfPowsA=0;
        sumOfPowsB=0;
        for i in range(length):
            sumOfProducts+=in1[i]*in2[2]
            sumOfPowsA += pow(in1[i],2)
            sumOfPowsB += pow(in1[i], 2)
        return (sumOfProducts/(math.sqrt(sumOfPowsA)*math.sqrt(sumOfPowsB)))
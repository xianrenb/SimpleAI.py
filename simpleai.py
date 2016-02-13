#!/usr/bin/env python

'''
    Simple AI
'''

import numpy as np

class SimpleAI(object):
    def __init__(self, class_n = 10, k = 5):
        """Constructor for SimpleAI

        Keyword arguments:
        class_n -- number of classes
        k -- parameter used in prediction
        """
        self.class_n = class_n
        self.k = k
        self.groupsize = class_n ** 2
        self.modelsize = class_n ** 2

    def getrandomgroup(self):
        samplegroup = []
        responsegroup = []
        for i in xrange(self.groupsize):
            x = np.random.random_integers(self.samplesize) - 1
            samplegroup.append(self.samples[x])
            responsegroup.append(self.responses[x])
        return samplegroup , responsegroup

    def getscoregroup(self, samplegroup, responsegroup, expectedresponse):
        scoregroup = []
        for i in xrange(self.groupsize):
            if responsegroup[i] == expectedresponse:
                score = np.float32(1.0)
            else:
                score = np.float32(0.0)
            scoregroup.append(score)
        return scoregroup

    def predict(self, samples):
        """Predict the response with the trained SimpleAI model

        Arguments:
        samples -- List of samples, each sample should be an array of numbers

        Output value:
        List of predicted (integer) responses matching the samples
        """
        result = []
        for i in xrange(len(samples)):
            result.append(self.predict_sample(samples[i]))
        return result

    def predict_sample(self, sample):
        k = self.k
        scores = []
        for i in xrange(self.class_n):
            class_scores = []
            for j in xrange(self.groupsize):
                x = np.random.random_integers(self.modelsize) - 1
                pseudomax, biasdistance = self.model[i][x]
                diff = sample - pseudomax
                distance = sum(diff * diff)**np.float32(0.5)
                score = biasdistance - distance
                class_scores.append(score)
            class_scores = np.array(class_scores)
            # find the largest k elements' indices
            ind = np.argpartition(class_scores, -k)[-k:]
            scores.append(class_scores[ind].mean())
        scores = np.array(scores)
        # find the largest element's index (in array)
        ind = np.argpartition(scores, -1)[-1:]
        # return the index, which is equal to the predicted response
        return ind[0]

    def train(self, samples, responses):
        """Train the SimpleAI model

        Arguments:
        samples -- List of samples, each sample should be an array of numbers
        responses -- List of responses, each response should be an integer representing the class of the sample
        """
        eps = np.float32(1e-7)
        self.samples = samples
        self.responses = responses
        self.samplesize = len(samples)
        self.model = []
        for i in xrange(self.class_n):
            self.model.append([])
            for j in xrange(self.modelsize):
                samplegroup , responsegroup = self.getrandomgroup()
                scoregroup = self.getscoregroup(samplegroup, responsegroup, i)
                s1 = np.zeros_like(samplegroup[0])
                s2 = np.float32(0.0)
                for k in xrange(self.groupsize):
                    s1 += samplegroup[k] * scoregroup[k]
                    s2 += scoregroup[k]
                # pseudomax should be the point with the highest score
                pseudomax = s1 / (s2 + eps)
                s1 = np.float32(0.0)
                s2 = np.float32(0.0)
                s3 = np.float32(0.0)
                s4 = np.float32(0.0)
                for k in xrange(self.groupsize):
                    diff = samplegroup[k] - pseudomax
                    d = sum(diff * diff)**np.float32(0.5)
                    s1 += d * scoregroup[k]
                    s2 += scoregroup[k]
                    s3 += d * (np.float32(1.0) - scoregroup[k])
                    s4 += np.float32(1.0) - scoregroup[k]
                matchdistance = s1 / (s2 + eps)
                unmatchdistance = s3 / (s4 + eps)
                biasdistance = (matchdistance + unmatchdistance) / np.float32(2.0)
                data = (pseudomax, biasdistance)
                self.model[i].append(data)
        # clear data to save memory
        self.samples = None
        self.responses = None

#!/usr/bin/env python

import random
import numpy as np
from utils.treebank import RTE_dataset
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import operator
from word2vec import *
from sgd import *

# Check Python Version
import sys
assert sys.version_info[0] == 3
assert sys.version_info[1] >= 5

# Reset the random seed to make sure that everyone gets the same results
random.seed(314)
dataset = RTE_dataset()
tokens = dataset.tokens()
nWords = len(tokens)
# print(type(dataset._tokenfreq))
# sorted_d = sorted(dataset._tokenfreq.items(), key=operator.itemgetter(1), reverse=True)[:1000]
# print('Dictionary in descending order by value : ',sorted_d)
# We are going to train 10-dimensional vectors for this assignment
dimVectors = 10

# Context size
C = 5

# Reset the random seed to make sure that everyone gets the same results
random.seed(31415)
np.random.seed(9265)

startTime=time.time()
wordVectors = np.concatenate(
    ((np.random.rand(nWords, dimVectors) - 0.5) /
       dimVectors, np.zeros((nWords, dimVectors))),
    axis=0)
wordVectors = sgd(
    lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C,
        negSamplingLossAndGradient),
    wordVectors, 0.3, 40010, None, True, PRINT_EVERY=10)
# Note that normalization is not called here. This is not a bug,
# normalizing during training loses the notion of length.

print("sanity check: cost at convergence should be around or below 20")
print("training took %d seconds" % (time.time() - startTime))

# concatenate the input and output word vectors
wordVectors = np.concatenate(
    (wordVectors[:nWords,:], wordVectors[nWords:,:]),
    axis=0)

# visualizeWords = [
#     "great", "cool",  "wonderful", "well", "amazing",
#     "worth", "sweet",   "bad", "eat", "eating", "meet", "meeting"
#      "female", "male", "queen", "king", "man", "woman", "rain", "snow",
#      "coffee", "tea"]
visualizeWords = [
    "thursday", "work", 'security', "tuesday", "friday", "monday", "sunday","iraq","iraqi", "russia","russian", "authorities",
    "war", "european", "northern", "southern", "january", "august", "chief", 
    "palestinian", "commission", "white", "black", "party", "south", "north", "television", "computer", "service", "spokesman"
    ]
visualizeIdx = [tokens[word] for word in visualizeWords]
visualizeVecs = wordVectors[visualizeIdx, :]
temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
U,S,V = np.linalg.svd(covariance)
coord = temp.dot(U[:,0:2])

for i in range(len(visualizeWords)):
    plt.text(coord[i,0], coord[i,1], visualizeWords[i],
        bbox=dict(facecolor='green', alpha=0.1))

plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))

plt.savefig('word_vectors.png')

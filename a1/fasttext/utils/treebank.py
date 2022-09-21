#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import os
import random
import re
from collections import defaultdict


def generate_ngrams(s, min, max):
    """ 

    generate character ngrams in this function.

    Arguments:
    s -- given word which will be decomposed into character n-grams
    min -- minimum number of ''n'' at character ''n''-grams
    max -- maximum number of ''n'' at character ''n''-grams

    Return:
    character_n_grams -- sorted list of character n-grams of word s
    
    Example: Given s petri, min 2, and max 4. 
    The correct return of this function is  ['<p', '<pe', '<pet', 'et', 'etr', 'etri', 'i>', 'pe', 'pet', 'petr','petri', 'ri', 'ri>', 'tr', 'tri', 'tri>']
    """
    s = s.lower()

    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)

    all_n_grams = set()
    start_token="<"
    end_token=">"
    all_n_grams.add(s)
    ### YOUR CODE HERE (~ 5 lines) ###

    ######

    character_n_grams=sorted(list(all_n_grams))
    return character_n_grams

class RTE_dataset:
    def __init__(self, path=None, tablesize = 1000000):
        if not path:
            path = "utils/datasets/RTE"

        self.path = path
        self.tablesize = tablesize

    def tokens(self):
        if hasattr(self, "_tokens") and self._tokens:
            return self._tokens

        tokens = dict()
        tokenfreq = dict()
        wordcount = 0
        ng_count = 0
        revtokens = []
        idx = 0
        self.ngram_hash = {}
        self.ngram_freq = defaultdict(int)
        self.ngram_idx = 0
        self.inv_ngram_hash = []
        self.word_to_ngrams_idx = defaultdict(list)

        for sentence in self.sentences():
            for w in sentence:
                wordcount += 1
                if not w in tokens:
                    tokens[w] = idx
                    revtokens += [w]
                    tokenfreq[w] = 1
                    idx += 1

                    ngrams = generate_ngrams(w, 2, 4)
                    for ng in ngrams:
                        ng_count += 1
                        if ng not in self.ngram_hash:
                            self.ngram_hash[ng] = self.ngram_idx
                            self.inv_ngram_hash += [ng]
                            self.ngram_freq[ng] += 1
                            self.ngram_idx += 1
                        else:
                            self.ngram_freq[ng] += 1
                            self.word_to_ngrams_idx[w].append(self.ngram_hash[ng])

                else:
                    tokenfreq[w] += 1



        tokens["UNK"] = idx
        revtokens += ["UNK"]
        tokenfreq["UNK"] = 1
        wordcount += 1

        self._tokens = tokens
        self._tokenfreq = tokenfreq
        self._wordcount = wordcount
        self._revtokens = revtokens
        return self._tokens

    def sentences(self):
        if hasattr(self, "_sentences") and self._sentences:
            return self._sentences

        sentences = []
        with open(self.path + "/test.tsv", "r") as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue

                splitted = line.strip().split()[1:]
                # Deal with some peculiar encoding issues with this file
                sentences += [[w.lower() for w in splitted]]

        self._sentences = sentences
        self._sentlengths = np.array([len(s) for s in sentences])
        self._cumsentlen = np.cumsum(self._sentlengths)

        return self._sentences

    def numSentences(self):
        if hasattr(self, "_numSentences") and self._numSentences:
            return self._numSentences
        else:
            self._numSentences = len(self.sentences())
            return self._numSentences

    def allSentences(self):
        if hasattr(self, "_allsentences") and self._allsentences:
            return self._allsentences

        sentences = self.sentences()
        rejectProb = self.rejectProb()
        tokens = self.tokens()
        allsentences = [[w for w in s
            if 0 >= rejectProb[tokens[w]] or random.random() >= rejectProb[tokens[w]]]
            for s in sentences * 30]

        allsentences = [s for s in allsentences if len(s) > 1]

        self._allsentences = allsentences

        return self._allsentences

    def getRandomContext(self, C=5):
        allsent = self.allSentences()
        sentID = random.randint(0, len(allsent) - 1)
        sent = allsent[sentID]
        wordID = random.randint(0, len(sent) - 1)

        context = sent[max(0, wordID - C):wordID]
        if wordID+1 < len(sent):
            context += sent[wordID+1:min(len(sent), wordID + C + 1)]

        centerword = sent[wordID]
        context = [w for w in context if w != centerword]

        if len(context) > 0 and len(self.word_to_ngrams_idx[centerword]) > 0:
            return centerword, self.word_to_ngrams_idx[centerword], context
        else:
            return self.getRandomContext(C)

    def sent_labels(self):
        if hasattr(self, "_sent_labels") and self._sent_labels:
            return self._sent_labels

        dictionary = dict()
        phrases = 0
        with open(self.path + "/dictionary.txt", "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                splitted = line.split("|")
                dictionary[splitted[0].lower()] = int(splitted[1])
                phrases += 1

        labels = [0.0] * phrases
        with open(self.path + "/sentiment_labels.txt", "r") as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue

                line = line.strip()
                if not line: continue
                splitted = line.split("|")
                labels[int(splitted[0])] = float(splitted[1])

        sent_labels = [0.0] * self.numSentences()
        sentences = self.sentences()
        for i in range(self.numSentences()):
            sentence = sentences[i]
            full_sent = " ".join(sentence).replace('-lrb-', '(').replace('-rrb-', ')')
            sent_labels[i] = labels[dictionary[full_sent]]

        self._sent_labels = sent_labels
        return self._sent_labels

    def dataset_split(self):
        if hasattr(self, "_split") and self._split:
            return self._split

        split = [[] for i in range(3)]
        with open(self.path + "/datasetSplit.txt", "r") as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue

                splitted = line.strip().split(",")
                split[int(splitted[1]) - 1] += [int(splitted[0]) - 1]

        self._split = split
        return self._split

    def getRandomTrainSentence(self):
        split = self.dataset_split()
        sentId = split[0][random.randint(0, len(split[0]) - 1)]
        return self.sentences()[sentId], self.categorify(self.sent_labels()[sentId])

    def categorify(self, label):
        if label <= 0.2:
            return 0
        elif label <= 0.4:
            return 1
        elif label <= 0.6:
            return 2
        elif label <= 0.8:
            return 3
        else:
            return 4

    def getDevSentences(self):
        return self.getSplitSentences(2)

    def getTestSentences(self):
        return self.getSplitSentences(1)

    def getTrainSentences(self):
        return self.getSplitSentences(0)

    def getSplitSentences(self, split=0):
        ds_split = self.dataset_split()
        return [(self.sentences()[i], self.categorify(self.sent_labels()[i])) for i in ds_split[split]]

    def sampleTable(self):
        if hasattr(self, '_sampleTable') and self._sampleTable is not None:
            return self._sampleTable

        nTokens = len(self.tokens())
        samplingFreq = np.zeros((nTokens,))
        self.allSentences()
        i = 0
        for w in range(nTokens):
            w = self._revtokens[i]
            if w in self._tokenfreq:
                freq = 1.0 * self._tokenfreq[w]
                # Reweigh
                freq = freq ** 0.75
            else:
                freq = 0.0
            samplingFreq[i] = freq
            i += 1

        samplingFreq /= np.sum(samplingFreq)
        samplingFreq = np.cumsum(samplingFreq) * self.tablesize

        self._sampleTable = [0] * self.tablesize

        j = 0
        for i in range(self.tablesize):
            while i > samplingFreq[j]:
                j += 1
            self._sampleTable[i] = j

        return self._sampleTable

    def rejectProb(self):
        if hasattr(self, '_rejectProb') and self._rejectProb is not None:
            return self._rejectProb

        threshold = 1e-5 * self._wordcount

        nTokens = len(self.tokens())
        rejectProb = np.zeros((nTokens,))
        for i in range(nTokens):
            w = self._revtokens[i]
            freq = 1.0 * self._tokenfreq[w]
            # Reweigh
            rejectProb[i] = max(0, 1 - np.sqrt(threshold / freq))

        self._rejectProb = rejectProb
        return self._rejectProb

    def sampleTokenIdx(self):
        return self.sampleTable()[random.randint(0, self.tablesize - 1)]

def test_treebank():
    """ Test treebank implementations, before running """
    ngram_lists=[['<w', '<wo', '<woo', 'an', 'and', 'and>', 'd>', 'dl', 'dla', 'dlan', 'la', 'lan', 'land', 'nd', 'nd>', 'od', 'odl', 'odla', 'oo', 'ood', 'oodl', 'wo', 'woo', 'wood', 'woodland'],
    ['<q', '<qu', '<que', 'as', 'asy', 'asy>', 'ea', 'eas', 'easy', 'qu', 'que', 'quea', 'queasy', 'sy', 'sy>', 'ue', 'uea', 'ueas', 'y>'],
    ['<i', '<in', '<inf', 'at', 'ati', 'atio', 'atu', 'atua', 'fa', 'fat', 'fatu', 'in', 'inf', 'infa', 'infatuation', 'io', 'ion', 'ion>', 'n>', 'nf', 'nfa', 'nfat', 'on', 'on>', 'ti', 'tio', 'tion', 'tu', 'tua', 'tuat', 'ua', 'uat', 'uati'],
    [' i', ' in', ' in ', ' p', ' pe', ' per', '<k', '<ki', '<kid', 'ds', 'ds ', 'ds i', 'er', 'eri', 'eril', 'id', 'ids', 'ids ', 'il', 'il>', 'in', 'in ', 'in p', 'ki', 'kid', 'kids', 'kids in peril', 'l>', 'n ', 'n p', 'n pe', 'pe', 'per', 'peri', 'ri', 'ril', 'ril>', 's ', 's i', 's in'],
    ['<p', '<pe', '<pet', 'et', 'etr', 'etri', 'i>', 'pe', 'pet', 'petr','petri', 'ri', 'ri>', 'tr', 'tri', 'tri>']]
    s_lists=["woodland", "queasy","infatuation","kids in peril","petri"]
    print("==== Answer check for generate_ngrams ====")
    print("\n=== Results ===")
    for i in range(len(s_lists)):
        if generate_ngrams(s_lists[i],2,4) != ngram_lists[i]: 
            print("Wrong Answer! character n-grams of {} should be {}".format(s_lists[i],ngram_lists[i]))
            print("Your answer was ", generate_ngrams(s_lists[i],2,4))
            return
    print("Sanity check passed")




if __name__ == "__main__":
    test_treebank()

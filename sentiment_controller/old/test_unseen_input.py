# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
# numpy
import numpy
import json
import time
# random
from random import shuffle
# classifier
from sklearn.linear_model import LogisticRegression
# our LabeledLineSentence class
from labeled_line_sentence import LabeledLineSentence

from sklearn.linear_model import SGDClassifier

import os

dirname = os.path.dirname(__file__)

# load our doc2vec model that we trained
model = Doc2Vec.load(os.path.join(dirname,'../models/yelp_model.d2v'))

# create an array of LabeledLineSentences for previously unseen
# good and bad reviews
# this does some basic formatting of the text as well to make it more
# digestible by gensim and sklearn

test_sources_good = {os.path.join(dirname, '../tests/good-bad/review_test_5_star.json'):'TEST_POS'}
test_sources_bad = {os.path.join(dirname, '../tests/good-bad/review_test_1_star.json'):'TEST_NEG'}
good = LabeledLineSentence(test_sources_good).to_array()
bad = LabeledLineSentence(test_sources_bad).to_array()

# take our train reviews from the model, and put them in array, good reviews first, bad reviews second half of array
train_arrays = numpy.zeros((12508, 50))
train_labels = numpy.zeros(12508)
for i in range(6254):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    train_arrays[i] = model.docvecs[prefix_train_pos]
    train_arrays[6254 + i] = model.docvecs[prefix_train_neg]
    train_labels[i] = 1
    train_labels[6254 + i] = 0

# take our test reviews from the model, and put them in array, good reviews first, bad reviews second half of array
# for each review, we'll infer the review's vector against our model
#print("Loading test reviews...")
test_arrays = numpy.zeros((12508, 50))
test_labels = numpy.zeros(12508)
for i in range(6254):
    test_arrays[i] = model.infer_vector(good[i][0])
    test_arrays[6254 + i] = model.infer_vector(bad[i][0])
    test_labels[i] = 1
    test_labels[6254 + i] = 0
#print("Done!")

# create a logistic regression classifier
classifier = LogisticRegression()
#classifier = SGDClassifier(loss='log', penalty='l1')
classifier.fit(train_arrays, train_labels)

# try an infer sentiment of a user inputted sentence
sentence = raw_input("Type a sentence: ")
while sentence != "":
    labeled_sentence = utils.to_unicode(sentence).split()
    inferred_vector = model.infer_vector(labeled_sentence)

    # predict its sentiment... 1 for good, 0 for bad
    prediction = classifier.predict([inferred_vector])
    if prediction == 0:
        print("We think this is a bad review.")
    else:
        print("We think this is a good review.")
    sentence = raw_input("Type a sentence: ")

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

print("Search driver starting...")

dirname = os.path.dirname(__file__)

# load our doc2vec model that we trained
model = Doc2Vec.load(os.path.join(dirname,'models/yelp_model_newest.d2v'))

# create an array of LabeledLineSentences for previously unseen
# good and bad reviews
# this does some basic formatting of the text as well to make it more
# digestible by gensim and sklearn

print("Loading data...")

# take our train reviews from the model, and put them in array, good reviews first, bad reviews second half of array
train_arrays = numpy.zeros((100000, 300))
train_labels = numpy.zeros(100000)

# take our train reviews from the model, and put them in array, good reviews first, bad reviews second half of array
for i in range(50000):
    prefix_train_pos = 'bad_' + str(i)
    prefix_train_neg = 'good_' + str(i)

    pos_review = model.docvecs[prefix_train_pos]
    neg_review = model.docvecs[prefix_train_neg]

    train_arrays[i] = pos_review
    train_labels[i] = 1

    train_arrays[50000 + i] = neg_review
    train_labels[50000 + i] = 0


# create a logistic regression classifier
classifier = LogisticRegression()
#classifier = SGDClassifier(loss='log', penalty='l1')
classifier.fit(train_arrays, train_labels)

print("Ready")

# try an infer sentiment of a user inputted sentence
sentence = raw_input()
while sentence != "":
    labeled_sentence = utils.to_unicode(sentence).split()
    inferred_vector = model.infer_vector(labeled_sentence)
    print(model.docvecs.most_similar([inferred_vector]))
    print ("STOP")
    sentence = raw_input()

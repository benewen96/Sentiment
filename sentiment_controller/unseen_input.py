# Used to get similar reviews for web application
# ARGS:
# 1: total train reviews
# 2: number of iterations (for csv output)
# 3: size of vector
# 4: good/bad sizes
#
# python unseen_input.py 75000 1 400 12500

# import dependencies
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from random import shuffle
from sklearn.linear_model import LogisticRegression
from labeled_line_sentence import LabeledLineSentence
from sklearn.linear_model import SGDClassifier
import numpy
import json
import time
import os
import sys

print("Search driver starting...")

dirname = os.path.dirname(__file__)

# so that we can spawn instances in node without passing args
sys.argv = ["", 60000, 1, 10, 12500]

print("Loading data...")

# load our doc2vec model that we trained
model = Doc2Vec.load(os.path.join(dirname,'models/yelp_model_10.d2v'))

# declare numpy zero ([0,..,0]) arrays to store the feature vectors and classification from the model
train_arrays = numpy.zeros((int(sys.argv[1]), int(sys.argv[3])))
train_labels = numpy.zeros(int(sys.argv[1]))

# create a logistic regression classifier
classifier = LogisticRegression()

# take our train reviews from the model, and put them in array, good reviews first, bad reviews second half of array
for i in range((int(sys.argv[1])/2)):
    prefix_train_pos = 'good_' + str(i)
    prefix_train_neg = 'bad_' + str(i)

    # get feature vectors from the model by using our previously declared labels
    pos_review = model.docvecs[prefix_train_pos]
    neg_review = model.docvecs[prefix_train_neg]

    # add the positive review from i=0
    train_arrays[i] = pos_review
    train_labels[i] = 1

    # add negative review from i=total_reviews/2
    train_arrays[(int(sys.argv[1])/2) + i] = neg_review
    train_labels[(int(sys.argv[1])/2) + i] = 0

# train the logistic regression classifier
classifier.fit(train_arrays, train_labels)

print("Ready")

# try and infer sentiment from inputted text
sentence = raw_input()
while sentence != "":
    labeled_sentence = utils.to_unicode(sentence).split()
    inferred_vector = model.infer_vector(labeled_sentence)
    print(model.docvecs.most_similar([inferred_vector]))
    print ("STOP")
    sentence = raw_input()

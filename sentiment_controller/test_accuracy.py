# ARGS:
# 1: total train reviews
# 2: number of iterations (for csv output)
# 3: size of vector

# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

# numpy
import numpy
import json
import time
# random
from random import shuffle
# classifier
from sklearn.linear_model import LogisticRegression
# our LabeledLineSentence class
from yelp_labeled_line_sentence import YelpLabeledLineSentence

from sklearn.linear_model import SGDClassifier

import os
import sys
import csv



dirname = os.path.dirname(__file__)
with open('result.csv', 'a') as f:
    # load our doc2vec model that we trained
    model = Doc2Vec.load(os.path.join(dirname,'models/yelp_model.d2v'))

    # create an array of LabeledLineSentences for previously unseen
    # good and bad reviews
    # this does some basic formatting of the text as well to make it more
    # digestible by gensim and sklearn
    good = YelpLabeledLineSentence(os.path.join(dirname, 'tests/good-bad/review_test_5_star.json'), 'good', 6254)
    bad = YelpLabeledLineSentence(os.path.join(dirname, 'tests/good-bad/review_test_1_star.json'), 'bad', 6254)

    # take our train reviews from the model, and put them in array, good reviews first, bad reviews second half of array
    train_arrays = numpy.zeros((int(sys.argv[1]), int(sys.argv[3])))
    train_labels = numpy.zeros(int(sys.argv[1]))

    # create a logistic regression classifier
    classifier = LogisticRegression()

    # take our train reviews from the model, and put them in array, good reviews first, bad reviews second half of array
    for i in range((int(sys.argv[1])/2)):
        prefix_train_pos = 'good_' + str(i)
        prefix_train_neg = 'bad_' + str(i)

        pos_review = model.docvecs[prefix_train_pos]
        neg_review = model.docvecs[prefix_train_neg]

        train_arrays[i] = pos_review
        train_labels[i] = 1

        train_arrays[(int(sys.argv[1])/2) + i] = neg_review
        train_labels[(int(sys.argv[1])/2) + i] = 0

    classifier.fit(train_arrays, train_labels)


    # take our test reviews from the model, and put them in array, good reviews first, bad reviews second half of array
    # for each review, we'll infer the review's vector against our model

    test_arrays = numpy.zeros((12508, int(sys.argv[3])))
    test_ratings = numpy.zeros(12508)

    test_labels = numpy.zeros(12508)

    good_correct = 0
    good_total = 0
    bad_correct = 0
    bad_total = 0

    for i, review in enumerate(good):
        test_arrays[i] = model.infer_vector(review[0])
        test_labels[i] = 1
        test_ratings[i] = review[1][2]

        prediction = classifier.predict([test_arrays[i]])
        if prediction == 1:
            good_correct += 1
        good_total +=1

    for i, review in enumerate(bad):
        test_arrays[i + 6254] = model.infer_vector(review[0])
        test_labels[i + 6254] = 0
        test_ratings[i + 6254] = review[1][2]

        prediction = classifier.predict([test_arrays[6254 + i]])
        if prediction == 0:
            bad_correct += 1
        bad_total += 1

    # print the accuracy of our classifier
    accuracy=classifier.score(test_arrays, test_labels) * 100
    print("Classifier reports a {}% accuracy".format(accuracy))

    # check good accuracy
    print("Out of {} good reviews, {} were correctly predicted".format(good_total,good_correct))

    # check bad accuracy
    print("Out of {} bad reviews, {} were correctly predicted".format(bad_total,bad_correct))

    print(test_arrays[0])

    tsne = TSNE(n_components=2)
    test_arrays_tsne = tsne.fit_transform(test_arrays)
    print(test_arrays_tsne)

    writer = csv.writer(f)
    writer.writerow([int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),accuracy])

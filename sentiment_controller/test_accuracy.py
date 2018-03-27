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
from sklearn.feature_extraction.text import CountVectorizer

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
    good = YelpLabeledLineSentence(os.path.join(dirname, '../data/test_10000.json'), 'good', 1000)
    bad = YelpLabeledLineSentence(os.path.join(dirname, '../data/test_10000.json'), 'bad', 1000)

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

    test_arrays_good = numpy.zeros((1000, int(sys.argv[3])))
    test_ratings_good = numpy.zeros(1000)
    test_labels_good = numpy.zeros(1000)

    test_arrays_bad = numpy.zeros((1000, int(sys.argv[3])))
    test_ratings_bad = numpy.zeros(1000)
    test_labels_bad = numpy.zeros(1000)

    good_correct = 0
    good_total = 0
    bad_correct = 0
    bad_total = 0

    for i, review in enumerate(good):
        test_arrays_good[i] = model.infer_vector(review[0])
        test_labels_good[i] = 1
        test_ratings_good[i] = review[1][2]

    for i, review in enumerate(bad):
        test_arrays_bad[i] = model.infer_vector(review[0])
        test_labels_bad[i] = 0
        test_ratings_bad[i] = review[1][2]

    # print the accuracy of our classifier
    accuracy=classifier.score(test_arrays_good, test_labels_good) * 100
    print("Classifier reports a {}% accuracy for good reviews".format(accuracy))

    accuracy=classifier.score(test_arrays_bad, test_labels_bad) * 100
    print("Classifier reports a {}% accuracy for bad reviews".format(accuracy))

    # for dim in range(1, int(sys.argv[3])):
    #     # plot probability of review being good vs feature vector value
    #     plt.scatter(test_arrays_good[:,dim], classifier.predict_proba(test_arrays_good)[:,1], color='green')
    #     plt.scatter(test_arrays_bad[:,dim], classifier.predict_proba(test_arrays_bad)[:,1], color='red')
    #
    #     plt.ylabel('Probability of Review Being Good')
    #     plt.xlabel('dim={}'.format(dim))
    #     plt.show()

    # reduce the n-dimensional feature vector to n=1 using t-SNE
    tsne = TSNE(n_components=1)
    test_arrays_tsne_good = tsne.fit_transform(test_arrays_good)
    test_arrays_tsne_bad = tsne.fit_transform(test_arrays_bad)

    print(test_arrays_tsne_bad[0])

    # plot probability of review being good vs feature vector value
    plt.scatter(test_arrays_tsne_good, classifier.predict_proba(test_arrays_good)[:,1], color='green')
    plt.scatter(test_arrays_tsne_bad, classifier.predict_proba(test_arrays_bad)[:,1], color='red')

    plt.ylabel('Probability of Review Being Good')
    plt.xlabel('t-SNE reduced feature vector (dim=1)')
    plt.show()

    # # reduce the n-dimensional feature vector to n=1 using t-SNE
    # tsne = TSNE(n_components=2)
    # test_arrays_tsne_good = tsne.fit_transform(test_arrays_good)
    # test_arrays_tsne_bad = tsne.fit_transform(test_arrays_bad)
    #
    # # plot feature vectors against each other
    # plt.scatter(test_arrays_tsne_good[:,0], test_arrays_tsne_good[:,1], color='green')
    # plt.scatter(test_arrays_tsne_bad[:,0], test_arrays_tsne_bad[:,1], color='red')
    #
    # plt.ylabel('x1')
    # plt.xlabel('x2')
    # plt.show()

    writer = csv.writer(f)
    writer.writerow([int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),accuracy])

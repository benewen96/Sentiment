# ARGS:
# 1: total train reviews
# 2: number of iterations (for csv output)
# 3: size of vector
# 4: good/bad sizes

# import dependencies
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from pandas.plotting import parallel_coordinates
from random import shuffle
from sklearn.linear_model import LogisticRegression
from yelp_labeled_line_sentence import YelpLabeledLineSentence
from sklearn.linear_model import SGDClassifier
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import numpy
import json
import time
import os
import sys
import csv

# api key for plotly
plotly.tools.set_credentials_file(username='benewen', api_key='TfY49IiC6FNVkRtl2gWV')

dirname = os.path.dirname(__file__)

# csv to output accuracy results
with open('result.csv', 'a') as f:
    # load our doc2vec model that we trained
    model = Doc2Vec.load(os.path.join(dirname,'models/yelp_model_10.d2v'))

    # create an array of LabeledLineSentences for previously unseen
    # good and bad reviews
    # this does some basic formatting of the text as well to make it more
    # digestible by gensim and sklearn
    good = YelpLabeledLineSentence(os.path.join(dirname, '../data/review.json'), 'good', int(sys.argv[4]))
    bad = YelpLabeledLineSentence(os.path.join(dirname, '../data/review.json'), 'bad', int(sys.argv[4]))

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

    test_arrays_good = numpy.zeros((int(sys.argv[4]), int(sys.argv[3])))
    test_ratings_good = numpy.zeros(int(sys.argv[4]))
    test_labels_good = numpy.zeros(int(sys.argv[4]))

    test_arrays_bad = numpy.zeros((int(sys.argv[4]), int(sys.argv[3])))
    test_ratings_bad = numpy.zeros(int(sys.argv[4]))
    test_labels_bad = numpy.zeros(int(sys.argv[4]))

    test_arrays = numpy.zeros((int(sys.argv[4]) * 2, int(sys.argv[3])))
    test_ratings = numpy.zeros(int(sys.argv[4]) * 2)
    test_labels = numpy.zeros(int(sys.argv[4]) * 2)

    good_correct = 0
    good_total = 0
    bad_correct = 0
    bad_total = 0


    for i, review in enumerate(good):
        test_arrays_good[i] = model.infer_vector(review[0])
        test_labels_good[i] = 1
        test_ratings_good[i] = review[1][2]
        test_arrays[i] = model.infer_vector(review[0])
        test_labels[i] = 1

    for i, review in enumerate(bad):
        test_arrays_bad[i] = model.infer_vector(review[0])
        test_labels_bad[i] = 0
        test_ratings_bad[i] = review[1][2]
        test_arrays[i + int(sys.argv[4])] = model.infer_vector(review[0])
        test_labels[i + int(sys.argv[4])] = 0

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
    #
    panda_array = pd.DataFrame(test_arrays,columns=[0,1,2,3,4,5,6,7,8,9])
    plt.figure()
    parallel_coordinates(panda_array)
    plt.show()

    # data = [
    #     go.Parcoords(
    #         line = dict(color = test_labels, colorscale = [[0,'#FF0000'],[1,'#008000']]),
    #         dimensions = list([
    #             dict(label = 'x0', values = panda_array[0]),
    #             dict(label = 'x1', values = panda_array[1]),
    #             dict(label = 'x2', values = panda_array[2]),
    #             dict(label = 'x3', values = panda_array[3]),
    #             dict(label = 'x4', values = panda_array[4]),
    #             dict(label = 'x5', values = panda_array[5]),
    #             dict(label = 'x6', values = panda_array[6]),
    #             dict(label = 'x7', values = panda_array[7]),
    #             dict(label = 'x8', values = panda_array[8]),
    #             dict(label = 'x9', values = panda_array[9])
    #         ])
    #     )
    # ]
    #
    # layout = go.Layout(
    #     plot_bgcolor = '#E5E5E5',
    #     paper_bgcolor = '#E5E5E5'
    # )
    #
    # fig = go.Figure(data = data, layout = layout)
    # py.iplot(fig, filename = 'all-reviews')

    # # plot probability of review being good vs feature vector value
    # plt.scatter(test_arrays_tsne_good[:,0], classifier.predict_proba(test_arrays_good)[:,1], color='green')
    # plt.scatter(test_arrays_tsne_bad[:,0], classifier.predict_proba(test_arrays_bad)[:,1], color='red')


    # # reduce the n-dimensional feature vector to n=1 using t-SNE
    # tsne = TSNE(n_components=1)
    # test_arrays_tsne_good = tsne.fit_transform(test_arrays_good)
    # test_arrays_tsne_bad = tsne.fit_transform(test_arrays_bad)
    #
    # # plot probability of review being good vs feature vector value
    # plt.scatter(test_arrays_tsne_good, classifier.predict_proba(test_arrays_good)[:,1], color='green')
    # plt.scatter(test_arrays_tsne_bad, classifier.predict_proba(test_arrays_bad)[:,1], color='red')
    #
    # plt.ylabel('Probability of Review Being Good')
    # plt.xlabel('t-SNE reduced feature vector (dim=1)')
    # plt.show()

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

    # writer = csv.writer(f)
    # writer.writerow([int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),accuracy])

# ARGS:
# 1: total train reviews
# 2: feature vector dimension
# 3: total reviews to insert to db

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
from pymongo import MongoClient

from datetime import datetime

import os
import sys

dirname = os.path.dirname(__file__)

client = MongoClient('mongodb://localhost:27017/')
reviews_file = os.path.join(dirname, '../data/review.json')

# load our doc2vec model that we trained
model = Doc2Vec.load(os.path.join(dirname,'models/yelp_model.d2v'))
# create a logistic regression classifier
classifier = LogisticRegression()
#classifier = SGDClassifier(loss='log', penalty='l1')

train_arrays = numpy.zeros((int(sys.argv[1]), int(sys.argv[2])))
train_labels = numpy.zeros(int(sys.argv[1]))

db = client.sentiment
reviews = db.reviews

def insertSource(source):
    with utils.smart_open(source) as fin:
        for item_no, line in enumerate(fin):
            if(item_no >= int(sys.argv[3])):
                break
            # our yelp reviews are in json, so we need to parse the text out
            parsed_line = json.loads(line)
            review = parsed_line['text']
            review_id = parsed_line['review_id']
            review_rating = parsed_line['stars']
            parsed_line['date'] = datetime.strptime(parsed_line['date'], '%Y-%m-%d')

            # predict sentiment
            labeled_sentence = utils.to_unicode(review).split()
            inferred_vector = model.infer_vector(labeled_sentence)
            prediction = classifier.predict([inferred_vector])
            if prediction == 0:
                parsed_line['sentiment'] = 'Bad'
            else:
                parsed_line['sentiment'] = 'Good'

            reviews.insert_one(parsed_line)

def insertBusinessData(source):
    if db.businesses.count() == 0:
        with utils.smart_open(source) as fin:
            for item_no, line in enumerate(fin):
                # our yelp reviews are in json, so we need to parse the text out
                parsed_line = json.loads(line)
                db.businesses.insert_one(parsed_line)
        print("Business data inserted")
    else:
        print("Business data already exists")


def loadData() :
    loadModel()
    insertSource(reviews_file)
    insertBusinessData(os.path.join(dirname, '../data/business.json'))
    print("Reviews inserted to database!")

def loadModel() :
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
    print("Loaded model!")

if reviews.count() == 0:
    loadData()
else:
    print("Reviews already exist")

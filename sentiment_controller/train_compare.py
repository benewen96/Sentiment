# This file is used for section 4.1.3 in the project report
# ARGS
# 1: number of reviews to train
# 2: number of iterations
# 3: size of vector

# import dependencies
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from sklearn.linear_model import LogisticRegression
from yelp_labeled_line_sentence import YelpLabeledLineSentence
from imdb_labeled_line_sentence import IMDBLabeledLineSentence
from random import shuffle
import numpy
import json
import time
import os
import sys

dirname = os.path.dirname(__file__)

start = time.clock()
print('Reading corpus...')

# source files for imdb reviews
imdb_sources = {os.path.join(dirname, '../data/aclImdb/train/pos'):'good',os.path.join(dirname, '../data/aclImdb/train/neg'):'bad', os.path.join(dirname, '../data/aclImdb/train/unsup'):'unsup'}

# load source files into LabeledLineSentences
imdb_reviews = IMDBLabeledLineSentence(imdb_sources)
yelp_reviews = YelpLabeledLineSentence(os.path.join(dirname, '../data/review.json'), 'train', 75000)

end = time.clock()
print('Done, time elapsed: {}'.format(end-start))

start = time.clock()
print('Training model...')

# declare yelp doc2vec model with parameters
yelp_model = Doc2Vec(min_count=1, window=10, size=400, sample=1e-4, negative=5, iter=1, workers=4)
# build our model vocabulary from our sentences
yelp_model.build_vocab(yelp_reviews)
# train our model
yelp_model.train(yelp_reviews, total_examples=yelp_model.corpus_count, epochs=yelp_model.iter)
# save model for future use
yelp_model.save(os.path.join(dirname,'models/yelp_model.d2v'))


# declare imdb doc2vec model with parameters
imdb_model = Doc2Vec(min_count=1, window=10, size=400, sample=1e-4, negative=5, iter=1, workers=4)
# build our model vocabulary from our sentences
imdb_model.build_vocab(imdb_reviews)
# train our model
imdb_model.train(imdb_reviews, total_examples=imdb_model.corpus_count, epochs=imdb_model.iter)
# save model for future use
imdb_model.save(os.path.join(dirname,'models/imdb_model.d2v'))

end = time.clock()
print('Done, time elapsed: {}'.format(end-start))

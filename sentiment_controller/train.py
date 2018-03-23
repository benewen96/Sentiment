# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
# numpy
import numpy
import json
import time

# classifier
from sklearn.linear_model import LogisticRegression
from yelp_labeled_line_sentence import YelpLabeledLineSentence

# lots of help from http://linanqiu.github.io/2015/10/07/word2vec-sentiment/
# also: http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/

import os

# random
from random import shuffle

dirname = os.path.dirname(__file__)

start = time.clock()
print('Reading corpus...')

reviews = YelpLabeledLineSentence(os.path.join(dirname, '../data/review.json'))
train_sentences = reviews.get_bad()+reviews.get_good()
print(train_sentences[1])
end = time.clock()
print('Done, time elapsed: {}'.format(end-start))

start = time.clock()
print('Training model...')

model = Doc2Vec(min_count=1, window=10, size=300, sample=1e-4, negative=5, iter=1, workers=4)
# build our model vocabulary from our sentences
model.build_vocab(train_sentences)

# train our model
model.train(reviews.random_train(), total_examples=model.corpus_count, epochs=model.iter)

# save model for future use
model.save(os.path.join(dirname,'models/yelp_model_newest_1.d2v'))
end = time.clock()
print('Done, time elapsed: {}'.format(end-start))

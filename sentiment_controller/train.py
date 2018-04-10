# ARGS
# 1: number of reviews to train
# 2: number of iterations
# 3: size of vector
# 4: model filename

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
import sys


# random
from random import shuffle

dirname = os.path.dirname(__file__)

start = time.clock()
print('Reading corpus...')

reviews = YelpLabeledLineSentence(os.path.join(dirname, '../data/review.json'), 'train', int(sys.argv[1]))

end = time.clock()
print('Done, time elapsed: {}'.format(end-start))

start = time.clock()
print('Training model...')

model = Doc2Vec(min_count=1, window=10, size=int(sys.argv[3]), sample=1e-4, negative=5, iter=int(sys.argv[2]), workers=4)
# build our model vocabulary from our sentences
model.build_vocab(reviews)

# train our model
model.train(reviews, total_examples=model.corpus_count, epochs=model.iter)

# save model for future use
model.save(os.path.join(dirname,'models/' + sys.argv[4] + 'd2v'))
end = time.clock()
print('Done, time elapsed: {}'.format(end-start))

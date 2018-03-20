import gensim
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_20newsgroups
import re
import numpy as np
import matplotlib.pyplot as plt
import os
import collections
import smart_open
import random
import json
import time

reviewJSON = open('../yelp_dataset/review_test_200.json')
model = gensim.models.doc2vec.Doc2Vec.load('yelp.model')

def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            parsed_line = json.loads(line)
            review = parsed_line['text']
            review_id = parsed_line['review_id']
            review_rating = parsed_line['stars']
            if len(review) > 0:
                if tokens_only:
                    yield gensim.utils.simple_preprocess(review)
                else:
                    # For training data, add tags
                    yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(review), [i])

start = time.clock()
print('Reading corpus...')

train_corpus = list(read_corpus(reviewJSON))
print('{} documents loaded.'.format(len(train_corpus)))

end = time.clock()
print('Done, time elapsed: {}'.format(end-start))

start = time.clock()
print('Creating document vectors...')

doc_vectors = []
for doc_id in range(len(train_corpus)):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    doc_vectors.append(inferred_vector)

end = time.clock()
print('Done, time elapsed: {}'.format(end-start))

start = time.clock()
print('Creating graph...')

X = doc_vectors
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

for i, doc in enumerate(doc_vectors):
    plt.annotate(
        i+1,
        xy=(X_tsne[i, 0], X_tsne[i, 1]), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])

end = time.clock()
print('Done, time elapsed: {}'.format(end-start))

plt.show()

from gensim import utils
import json
from gensim.models.doc2vec import LabeledSentence

# random
from random import shuffle
import os

# define our IMDBLabeledLineSentence class that we can use to feed into our Doc2Vec model
class IMDBLabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

    def __iter__(self):
        for source, sentiment in self.sources.items():
            print(source)
            count = -1
            for file in os.listdir(source):
                with open(os.path.join(source, file)) as f:
                    review = f.readline()
                    review_id,review_rating = file.split('_')
                    count += 1
                    yield LabeledSentence(utils.to_unicode(review).split(), [sentiment + '_%s' % count, review_id])

    def to_array(self):
        self.sentences = []
        for source, sentiment in self.sources.items():
            count = -1
            for file in os.listdir(source):
                with open(os.path.join(source, file)) as f:
                    review = f.readline()
                    review_id,review_rating = file.split('_')
                    count += 1
                    self.sentences.append(LabeledSentence(utils.to_unicode(review).split(), [sentiment + '_%s' % count]))
        return self.sentences

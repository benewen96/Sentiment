from gensim import utils
import json
from gensim.models.doc2vec import LabeledSentence

# random
from random import shuffle

# define our YelpLabeledLineSentence class that we can use to feed into our Doc2Vec model
class YelpLabeledLineSentence(object):
    def __init__(self, source):
        self.source = source

    def __iter__(self):
        with utils.smart_open(self.source) as fin:
            for item_no, line in enumerate(fin):
                # our yelp reviews are in json, so we need to parse the text out
                parsed_line = json.loads(line)
                review = parsed_line['text']
                review_id = parsed_line['review_id']
                review_rating = parsed_line['stars']
                yield LabeledSentence(utils.to_unicode(review).split(), [review_id])

    def to_array(self):
        self.sentences = []
        with utils.smart_open(self.source) as fin:
            for item_no, line in enumerate(fin):
                # our yelp reviews are in json, so we need to parse the text out
                parsed_line = json.loads(line)
                review = parsed_line['text']
                review_id = parsed_line['review_id']
                review_rating = parsed_line['stars']
                self.sentences.append(LabeledSentence(utils.to_unicode(review).split(), [review_id]))
        return self.sentences

    def get_good(self):
        self.good = []
        with utils.smart_open(self.source) as fin:
            for item_no, line in enumerate(fin):
                if(len(self.good) >= 50000):
                    return self.good
                # our yelp reviews are in json, so we need to parse the text out
                parsed_line = json.loads(line)
                review = parsed_line['text']
                review_id = parsed_line['review_id']
                review_rating = parsed_line['stars']
                if(review_rating == 5):
                    self.good.append(LabeledSentence(utils.to_unicode(review).split(), ['good_%s' % str(len(self.good)), review_id]))
        return self.good


    def get_bad(self):
        self.bad = []
        with utils.smart_open(self.source) as fin:
            for item_no, line in enumerate(fin):
                if(len(self.bad) >= 50000):
                    return self.bad
                # our yelp reviews are in json, so we need to parse the text out
                parsed_line = json.loads(line)
                review = parsed_line['text']
                review_id = parsed_line['review_id']
                review_rating = parsed_line['stars']
                if(review_rating == 1):
                    self.bad.append(LabeledSentence(utils.to_unicode(review).split(), ['bad_%s' % str(len(self.bad)), review_id]))
        return self.bad

    def random_train(self):
        shuffle(self.bad)
        shuffle(self.good)
        return self.bad+self.good

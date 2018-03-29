from gensim import utils
import json
from gensim.models.doc2vec import LabeledSentence

# random
from random import shuffle

# define our LabeledLineSentence class that we can use to feed into our Doc2Vec model
class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    # our yelp reviews are in json, so we need to parse the text out
                    parsed_line = json.loads(line)
                    review = parsed_line['text']
                    review_id = parsed_line['review_id']
                    review_rating = parsed_line['stars']
                    yield LabeledSentence(utils.to_unicode(review).split(), [prefix + '_%s' % item_no, review_id])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    # our yelp reviews are in json, so we need to parse the text out
                    parsed_line = json.loads(line)
                    review = parsed_line['text']
                    review_id = parsed_line['review_id']
                    review_rating = parsed_line['stars']
                    self.sentences.append(LabeledSentence(utils.to_unicode(review).split(), [prefix + '_%s' % item_no, review_id]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

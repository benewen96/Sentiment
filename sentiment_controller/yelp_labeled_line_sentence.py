from gensim import utils
import json
from gensim.models.doc2vec import LabeledSentence

# random
from random import shuffle

# define our YelpLabeledLineSentence class that we can use to feed into our Doc2Vec model
class YelpLabeledLineSentence(object):
    def __init__(self, source, mode, total):
        self.source = source
        self.total = total
        self.mode = mode

    def __iter__(self):
        good_count = 0
        bad_count = 0
        last = ""

        if(self.mode == 'train'):
            with utils.smart_open(self.source) as fin:
                for item_no, line in enumerate(fin):
                    if(good_count + bad_count >= self.total):
                        print('{} good reviews, {} bad reviews.'.format(good_count, bad_count))
                        return
                    # our yelp reviews are in json, so we need to parse the text out
                    parsed_line = json.loads(line)
                    review = parsed_line['text']
                    review_id = parsed_line['review_id']
                    review_rating = parsed_line['stars']

                    if(review_rating <= 2 and last != "bad"):
                        label = 'bad_%s' % bad_count
                        bad_count = bad_count + 1
                        last = "bad"
                    elif(review_rating >= 4 and last != "good"):
                        label = 'good_%s' % good_count
                        good_count = good_count + 1
                        last = "good"
                    yield LabeledSentence(utils.to_unicode(review).split(), [label, review_id])
        elif(self.mode == 'bad'):
            with utils.smart_open(self.source) as fin:
                for item_no, line in enumerate(fin):
                    if(bad_count >= self.total):
                        print('{} good reviews, {} bad reviews.'.format(good_count, bad_count))
                        return
                    # our yelp reviews are in json, so we need to parse the text out
                    parsed_line = json.loads(line)
                    review = parsed_line['text']
                    review_id = parsed_line['review_id']
                    review_rating = parsed_line['stars']

                    if(review_rating <= 2):
                        label = 'bad_%s' % bad_count
                        bad_count = bad_count + 1
                        yield LabeledSentence(utils.to_unicode(review).split(), [label, review_id, review_rating])
        elif(self.mode == 'good'):
            with utils.smart_open(self.source) as fin:
                for item_no, line in enumerate(fin):
                    if(good_count >= self.total):
                        print('{} good reviews, {} bad reviews.'.format(good_count, bad_count))
                        return
                    # our yelp reviews are in json, so we need to parse the text out
                    parsed_line = json.loads(line)
                    review = parsed_line['text']
                    review_id = parsed_line['review_id']
                    review_rating = parsed_line['stars']

                    if(review_rating >= 4):
                        label = 'good_%s' % good_count
                        good_count = good_count + 1
                        yield LabeledSentence(utils.to_unicode(review).split(), [label, review_id, review_rating])
        else:
            count = 0
            with utils.smart_open(self.source) as fin:
                for item_no, line in enumerate(fin):
                    if(count >= self.total):
                        print('{} good reviews, {} bad reviews.'.format(good_count, bad_count))
                        return
                    # our yelp reviews are in json, so we need to parse the text out
                    parsed_line = json.loads(line)
                    review = parsed_line['text']
                    review_id = parsed_line['review_id']
                    review_rating = parsed_line['stars']

                    label = ''
                    count = count + 1
                    yield LabeledSentence(utils.to_unicode(review).split(), [label, review_id, review_rating])


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

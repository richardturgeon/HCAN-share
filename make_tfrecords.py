import argparse
import ast
from   collections import defaultdict
import itertools as it
from   itertools import groupby
import re
import os, sys
import json
import numpy as np
import collections
from   gensim.models import Word2Vec
from   matplotlib import pyplot as plt
from   sklearn.manifold import TSNE
import logging
import pickle
import random

import tensorflow as tf


# YELP review rating by number of "stars" ranging from 1 to 5
MIN_STARS  = 1
MAX_STARS  = 5
RNG_STARS = MAX_STARS - MIN_STARS + 1

MYPATH = os.getcwd()


def logger(prefix):
    """logging setup """
    logging.getLogger().setLevel(logging.WARNING)
    formatter = logging.Formatter('%(asctime)-2s - %(levelname)-2s - %(message)s', "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(prefix + 'make_tfrecords.log')
    fh.setFormatter(formatter)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)

    logging.getLogger().addHandler(fh)
#   logging.getLogger().addHandler(ch)      # too much chatter
    return logging.getLogger(__name__)

def print_indented_text(text, indent=4, width=70):
    """ http://code.activestate.com/recipes/497010-format-a-text-block/ """
    width = width - indent
    pad = " " * indent
    out = []
    stack = [word for word in text.replace('\n', ' ').split(' ') if word]
    while stack:
        line = ""
        while stack:
            if len(line) == 0:
                line = stack.pop(0)
                continue
            if len(line) + len(stack[0]) >= width:
                break
            line += ' ' + stack.pop(0)
        out.append(pad + line)
    reform = "\n".join(out)
    print(reform)

def print_sample_text(source):
    ratings = sorted(source.keys())
    for rating in ratings:
        reviews = source[rating]
        for review in reviews:
            doclen = len(review.split())
            if doclen == 0:
                continue
            stars = rating + 1
            print(f"\n{stars} star rating for review of {doclen} words \n")
            print_indented_text(review, 8, 150)
    print(" ")

class Supervised_data(object):
    """ """
    def __init__(self, x_y_dict):
        self.x_y_dict = x_y_dict

    def get_length(self):
        return len(self.x_y_dict)

    def get_next(self):
        for i in range(len(self.x_y_dict)):
            entry   = self.x_y_dict[i]
            label   = entry['label']
            indexes = entry['idx']
            breaks  = entry['breaks']
            yield indexes, breaks, label


class feature_extractor(object):

    def balance(self, ndx, goal, batch_size, counters):
        ndx = int(ndx)
        bypass = counters[ndx] >= goal
        if not bypass:
            counters[ndx] += 1
            for i in range(len(counters)):
                if counters[i] < goal:
                    break
            else:
                goal += batch_size
        return goal, bypass


    def convert_to_tfr(self, data_source, path, split=.20, outpfx=''):
        """
            Each x is a variable length int32 array
            Each break  is a variable length int32 array
            Each y is a fixed length array[5] of int32, a one-hot encoding
        """
        train_path = outpfx + 'train-' + path
        test_path  = outpfx + 'test-'  + path
        print(f"writing train and test tfrecords to {train_path}, {test_path}")

        assert(split > 0. and split < 1.)
        total_recs  = data_source.get_length()
        test_count  = round(total_recs * split)
        train_count = total_recs - test_count

        with tf.python_io.TFRecordWriter(train_path) as train_writer:
            with tf.python_io.TFRecordWriter(test_path) as test_writer:
                for i, (x, breaks, y) in enumerate(data_source.get_next()):
                    x_int_list = tf.train.Int64List(value = x)
                    x_feature = tf.train.Feature(int64_list = x_int_list)

                    y_int_list = tf.train.Int64List(value = y)
                    y_feature = tf.train.Feature(int64_list = y_int_list)

                    b_int_list = tf.train.Int64List(value = breaks)
                    b_feature = tf.train.Feature(int64_list = b_int_list)

                    feature_dict = {'data': x_feature, 'label': y_feature, 'breaks': b_feature}
                    feature_set  = tf.train.Features(feature = feature_dict)
                    example      = tf.train.Example(features = feature_set)

                    if i <= test_count:
                        test_writer.write(example.SerializeToString())
                    else:
                        train_writer.write(example.SerializeToString())

            print(f"Training reviews: {train_count}")
            print(f"Testing  reviews: {test_count}")
            print(f"A total of {total_recs} tfrecords written")

            return train_count, test_count

    def __init__(self, json_path, embedding_size=512, balance=False, filter_years=None, print_samples=False,
                trunc=0, outpfx='', metadata_tag='', debug=False):
        """ """
        PUNCT = ['.','!','?']

        if filter_years is not None and filter_years != 0:
            if not isinstance(filter_years, list):
                filter_years = [filter_years]
            for i in range(len(filter_years)):
                filter_years[i] = str(filter_years[i])

        year_data = defaultdict(int)

        #collect sample reviews
        if print_samples:
            buckets = [''] * 10
            text_capture = {}
            for i in range(RNG_STARS):
                text_capture[i] = buckets[:]

        #optionally, prepare for sampling with class (review stars) balancing
        RECORD_BATCH_SIZE = 10000
        record_goal = RECORD_BATCH_SIZE
        record_counters = np.zeros(RNG_STARS, dtype=int)

        #record components
        labels = []
        reviews = []
        sentence_breaks = []

        max_sentence_words = 0
        max_document_sentences = 0
        max_document_words = 0

        #process json one line at a time
        file_size = os.path.getsize(json_path)

        english_vocab = set('i we is the for at to are this too very one and but she her he it'.split())

        french_vocab = set('je une mais votre'.split())
        french_count = 0
        german_vocab = set('ich auch wir nein'.split())
        german_count = 0

        with open(json_path,'r') as f:
            lineno = 0
            curr_pos = 0
            last_pct_cpt = 0
            nbr_retained = 0

            for line in f:
                lineno += 1
                if lineno % 5000 == 0:
                    pct_cpt = int(100 * (curr_pos / file_size))
                    if pct_cpt != last_pct_cpt:
                        last_pct_cpt = pct_cpt
                        sys.stdout.write(f"processing review {lineno} - {pct_cpt}% complete - reviews retained: {nbr_retained}  \r")
                        sys.stdout.flush()

                        # short runs for end-to-end debug
                        '''
                        if debug:
                            break
                        '''
                dic = ast.literal_eval(line)
                curr_pos += len(line)

                #extract records from "filter_years", a list, (to reduce dataset size)
                year = dic['date'][:4]
                year_data[year] += 1
                if filter_years:
                    if year not in filter_years:
                        continue

                #ensure review stars (the label) is in expected range
                stars = int(dic['stars']) - MIN_STARS
                if stars < 0 or stars > RNG_STARS:
                    continue

                #balance samples by label/stars
                if balance:
                    old_record_goal = record_goal
                    record_goal, bypass_record = self.balance(stars, record_goal, RECORD_BATCH_SIZE, record_counters)

                    if old_record_goal != record_goal:
                        s1 = record_counters[0]
                        s2 = record_counters[1]
                        s3 = record_counters[2]
                        s4 = record_counters[3]
                        s5 = record_counters[4]
                        print(f"line: {lineno} stars 1: {s1} 2: {s3} 3: {s3} 4: {s4} 5: {s5}                                       \r")

                    if bypass_record:
                        continue

                #sanitize text
                raw_text = dic['text']
                text = raw_text.lower()
                text = re.sub(" vs\. ", " vs ", text)
                text = re.sub("dr\. ", "dr ", text)
                text = re.sub("mr\. ", "mr ", text)
                text = re.sub("mrs\. ", "mrs ", text)
                text = re.sub(" ms\. ", " ms ", text)
                text = re.sub(" inc\. ", " inc ", text)
                text = re.sub(" llc\. ", " llc ", text)
                text = re.sub(" ltd\. ", " ltd ", text)
                text = re.sub("approx\. ", " approx ", text)
                text = re.sub("appt\. ", " appt ", text)
                text = re.sub(" apt\. ", " apt ", text)
                text = re.sub("i\.e\.", " ie ", text)
                text = re.sub("e\.g\.", " ie ", text)          # for example
                text = re.sub(" p\.s\.", "", text)
                text = re.sub(" p\.s", "", text)
                text = re.sub(" a\.m\.", " AM", text)
                text = re.sub(" p\.m\.", " PM", text)
                text = re.sub("\'re ", " are ", text)           # we're, you're, they're
                text = re.sub("(s)", "s", text)
                text = re.sub("\'", '', text)                   # \' char escape required - this wasnt getting done Nov 15, 2019
                text = re.sub('-', '', text)                    # e-mail, etc
                text = re.sub('`', '', text)                    # joe`s
                text = re.sub("\.{2,}", '.', text)
                text = re.sub('[^\w_|\.|\?|!]+', ' ', text)
                text = re.sub('\.', ' . ', text)
                text = re.sub('\?', ' ? ', text)
                text = re.sub('!', ' ! ', text)

                #tokenize, drop empty reviews
                text = text.split()
                text_units = len(text)
                if text_units == 0:
                    continue

                text_set = set(text)
                """
                if english_vocab.isdisjoint(text_set):
                    print(f"\nTHIS DOES NOT APPEAR TO BE ENGLISH\n {text}\n")
                    continue
                """    
                if not german_vocab.isdisjoint(text_set):
                    german_count += 1
                    #print(f"discarding german sentence: {german_count} \n{text}")
                    continue

                if not french_vocab.isdisjoint(text_set):
                    french_count += 1
                    #print(f"discarding french sentence: {french_count} \n{text}")
                    continue

                #capture representative examples 
                if print_samples:
                    bucket = int(text_units / 100)
                    if bucket < len(buckets):            # 100 token buckets
                        saver = text_capture[stars]
                        if saver[bucket] == '':
                            saver[bucket] = raw_text

                #apply truncation if any, ensure text is sentence-terminated
                if trunc > 0 and text_units > trunc:
                    text = text[:trunc]
                if text[-1] not in PUNCT:
                    text.append(PUNCT[0])

                sentences = []          # list of document sentences
                sentence = []           # list of sentence words
                breaks = []             # list of sentence lengths
                word_count = 0          # nbr words in document   

                #split text into sentences
                for t in text:
                    sentence.append(t)
                    if t in PUNCT:
                        sentence_len = len(sentence)
                        if sentence_len > 1:
                            word_count += sentence_len
                            sentences.append(sentence)
                            breaks.append(word_count)

                            if word_count > max_document_words:
                                max_document_words = word_count
                            if sentence_len > max_sentence_words:
                                max_sentence_words = sentence_len
                                if debug:
                                    print("")
                                    print(f"*** longest sentence encountered thus far - {sentence_len} tokens ***")
                                    print("")
                                    print(sentence)
                                    print("")
                                    #print(raw_text)
                                    #print("")
                        sentence = []

                #add split sentences to reviews
                if len(sentences) > 0:
                    reviews.append(sentences)
                    if len(sentences) > max_document_sentences:
                        max_document_sentences = len(sentences)

                    #add label and sentence boundaries 
                    labels.append(dic['stars'])
                    sentence_breaks.append(breaks)
                    nbr_retained += 1

        print('\nsaved %i reviews' % len(reviews))

        if print_samples:
            print(" ")
            print("A sampling of reviews based on rating and length....")
            print_sample_text(text_capture)

        years = sorted(year_data, reverse=True)
        for year in years:
            print(f"{year} - records: {year_data[year]}")

        #generate Word2Vec embeddings, use all processed raw text to train word2vec
        print("generating word2vec embeddings")
        self.all_sentences = [sentence for document in reviews for sentence in document]
        self.model = Word2Vec(self.all_sentences, min_count=5, size=embedding_size, workers=4, iter=5)
        self.model.init_sims(replace=True)

        #save all word embeddings to matrix
        print("saving word vectors to matrix")
        self.vocab = np.zeros((len(self.model.wv.vocab)+1,embedding_size))
        word2id = {}

        #first row of embedding matrix isn't used so that 0 can be masked
        for key, val in self.model.wv.vocab.items():
            idx = val.__dict__['index'] + 1
            self.vocab[idx, :] = self.model[key]
            word2id[key] = idx

        #normalize embeddings
        self.vocab -= self.vocab.mean()
        self.vocab /= (self.vocab.std()*2.5)

        #reset first row to 0
        self.vocab[0,:] = np.zeros((embedding_size))

        #add additional word embedding for unknown words
        self.vocab = np.concatenate((self.vocab, np.random.rand(1,embedding_size)))

        #index for unknown words
        unk = len(self.vocab)-1

        #capture word, sentence and rating (stars) distributions
        word_cap = 50
        word_size = 100
        word_hist = np.zeros(word_cap, dtype='int32')

        sent_cap = 40
        sent_size = 5
        sent_hist = np.zeros(sent_cap, dtype='int32')

        l_hist = np.zeros(RNG_STARS, dtype='int32')

        #convert words to word indicies
        print("converting words to indices")
        self.data = []

        for idx, document in enumerate(reviews):
            dic = {}
            dic['text'] = document

            len_sentences = len(document)
            sent_ndx = int(len_sentences / sent_size)
            if sent_ndx >= sent_cap:
                sent_ndx = sent_cap - 1
            sent_hist[sent_ndx] += 1

            # Up to this point the logic mimics that found in feature_extraction_yelp.py. 
            # indicies is a list of lists, each token of the latter is an English word.
            # Now we reach into the logic of tf_cnn.py to identify additional preprocessing
            # that can be done out-of-line, i.e. here.
            #
            # tf_cnn.py first flattens the document from a list of sentences and then flattens 
            # it into a list of words, i.e. run-on sentences.
            #
            # It then takes an in-storage document array like that built here (as tfrecords)
            # and applies LabelEncoder and LabelBinarizer to convert the floating point 'stars'
            # labels ranging from 1.0 to 5.0 to zero based integers 0 through 4 and then to 
            # one-hot encodings (as LabelBinarizer.transform does). We use these ranges
            # to generate the labels directly using these presumptions one record at
            # a time.

            indicies = []
            for sentence in document:
                len_sentence = len(sentence)
                word_ndx = int(len_sentence / word_size)
                if word_ndx >= word_cap:
                    word_ndx = word_cap - 1
                word_hist[word_ndx] += 1

                for word in sentence:
                    if word in word2id:
                        token = word2id[word]
                    else:
                        token = unk
                    indicies.append(token)

            # add digitized words and sentence boundaries
            dic['idx'] = indicies
            dic['breaks'] = sentence_breaks[idx]

            # convert label of stated range to a one-hot array
            dic['label'] = labels[idx]                              # the old label, just for reference
            int_lbl = int(labels[idx])
            ndx_lbl = int_lbl - MIN_STARS
            if ndx_lbl < 0 or ndx_lbl >= RNG_STARS:
                continue
            l_hist[ndx_lbl] += 1

            nbr_stars = int(labels[idx]) - MIN_STARS
            one_hot_stars = np.zeros(RNG_STARS).astype(np.int)
            one_hot_stars[nbr_stars] = 1
            dic['label'] = one_hot_stars                            # the new label, overwriting the above
            self.data.append(dic)

        # display distributions
        print(" ")
        print(f"Number of discarded French reviews: {french_count}")
        print(f"Number of discarded German reviews: {german_count}")

        print(" ")
        print("Review length distribution - #sentences")
        for i in range(sent_cap):
            tag = i * sent_size
            nbr_hits = sent_hist[i]
            if nbr_hits > 0:
                print("%4d - %4d: %7d" % (tag, tag + sent_size - 1, sent_hist[i]))

        print(" ")
        print("Review length distribution #words")
        for i in range(word_cap):
            tag = i * word_size
            nbr_hits = word_hist[i]
            if nbr_hits > 0:
                print("%4d - %4d: %7d" % (tag, tag + word_size - 1, word_hist[i]))

        print(" ")
        print("Rating summary (# stars)")
        distribution_list = []
        for i in range(RNG_STARS):
            print("%d - %7d" % ((i + MIN_STARS), l_hist[i]))
            distribution_list.append(l_hist[i])

        # capture tfrecord metadata
        # max_review_words is the number of words contained in the longest review.
        # classes is the number of stars assigned by the reviewer - currently one 
        # through five for a total of five classifications.

        # generate tfrecords 
        random.shuffle(self.data)
        self.iterable_data = Supervised_data(self.data)
        train_count, test_count = self.convert_to_tfr(self.iterable_data, "HCAN.tfrecords", outpfx=outpfx)

        metadata = dict(
            max_review_words=max_document_words,
            max_review_sentences=max_document_sentences,
            max_sentence_words=max_sentence_words,
            classes=RNG_STARS,
            balanced=balance,
            notes=metadata_tag,
#           distribution=distribution_list,      ### this causes JSON error - list of ints
            train_count=train_count,
            test_count=test_count
        )
        print(f"Review maximum sentences: {max_document_sentences} words: {max_document_words}")
        print(f"There are {RNG_STARS} classifications")
        with open(outpfx + 'HCAN-metadata.json', 'w') as f:
            json.dump(metadata, f)


    def visualize_embeddings(self):
        #get most common words
        print("getting common words")
        all_words = [word for sent in self.all_sentences for word in sent]
        counts = collections.Counter(all_words).most_common(500)

        #reduce embeddings to 2d using tsne
        print("reducing embeddings to 2D")
        embeddings = np.empty((500,embedding_size))
        for i in range(500):
            embeddings[i,:] = model[counts[i][0]]
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=7500)
        embeddings = tsne.fit_transform(embeddings)

        #plot embeddings
        print("plotting most common words")
        fig, ax = plt.subplots(figsize=(30, 30))
        for i in range(500):
            ax.scatter(embeddings[i,0],embeddings[i,1])
            ax.annotate(counts[i][0], (embeddings[i,0],embeddings[i,1]))
        plt.show()

    def get_embeddings(self):
        return self.vocab

    def get_data(self):
        return self.data


def main(yelp=None, balance=False, trunc=0, tag='', outpfx='', filter_years=None, print_samples=False, debug=False):
    """ """
    if outpfx:
        print(f"files created by this execution will be prefixed with {outpfx}")
        outpfx = outpfx + '-'

    if trunc > 0:
        print(f"reviews will be truncated to approx {trunc} words")

    logger(outpfx)
    print(sys.argv)

    #process json
    json_path = yelp

    fe = feature_extractor(
        json_path,
        512,
        trunc=trunc,
        balance=balance,
        filter_years=filter_years,
        outpfx=outpfx,
        metadata_tag=tag,
        print_samples=print_samples,
        debug=debug
    )

    vocab = fe.get_embeddings()
    data = fe.get_data()

    #create directory for saved model
    if not os.path.exists('./data'):
        os.makedirs('./data')

    embed_file = './data/' + outpfx + 'yelp16_embeddings'
    np.save(embed_file, vocab)
    print(f'vocabulary embeddings saved in {embed_file}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--yelp',
        type=str,
        help='Yelp json file path'
    )

    parser.add_argument(
        '--balance',
        default=False,
        action='store_true',
        help='balance review records by ratings (stars)'
    )

    parser.add_argument(
        '--trunc',
        type=int,
        default=0,
        help='truncate reviews to this word count'
    )

    parser.add_argument(
        '--filter_years',
        type=int,
        nargs='+',
        default=0,
        help='extract reviews from the specified year(s)'
    )

    parser.add_argument(
        '--tag',
        type=str,
        default='',
        help='string added to JSON metadata file'
    )

    parser.add_argument(
        '--outpfx',
        type=str,
        default='',
        help='prefix prepended to tfrecord, JSON and log files'
    )

    parser.add_argument(
        '--print_samples',
        default=False,
        action='store_true',
        help='print sample reviews by rating and length'
    )

    parser.add_argument(
        '--debug',
        default=False,
        action='store_true',
        help='limit sampling for quick end-to-end test'
    )

    args = vars(parser.parse_args())
    main(**args)


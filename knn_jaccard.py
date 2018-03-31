#! /usr/bin/python

"""
    CMPE-139
    Multiprocess text polarity classifier using KNN
    Jaccard similarity
"""

from collections import Counter
from multiprocessing import Pool
from functools import partial
from itertools import chain
import math, operator, random, re, sys
from stemming.porter2 import stem
import pandas as pd


def k_nearest_neighbors(X_test_inst, X_train, y_train, k):
    """Finds k nearest neighbors for test instance from training data"""

    similarities = [(y, jaccard_similarity(X_test_inst, x)) for x,y in zip(X_train, y_train)]
    similarities.sort(key=operator.itemgetter(1))
    return similarities[-k:]


def jaccard_similarity(q, d):
    """Calculate similarity between docs using Jaccard similarity"""

    intersection = len(q.intersection(d))
    union = len(q.union(d))
    return intersection / float(union)
    

def classify(k, X_train, y_train, X_test_inst):
    """Polarity classification of test document using KNN"""

    knn = k_nearest_neighbors(X_test_inst, X_train, y_train, k)
    return calculate_polarity(knn)


def calculate_polarity(knn):
    """Calculate polarity of document"""

    polarity = sum(p * s for p, s in knn) # weighted by similarity score

    # polarity is neutral, assign random polarity
#    if polarity == 0:
#        polarity = random.getrandbits(1)

    return polarity > 0


def tokenize(stop_words, doc):
    """Tokenizes the document"""

    # remove HTML tags
    doc = re.sub('<.*?>', ' ', doc)
    # remove non-alphabet
    doc = re.sub('[^a-zA-Z]', ' ', doc)

    tokens = [stem(w) for w in doc.lower().split() if len(w) > 1 and w not in stop_words]
    return tokens


def format_data(train_df, test_df, pool):
    tokenized_train_doc = pool.map(partial(tokenize, stop_words), train_df['doc'])
    tokenized_test_doc = pool.map(partial(tokenize, stop_words), test_df['doc'])

    all_terms = tokenized_train_doc + tokenized_test_doc
    term_doc_freq = Counter(chain.from_iterable(set(t) for t in all_terms))
    min_doc_freq = 2
    filter_terms = lambda t: set([w for w in t if term_doc_freq[w] > min_doc_freq])
    X_train = map(filter_terms, tokenized_train_doc)
    y_train = train_df['polarity']
    X_test = map(filter_terms, tokenized_test_doc)

    return X_train, y_train, X_test


def save(data):
    f = open(sys.argv[3] if len(sys.argv) > 3 else 'results.txt', 'w+')
    for i in data:
        f.write('+1\n' if i > 0 else '-1\n')
    f.close()


if __name__ == "__main__":
    pool = Pool(processes=None)
    stop_words = set(pd.read_csv('stop_words.txt', header=None, sep='\n', squeeze=True).values)

    if len(sys.argv) < 3:
        print('Missing arguments.\nex: ./pr1.py train.dat test.dat results.txt(optional)')
        sys.exit(1)

    train_fname = sys.argv[1]
    test_fname = sys.argv[2]
    train_df = pd.read_csv(train_fname, header=None, sep='\t|\n', names=['polarity','doc'], engine='python')
    test_df = pd.read_csv(test_fname, header=None, sep='\n', names=['doc'])

    print('Tokenizing documents...')

    X_train, y_train, X_test = format_data(train_df, test_df, pool)

    print('Done\nClassifying...')

    k = int(math.ceil(math.sqrt(len(X_test))))
    polarities = pool.map(partial(classify, k, X_train, y_train), X_test)

    save(polarities)


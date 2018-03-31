#! /usr/bin/python

"""
    CMPE-139
    Multiprocess text polarity classifier using KNN
    Cosine similarity
"""

import sys, re, random, operator, math
from multiprocessing import Pool
from functools import partial
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from stemming.porter2 import stem


def k_nearest_neighbors(X_test, X_train, y_train, k):
    """Finds k nearest neighbors for test instance from training data"""

    similarities = zip(y_train, cos_sim(X_test, X_train))
    similarities.sort(key=operator.itemgetter(1))
    return similarities[-k:]


def cos_sim(q, d):
    """Calculate cosine similarity for l2 normalized data"""

    return (q * d).toarray()[0]


def classify(k, X_train, y_train, X_test, X_test_idx):
    """Polarity classification of test document using KNN"""

    knn = k_nearest_neighbors(X_test[X_test_idx], X_train, y_train, k)
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
    # remove stop words after stemming
#    tokens = [w for w in tokens if w not in stop_words]

    return ' '.join(tokens)


def vectorize(train_df, test_df, pool):
    """Vectorizes the documents using TF-IDF"""

    vectorizer = TfidfVectorizer(norm='l2', min_df=2)
    stop_words = set(pd.read_csv('stop_words.txt', header=None, sep='\n', squeeze=True).values)

    tokenized_train_doc = pool.map(partial(tokenize, stop_words), train_df['doc'])
    tokenized_test_doc = pool.map(partial(tokenize, stop_words), test_df['doc'])

    # transpose X_train for cosine similarity calculation
    X_train = vectorizer.fit_transform(tokenized_train_doc).T
    y_train = train_df['polarity']
    X_test = vectorizer.transform(tokenized_test_doc)

    return X_train, y_train, X_test


def save(data):
    f = open(sys.argv[3] if len(sys.argv) > 3 else 'results.txt', 'w+')
    for i in data:
        f.write('+1\n' if i > 0 else '-1\n')
    f.close()


if __name__ == "__main__":
    pool = Pool(processes=None)

    if len(sys.argv) < 3:
        print('Missing arguments.\nex: ./knn.py train.dat test.dat results.txt(optional)')
        sys.exit(1)

    train_fname = sys.argv[1]
    test_fname = sys.argv[2]
    train_df = pd.read_csv(train_fname, header=None, sep='\t|\n', names=['polarity','doc'], engine='python')
    test_df = pd.read_csv(test_fname, header=None, sep='\n', names=['doc'])

    print('Tokenizing documents...')

    X_train, y_train, X_test = vectorize(train_df, test_df, pool)

    print('Done\nClassifying...')

    k = int(math.ceil(math.sqrt(X_test.shape[0])))
    polarities = pool.map(partial(classify, k, X_train, y_train, X_test),
                          xrange(X_test.shape[0]))

    save(polarities)


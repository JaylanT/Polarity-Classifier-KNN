# Polarity-Classifier-KNN
CMPE-139 Program 1

IMDB dataset of 25000 labeled training data and 25000 unlabeled testing data

84% accuracy with Jaccard similarity. Cosine similarity not tested.

Runtime of approximately 7 minutes for cosine similarity and 30 minutes for Jaccard similarity on an i5-6600k.

Running:

./knn_algo.py train.dat test.dat out_file

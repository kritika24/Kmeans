# Reference: http://scikit-learn.org/0.16/datasets/index.html
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import scipy.io as sio
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
import pprint as pprint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics


#converting the data into train and test
train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')
pprint.pprint(test)
names=list[train.target_names]
labels=train.target


#vectorize the train dataset
categories=None
vectorizer = TfidfVectorizer(max_df=0.5,
                             min_df=2,
                             stop_words='english')
X = vectorizer.fit_transform(train.data)
pprint.pprint(X)
X.shape


#Multinomial Naive Bayers Classifier
from sklearn.naive_bayes import MultinomialNB
vectors_test = vectorizer.transform(test.data)
clf = MultinomialNB(alpha=.01)
clf.fit(X, train.target)
pred = clf.predict(vectors_test)
metrics.f1_score(test.target, pred, average='weighted')


#clustering of the dataset
true_k=12
km = KMeans(n_clusters=20, init='k-means++', max_iter=100, n_init=1)
km.fit(X)
rder_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("cluster %d:" % i)
    for ind in order_centroids[i,:20]:
        print('%s' % terms[ind])
    print()

'''#Filtering text
newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'),categories=categories)
vectors_test = vectorizer.transform(newsgroups_test.data)
pred = clf.predict(vectors_test)
metrics.f1_score(pred, newsgroups_test.target, average='weighted')'''

def show_top10(classifier, vectorizer, categories):
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        top10 = np.argsort(classifier.coef_[i])[-10:]
        print("%s: %s" % (category, " ".join(feature_names[top10])))
        
show_top10(clf, vectorizer, train.target_names)



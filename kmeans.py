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

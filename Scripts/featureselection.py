import os
import re
import sklearn
import pandas as pd 
import numpy as np 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest,chi2



chi_square_parameters = [[5000],[4000],[4000],[4000],[4000],[4000],[4000],['all'],['all'],['all'],
['all'],['all'],[2500],['all']]


def featureselection(train_corpus_tf_idf,test_corpus_tf_idf,label_train,train_choice,test_choice):

	val = (train_choice)*3 + test_choice

	k = chi_square_parameters[val][0]

	if(k=='all'):
		K = train_corpus_tf_idf.shape[1]
	else:
		K = k 

	vectorizer_chi2 = SelectKBest(chi2,k=K)

	chi_train_corpus_tf_idf = vectorizer_chi2.fit_transform(train_corpus_tf_idf,label_train)

	chi_test_corpus_tf_idf = vectorizer_chi2.transform(test_corpus_tf_idf)

	return [chi_train_corpus_tf_idf,chi_test_corpus_tf_idf]

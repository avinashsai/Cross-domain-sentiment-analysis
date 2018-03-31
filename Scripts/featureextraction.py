import os
import re
import sklearn
import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer

vector_parameters = [[2,0.8],[3,0.8],[3,0.8],[3,0.8],[3,0.8],[3,0.8],[3,0.8],[3,0.8],[3,0.8],
[3,0.8],[3,0.8],[3,0.8]]


def featureextraction(train_corpus,test_corpus,label_train,train_choice,test_choice):

	val = (train_choice)*3 + test_choice

	param  = vector_parameters[val]
	mindf = param[0]
	maxdf = param[1]

	vectorizer = TfidfVectorizer(min_df=mindf,max_df=maxdf,use_idf=True,sublinear_tf=True,stop_words='english')

	train_corpus_tf_idf = vectorizer.fit_transform(train_corpus,label_train)

	test_corpus_tf_idf = vectorizer.transform(test_corpus)

	return [train_corpus_tf_idf,test_corpus_tf_idf]



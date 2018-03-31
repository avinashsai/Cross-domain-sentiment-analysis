import os
import re
import sys
import sklearn
import pandas as pd 
import numpy as np 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score,confusion_matrix
from featureextraction import featureextraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest,chi2




from train import *
from featureextraction import *
from featureselection import *
from preprocess import *
from bagging import *


def getvalue(val):
	if(val=="books"):
		return 0
	elif(val=="dvd"):
		return 1
	elif(val=="electronics"):
		return 2
	else:
		return 3

def getchoices(train,test):

	if(train=="books"):
		train_choice = 0
		if(test=="dvd"):
			test_choice = 0
		elif(test=="electronics"):
			test_choice = 1
		else:
			test_choice = 2

	if(train=="dvd"):
		train_choice = 1
		if(test=="books"):
			test_choice = 0
		elif(test=="electronics"):
			test_choice = 1
		else:
			test_choice = 2

	if(train=="electronics"):
		train_choice = 2
		if(test=="books"):
			test_choice = 0
		elif(test=="dvd"):
			test_choice = 1
		else:
			test_choice = 2

	if(train=="kitchen"):
		train_choice = 3
		if(test=="books"):
			test_choice = 0
		elif(test=="dvd"):
			test_choice = 1
		else:
			test_choice = 2

	return train_choice,test_choice


if __name__ == "__main__":

	train = sys.argv[1]

	test = sys.argv[2]

	
	train_choice,test_choice = getchoices(train,test)

	pre_choice_train = getvalue(train)
	pre_choice_test = getvalue(test)

	corpus_train,label_train,corpus_test,label_test = preprocessing(pre_choice_train,pre_choice_test)

	
	train_corpus_tf_idf,test_corpus_tf_idf = featureextraction(corpus_train,corpus_test,label_train,train_choice,test_choice)

	chi_train_corpus_tf_idf,chi_test_corpus_tf_idf = featureselection(train_corpus_tf_idf,test_corpus_tf_idf,label_train,train_choice,test_choice)

	classifier_train(chi_train_corpus_tf_idf,label_train,chi_test_corpus_tf_idf,label_test,train_choice,test_choice)

	bagging_train(chi_train_corpus_tf_idf,label_train,chi_test_corpus_tf_idf,label_test,train_choice,test_choice)
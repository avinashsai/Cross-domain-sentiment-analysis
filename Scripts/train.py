import os
import re
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

import matplotlib.pyplot as plt 


def classify_sentiment(classifier,chi_train_corpus_tf_idf,chi_test_corpus_tf_idf,label_train,label_test):
    clf   = classifier
    clf.fit(chi_train_corpus_tf_idf,label_train)
    pred = clf.predict(chi_test_corpus_tf_idf)
    accuracy = clf.score(chi_test_corpus_tf_idf,label_test)
    cm = confusion_matrix(pred,label_test)
    f1 = f1_score(pred,label_test)
    return accuracy,f1,cm 




def classifier_train(chi_train_corpus_tf_idf,label_train,chi_test_corpus_tf_idf,label_test,train_choice,test_choice):

    rbf_parameters = [[0.9],[0.9],[0.9],[0.9],[0.9],[0.8],[0.9],[0.9],[0.8],[0.9],[0.9],[0.9]]

    val = (train_choice)*3 + test_choice

    Gamma = rbf_parameters[val][0]

    classifiers = [LogisticRegression(random_state=0),SVC(gamma=Gamma),DecisionTreeClassifier(random_state=0),
                   SVC(kernel='linear',gamma=Gamma),MultinomialNB(),KNeighborsClassifier(n_neighbors=7)]

    
    accu = []
    
    classify = ["LR","SVM-RBF","DT","SVM-L","MNB","KNN"]

    for i in range(len(classifiers)):
        acc,f1,cm = classify_sentiment(classifiers[i],chi_train_corpus_tf_idf,chi_test_corpus_tf_idf,label_train,label_test)

        accu.append(acc)

        print(classify[i]+" "+"F1 score is :",f1)
        print(classify[i]+" "+"confusion matrix is:")
        print(cm)
        print("\n")
    

    

    
    


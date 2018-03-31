# Cross-domain-sentiment-analysis

This repository consists of code for the submitted research paper on Cross Domain Sentiment Analysis. A statistical feature selection method is used to select best Set of features.


The proposed method is tested on reviews from 4 different domains. The data set can be downloaded from https://www.cs.jhu.edu/~mdredze/datasets/sentiment/.


We used 1600 reviews for training and 400 reviews for testing. The approach has been tested on all pairs of reviews.


# Installation

To execute files in the repository, you need to install these packages:

**Scikit-Learn**

Install using http://scikit-learn.org/stable/install.html

**NLTK**

Install using https://www.nltk.org/install.html

# Running the tests

To execute files in the repository

```
git clone https://github.com/avinashsai/Cross-domain-sentiment-analysis.git

```
Reviews are in XML Format stored in **sorted_acl_data** folder. To Extract original Reviews run

```
cd Scripts
python3 extract.py

```
Reviews will be stored in **Actualdata** Folder. 

To run the approach :

```
cd Scripts
python3 main.py
```

You will get results tested on all Machine Learning Classifiers


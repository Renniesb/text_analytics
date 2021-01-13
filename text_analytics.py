# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 22:24:13 2020

@author: Rennie.Bevineau
"""


import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer
pd.set_option('display.max_columns', 20)


doc = pd.read_csv('Tweets.csv') #capture the whole csv document
text_col = doc.text #capture only the text column for data analysis


#run tf-idf on the text column
vectorizer = TfidfVectorizer(use_idf=True)
tfIdf = vectorizer.fit_transform(text_col)
features_names = vectorizer.get_feature_names()

dense = tfIdf.todense()

df = pd.DataFrame(dense, columns=features_names)

toplist = []

for index,row in df.iterrows():
    # Find the 5 words with the highest td-idf score
    sor = row[:-1].round(2).sort_values(ascending=False)[:5].to_dict()
    toplist.append(sor)
    

 
doc['tfidf']= toplist
doc.to_csv("Tweets_tfIdf.csv")
        
        
#Using this data set, you will create a text analytics
#Python application that extracts themes from each comment using term frequency-inverse 
#document frequency (TF-IDF) or simple word counts. For the deliverable, provide your Python file
#and a .csv with your results added as a column to the original data set.
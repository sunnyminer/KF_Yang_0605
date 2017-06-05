from __future__ import print_function
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import lda
import os

n_samples = 2000
n_features = 1000
n_topics = 10
n_top_words = 20

def textExtract(doc):
#	print("Currently loading " + doc)
	result = ""
	with open(doc) as d:
		for line in d:
		    result += line
        #print(result)
	return result

#read documents in certain directory
myPath = "DucRaw";
allTextDoc = [myPath + '/' + f for f in os.listdir(myPath) if f.endswith(".txt") ]
#print(allTextDoc)
allText = []
for text in allTextDoc:
	print(text)
	result = textExtract(text)
	allText.append(result)

print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')
tf = tf_vectorizer.fit_transform(allText)

print("Fitting LDA models with tf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
model= lda.LDA(n_topics=n_topics,n_iter=5,random_state=1)
model.fit(tf)

tf_feature_names = tf_vectorizer.get_feature_names()
print (tf_feature_names)
print(model.topic_word_)
print(model.doc_topic_)
print("len",len(model.topic_word_))

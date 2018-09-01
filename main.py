import nltk
nltk.download('sentence_polarity')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
import random
import string
from nltk.corpus import movie_reviews as mr 
from nltk.corpus import stopwords
from nltk.sentiment import SentimentAnalyzer
import numpy as np
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.probability import FreqDist
from sklearn.svm import SVC, LinearSVC
import os.path
import pickle
import re
from itertools import chain
from nltk.classify import NaiveBayesClassifier as nbc




def add_feature(reviews, include_negation, keep_adj_adv, stemming):

	#including negation 
	if include_negation:
		sentim_analyzer = SentimentAnalyzer() # use nltk library
		reviews = [nltk.sentiment.util.mark_negation(review) for review in reviews]

	#using only adjectives and adverbs
	if keep_adj_adv:
		i = 0
		for review, label in reviews:
			# print(i)
			pos_tagged = nltk.pos_tag(review)	 # create part of speech tag
			review = [word for word, tag, in zip(review,pos_tagged) if re.search(r'^(JJ|RB)R?S?$', tag[1])] # look for adjectives and adverbs 
			reviews[i] = tuple([review,label]) # create new document
			i += 1

	#using stemming
	if stemming:
		pstem = nltk.stem.PorterStemmer() 
		i = 0
		for review,label in reviews:
			new_review = []
			for word in review:
				if word == 'oed': #nltk bug: http://stackoverflow.com/questions/41517595/nltk-stemmer-string-index-out-of-range
					continue
				new_review.append(pstem.stem(word))
			reviews[i] = tuple([new_review,label])
			i += 1


	return reviews


stop = stopwords.words('english')

#Use this commented section if the pcikle file does not exist
# documents = [([w for w in mr.words(i) if w.lower() not in stop and w.lower() not in string.punctuation], i.split('/')[0])
# 					 for i in mr.fileids()]

# save_doc = open('movie_doc.pickle', 'wb')
# pickle.dump(documents, save_doc) # save the dataset for the future use
# save_doc.close()


# getting the saved doc(if it already exists)
save_doc = open('movie_doc.pickle', 'rb')
documents = pickle.load(save_doc) # load the dataset
save_doc.close()


# shuffling reviews 
random.shuffle(documents)

# modify the boolean values to add whichever feature you want to test
documents = add_feature(documents,True,True,True) # add_feature(review,keep_adjectives,include_negatives,rem_stopwords)

word_features = FreqDist(chain(*[i for i,j in documents]))
word_features = list(word_features.keys())[:5000] 


# splitting into 80% for training data and 20% for testing data
numtrain = int(len(documents) * 80 / 100)

train_set = [({i:(i in tokens) for i in word_features}, tag) for tokens,tag in documents[:numtrain]]
test_set = [({i:(i in tokens) for i in word_features}, tag) for tokens,tag in documents[numtrain:]]

# Trianing and testing different learning classifiers

nb_classifier = nbc.train(train_set)							#Naive Bayes Classifier
print("Accuracy for Naive Bayes classifier is: ",nltk.classify.accuracy(nb_classifier, test_set))


dt_classifier = nltk.DecisionTreeClassifier.train(train_set)	#Decision Tree Classifier	
print("Accuracy for Decision Tree classifier is: ",nltk.classify.accuracy(dt_classifier,test_set))


svm_classifier = nltk.classify.SklearnClassifier(LinearSVC())	#Support Vector Machine Classifier	
svm_classifier.train(train_set)
print("Accuracy for Support Vector Machine classifier is: ",nltk.classify.accuracy(svm_classifier,test_set) )










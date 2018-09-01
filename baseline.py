import nltk
nltk.download('sentence_polarity')
import random
from nltk.corpus import movie_reviews
import numpy as np

#getting all reviews
sentences = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories()
			 for fileid in movie_reviews.fileids(category)]

#getting positive and negative words and storing them as sets for O(1) access 
with open('negative-words.txt', encoding = "ISO-8859-1") as f1:
    bad_words = f1.read().splitlines()

with open('positive-words.txt', encoding = "ISO-8859-1") as f2:
    good_words = f2.read().splitlines()

good_words = set(good_words)
bad_words = set(bad_words)

#Iterating over all the sentences and using the model to predict each sentence as positive or negative
correct_predictions = 0
for i in range(0,2000):
	pos_score = 0
	neg_score = 0
	for word in sentences[i][0]:
		if word in good_words:
			pos_score += 1
		if word in bad_words:
			neg_score += 1
	if pos_score >= neg_score:
		predicted_label = 1 #positive
	else:
		predicted_label = 0 #negative
	if predicted_label == 0 and sentences[i][1] == 'neg':
		correct_predictions += 1
	elif predicted_label == 1 and sentences[i][1] == 'pos':
		correct_predictions += 1


accuracy = correct_predictions / 2000 * 100

print("Accuracy for the baseline model:",accuracy)








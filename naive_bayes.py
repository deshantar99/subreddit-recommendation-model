from data import Dataset, Labels
from utils import evaluate, getWordFreq, updateWordFreq
import math
import os, sys
import numpy as np
import webbrowser
import datetime
import pickle


class NaiveBayes:
	def __init__(self):
		# total number of documents in the training set.
		self.n_doc_total = 0
		# total number of documents for each label/class in the trainin set.
		self.n_doc = {l: 0 for l in Labels}
		# frequency of words for each label in the trainng set.
		self.vocab = {l: {} for l in Labels}

		self.stop_words = {'a','and','the','i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}


	def train(self, ds):
		"""
		ds: list of (id, x, y) where id corresponds to document file name,
		x is a string for the email document and y is the label.

		TODO: Loop over the dataset (ds) and update self.n_doc_total,
		self.n_doc and self.vocab.
		"""
		for d in ds:
			self.n_doc_total += 1
			self.n_doc[d[1]] += 1
			if(d[1] in self.vocab):
				updateWordFreq(self.vocab[d[1]], d[0])
			else:
				self.vocab[d[1]] = getWordFreq(d[0])

		# with open('trained_likelihoods.txt', 'wb') as likelihood_file:
		# 	pickle.dump(self.n_doc_total, likelihood_file)
		# 	pickle.dump(self.n_doc, likelihood_file)
		# 	pickle.dump(self.vocab, likelihood_file)



	def predict(self, x):
		"""
		x: string of words in the document.

		TODO: Use self.n_doc_total, self.n_doc and self.vocab to calculate the
		prior and likelihood probabilities.
		Add the log of prior and likelihood probabilities.
		Use MAP estimation to return the Label with hight score as
		the predicted label.
		"""
		# with open('trained_likelihoods.txt', 'rb') as likelihood_file:
		# 	self.n_doc_total = pickle.load(likelihood_file)
		# 	self.n_doc = pickle.load(likelihood_file)
		# 	self.vocab = pickle.load(likelihood_file)

		x = x.lower().split()
		priors = [0] * len(Labels)
		likelihood = {}
		posterior = [0] * len(Labels)

		for i, label in enumerate(Labels):
			priors[i] = self.n_doc[label] / self.n_doc_total
			likelihood[label] = 0
			for word in x:
				if word not in self.stop_words:
					if(word in self.vocab[label]):
						likelihood[label] += math.log((self.vocab[label][word] + 1) / (len(self.vocab[label]) + len(Labels) + 1))
					else:
						likelihood[label] += math.log(1 / (len(self.vocab[label]) + len(Labels) + 1))
			posterior[i] = math.log(priors[i]) + likelihood[Labels(i)]
		return Labels(posterior.index(max(posterior)))



def main(train_split):
	nb = NaiveBayes()
	# ds = Dataset(train_split).fetch()
	# val_ds = Dataset('val').fetch()
	# test_ds = Dataset('test').fetch()

	train_option = input("Train the data again? (y/n)")

	ds = Dataset(train_split).fetch()
	nb.train(ds)
	if(train_option == "y"):
		evaluate(nb, ds)
		test_ds = Dataset('test').fetch()
		evaluate(nb, test_ds)


	text = "not_exit"

	while text!= "exit":
	    text = input("Enter Text to search: ")
	    if text == "exit":
	    	break
	    y = nb.predict(text).name
	    print('Result is : -  ', 'https://www.reddit.com/r/' + y)

	    open_url = input("Do you want to open result url? (y/n): ")
	    if open_url == 'y':
	    	webbrowser.open('https://www.reddit.com/r/' + y)
	    print('\n\n')


if __name__ == "__main__":
	train_split = 'train'
	main(train_split)

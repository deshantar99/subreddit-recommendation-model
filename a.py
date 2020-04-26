from data import Dataset, Labels
from utils import evaluate
import math
import os, sys
import numpy as np
# import webbrowser


stop = {'a','and','the','i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}

class NaiveBayes:
	def __init__(self):
		# total number of documents in the training set.
		self.n_doc_total = 0
		# total number of documents for each label/class in the trainin set.
		self.n_doc = {l: 0 for l in Labels}
		# frequency of words for each label in the trainng set.
		self.vocab = {l: {} for l in Labels}
		# self.vocab = {l: {i: 0 for i in l.name.split('_')} for l in Labels}

	def train(self, ds):
		for ds_val in ds:
			self.n_doc_total += 1
			self.n_doc[ds_val[1].value] +=1
			for str in ds_val[0].split():
				if str not in stop:
					str = str.lower()
					i = self.vocab.get(ds_val[1]).get(str)
					if(i != None):
						self.vocab.get(ds_val[1]).update({str: i+1})
					else:
						self.vocab.get(ds_val[1]).update({str: 1})
		print("Trainning Done...........")

		# pclearrint(ds)
		"""
		ds: list of (id, x, y) where id corresponds to document file name,
		x is a string for the email document and y is the label.
		TODO: Loop over the dataset (ds) and update self.n_doc_total,
		self.n_doc and self.vocab.
		"""

	def predict(self, x):
		my_list=[]
		for lab in self.vocab.items():
			frst = math.log((self.n_doc.get(lab[0]))/(self.n_doc_total))
			secnd = 0;
			for wrd in x.split():
				if wrd not in stop:
					wrd = wrd.lower()
					if wrd not in lab[1]:
						secnd += math.log(1/(len(lab[1])+len(self.vocab) + 1))
					else:
						secnd += math.log((1+lab[1].get(wrd))/(len(lab[1])+len(self.vocab) + 1))
			my_list.append(secnd + frst)
		# print(np.argmax(my_list))
		return Labels(np.argmax(my_list))


		"""
		x: string of words in the document.

		TODO: Use self.n_doc_total, self.n_doc and self.vocab to calculate the
		prior and likelihood probabilities.
		Add the log of prior and likelihood probabilities.
		Use MAP estimation to return the Label with hight score as
		the predicted label.
		"""


def main(train_split):
	nb = NaiveBayes()
	ds = Dataset(train_split).fetch()
	# val_ds = Dataset('val').fetch()
	test_ds = Dataset('test').fetch()
	nb.train(ds)


	print('-'*20 + ' TRAIN ' + '-'*20)
	evaluate(nb, ds)



	# print('-'*20 + ' VAL ' + '-'*20)
	# evaluate(nb, val_ds)

	print('-'*20 + ' TEST ' + '-'*20)
	evaluate(nb, test_ds)

	# text = "not_exit"

	# while text!= "exit":
	#     text = input("Enter Text to search: ")
	#     if text == "exit":
	#     	break
	#     y = nb.predict(text).name
	#     print('Result is : -  ', 'https://www.reddit.com/r/' + y)

	#     open_url = input("Do you want to open result url? (y/n): ")
	#     if open_url == 'y':
	#     	webbrowser.open('https://www.reddit.com/r/' + y)
	#     print('\n\n')






if __name__ == "__main__":
	train_split = 'train'
	main(train_split)

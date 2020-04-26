from data import Dataset, Labels
import math
import os


stop_words = {'a','and','the','i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}

def getWordFreq(emailString):
	"""
	Iterates through each word in emailString and calculates the
	word frequency.
	Input: emailString -> string
	Output: wordFreq -> dict {word : frequency}
	"""
	wordFreq = {}
	emailString = emailString.split()
	for word in emailString:
		word = word.strip().lower()
		if(word in wordFreq and word not in stop_words):
			wordFreq[word] += 1
		else:
			wordFreq[word] = 1
	return wordFreq



def updateWordFreq(wordFreq, emailString):
	"""
	Updates an existing wordFreq dict with words and their frequency
	from the new emailString.
	Input: wordFreq -> dict {word : frequency}, emailString -> string
	Output: None
	"""
	emailString = emailString.split()
	for word in emailString:
		word = word.strip().lower()
		if(word in wordFreq and word not in stop_words):
			wordFreq[word] += 1
		else:
			wordFreq[word] = 1


def print_matrix(m):
	s = [[str(m[l][ll]) for ll in Labels] for l in Labels]
	lens = [max(map(len, col)) for col in zip(*s)]
	fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
	table = [fmt.format(*row) for row in s]
	print('\n'.join(table))

def print_metrics(m):
	mlen = max([len(l.name) for l in Labels])
	for k, v in m.items():
		print('\t{}\t{}'.format(k.name+':'+' '*(mlen-len(k.name)), v))

def evaluate(model, ds):
	# confusion matrix
	cm = {l: {ll: 0 for ll in Labels} for l in Labels}
	for x, t in ds:
		y = model.predict(x)
		cm[y][t] += 1

	# print("Confusion Matrix")
	# print_matrix(cm)

	# precision, recall and f1
	sum_row  = {l: sum(cm[l].values()) for l in Labels}
	sum_col = {l: 0 for l in Labels}
	for l in Labels:
		for ll in Labels:
			sum_col[l] += cm[ll][l]

	p = {l: cm[l][l] / (sum_row[l] if sum_row[l] else 1) for l in Labels}
	r = {l: cm[l][l] / (sum_col[l] if sum_col[l] else 1) for l in Labels}
	f1 = {l: 2*p[l]*r[l]/(p[l]+r[l]) if (p[l]+r[l]) else 0 for l in Labels}

	# print("\nPrecision")
	# print_metrics(p)
	# print("\nRecall")
	# print_metrics(r)
	# print("\nF1")
	# print_metrics(f1)

	acc = sum([cm[l][l] for l in Labels]) / len(ds)
	precision = sum(p.values())/len(p.values())
	recall = sum(r.values())/len(r.values())
	fone = sum(f1.values())/len(f1.values())

	print("\nAccuracy:\t{}".format(acc))
	print("Precision:\t{}".format(precision))
	print("Recall:\t\t{}".format(recall))
	print("F1:\t\t\t{}".format(fone))

	with open("../stats.txt", "a") as stat_file:
		stat_file.write("Accuracy: " + str(acc) + "\n")
		stat_file.write("Precision: " + str(precision) + "\n")
		stat_file.write("Recall: " + str(recall) + "\n")
		stat_file.write("fone: " + str(fone) + "\n")

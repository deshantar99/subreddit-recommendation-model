import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, metrics, naive_bayes, svm
# from sklearn.metrics import accuracy_score


np.random.seed(500)

test_data_file = 'split_data/test.csv'
train_data_file = 'split_data/train.csv'

train_file = pd.read_csv(train_data_file)
test_file = pd.read_csv(test_data_file)

train_file['title'].dropna(inplace=True)
train_file['title'] = [entry.lower() for entry in train_file['title']]
train_file['title'] = [word_tokenize(entry) for entry in train_file['title']]


test_file['title'].dropna(inplace=True)
test_file['title'] = [entry.lower() for entry in test_file['title']]
test_file['title'] = [word_tokenize(entry) for entry in test_file['title']]


tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

print('----finished tokenizing------')

for index,entry in enumerate(test_file['title']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    test_file.loc[index,'title_final'] = str(Final_words)


for index,entry in enumerate(train_file['title']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    train_file.loc[index,'title_final'] = str(Final_words)



print('----finished lemmatizing')


Train_X, Test_X = train_file['title_final'], test_file['title_final']
Train_Y, Test_Y = train_file['subreddit'], test_file['subreddit']




Encoder = LabelEncoder()
Test_Y = Encoder.fit_transform(Test_Y)
Train_Y = Encoder.fit_transform(Train_Y)

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(train_file['title_final'].append(test_file['title_final']))
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)


print('---finished fitting-----')

Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf, Train_Y)

# train_NB = Naive.predict(Train_X_Tfidf)
# test_NB = Naive.predict(Test_X_Tfidf)


print('-'*5 + 'TRAIN' + '-'*5)
print('Accuracy: ', metrics.accuracy_score(train_NB, Train_Y) * 100)
print('Precision: ', metrics.precision_score(train_NB, Train_Y) * 100)
print('Recall: ', metrics.recall_score(train_NB, Train_Y) * 100)
print('F1: ', metrics.)

print("NB accuracy_score: ", metrics.accuracy_score(test_NB, Test_Y) * 100)



# SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
# SVM.fit(Train_X_Tfidf, Train_Y)
#
# predict_SVM = SVM.predict(Test_X_Tfidf)
# train_svm = SVM.predict(Train_X_Tfidf)
# print('SVM train score: ' + str(accuracy_score(train_svm, Train_Y) * 100))
# print("SVM accuracy_score: " + str(accuracy_score(predict_SVM, Test_Y) * 100))

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
import pickle

test_data_file = 'split_data/test.csv'
train_data_file = 'split_data/train.csv'

train_file = pd.read_csv(train_data_file)
test_file = pd.read_csv(test_data_file)

def tokenize_string(wordStr):
    wordStr.dropna(inplace=True)
    wordStr = [entry.lower() for entry in wordStr]
    wordStr = [word_tokenize(entry) for entry in wordStr]
    return wordStr


def lemmatize_string(wordStr, fileptr, has_file_flag):
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    finalStr=[]

    for index,entry in enumerate(wordStr):
        Final_words = []
        word_Lemmatized = WordNetLemmatizer()
        for word, tag in pos_tag(entry):
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
        if has_file_flag == 1:
            fileptr.loc[index,'title_final'] = str(Final_words)
        else:
            finalStr.append(str(Final_words))
            return finalStr

def train_model(model_type):
    train_file['title'] = tokenize_string(train_file['title'])
    test_file['title'] = tokenize_string(test_file['title'])

    lemmatize_string(train_file['title'], train_file, 1)
    lemmatize_string(test_file['title'], test_file, 1)

    Train_X, Test_X = train_file['title_final'], test_file['title_final']
    Train_Y, Test_Y = train_file['subreddit'], test_file['subreddit']

    Encoder = LabelEncoder()
    Test_Y = Encoder.fit_transform(Test_Y)
    Train_Y = Encoder.fit_transform(Train_Y)

    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(train_file['title_final'].append(test_file['title_final']))
    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)

    if(model_type == 'nb'):
        Naive = naive_bayes.MultinomialNB()
        Naive.fit(Train_X_Tfidf, Train_Y)

        with open('trained_nb.txt', 'wb') as NB_file:
            pickle.dump(Naive, NB_file)
            pickle.dump(Tfidf_vect, NB_file)
            pickle.dump(Train_X_Tfidf, NB_file)
            pickle.dump(Train_Y, NB_file)

    elif(model_type == 'svm'):
        SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
        SVM.fit(Train_X_Tfidf, Train_Y)

        predict_SVM = SVM.predict(Test_X_Tfidf)
        train_svm = SVM.predict(Train_X_Tfidf)
        print('SVM train score: ' + str(accuracy_score(train_svm, Train_Y) * 100))
        print("SVM accuracy_score: " + str(accuracy_score(predict_SVM, Test_Y) * 100))




train_choice = input('Do you want to train again? (y/n)')

if(train_choice == 'y'):

    # train_file['title'] = tokenize_string(train_file['title'])
    # test_file['title'] = tokenize_string(test_file['title'])
    #
    # print('----finished tokenizing------')
    #
    # lemmatize_string(train_file['title'], train_file, 1)
    # lemmatize_string(test_file['title'], test_file, 1)
    #
    # print('----finished lemmatizing-------')
    #
    # Train_X, Test_X = train_file['title_final'], test_file['title_final']
    # Train_Y, Test_Y = train_file['subreddit'], test_file['subreddit']
    #
    # print('-----finished setting data--------')
    #
    #
    # Encoder = LabelEncoder()
    # Test_Y = Encoder.fit_transform(Test_Y)
    # Train_Y = Encoder.fit_transform(Train_Y)
    #
    # Tfidf_vect = TfidfVectorizer(max_features=5000)
    # Tfidf_vect.fit(train_file['title_final'].append(test_file['title_final']))
    # Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    # Test_X_Tfidf = Tfidf_vect.transform(Test_X)
    #
    #
    # print('---finished fitting-----')
    #
    # Naive = naive_bayes.MultinomialNB()
    # Naive.fit(Train_X_Tfidf, Train_Y)
    #
    # with open('trained_nb.txt', 'wb') as NB_file:
    #     pickle.dump(Naive, NB_file)
    #     pickle.dump(Tfidf_vect, NB_file)
    #     pickle.dump(Train_X_Tfidf, NB_file)
    #     pickle.dump(Train_Y, NB_file)
    pass

with open('utils.txt', 'rb') as utils_file:
    Tfidf_vect = pickle.load(utils_file)
    Train_X_Tfidf = pickle.load(utils_file)

with open('trained_nb.txt', 'rb') as NB_file:
    Naive = pickle.load(NB_file)
    # Tfidf_vect = pickle.load(NB_file)
    # Train_X_Tfidf = pickle.load(NB_file)
    # Train_Y = pickle.load(NB_file)


Encoder = LabelEncoder()
Train_Y = train_file['subreddit']
Train_Y = Encoder.fit_transform(Train_Y)

# train_NB = Naive.predict(Train_X_Tfidf)
# # test_NB = Naive.predict(Test_X_Tfidf)
# #
# print('-'*5 + 'TRAIN' + '-'*5)
# print('Accuracy: ', metrics.accuracy_score(train_NB, Train_Y) * 100)
# print('Precision: ', metrics.precision_score(train_NB, Train_Y, average='micro') * 100)
# print('Recall: ', metrics.recall_score(train_NB, Train_Y, average='micro') * 100)
# print('F1: ', metrics.f1_score(train_NB, Train_Y, average='micro') * 100)
# #
# print('-'*5 + 'TEST' + '-'*5)
# print('Accuracy: ', metrics.accuracy_score(test_NB, Test_Y) * 100)
# print('Precision: ', metrics.precision_score(test_NB, Test_Y, average='micro') * 100)
# print('Recall: ', metrics.recall_score(test_NB, Test_Y, average='micro') * 100)
# print('F1: ', metrics.f1_score(test_NB, Test_Y, average='micro') * 100)

query = ''
while(query != 'exit'):
    query = input("Enter your query('exit' to end): ")
    if(query == 'exit'): break
    query = tokenize_string(pd.Series(query))
    query = lemmatize_string(query, None, 0)

    query_Tfidf = Tfidf_vect.transform(query)
    predicted = Naive.predict(query_Tfidf)
    print(Encoder.inverse_transform(predicted))

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
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
import pickle

test_data_file = 'split_data/test.csv'
train_data_file = 'split_data/train.csv'

train_file = pd.read_csv(train_data_file)
test_file = pd.read_csv(test_data_file)

def tokenize_string(wordStr):
    #tokenizes all of the post titles in the wordStr array
    #input: post['title'] for a subreddit data file
    #output: tokenized post['title'] for a subreddit data file
    wordStr.dropna(inplace=True)
    wordStr = [entry.lower() for entry in wordStr]
    wordStr = [word_tokenize(entry) for entry in wordStr]
    return wordStr


def lemmatize_string(wordStr, fileptr, has_file_flag):
    #lemmatizes all of the words in post titles
    #input: wordStr: array of post['title'], fileptr: pointer to the datafile,
    #input: has_file_flag: flag to decide whether to edit the file or return (cont.)
    #input: an array of changes

    #output: append lemmatized string to 'title_final' column of file if has_file_flag is 1
    #output: return lemmatized string if has_file_flag is 0, needed when predict user queries

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
    #trains the model on the train data; model defined by model_type
    #model_type: nb, svm, knn, rocchio

    #tokenize train and test titles
    train_file['title'] = tokenize_string(train_file['title'])
    test_file['title'] = tokenize_string(test_file['title'])

    #lemmatize train and test titles
    lemmatize_string(train_file['title'], train_file, 1)
    lemmatize_string(test_file['title'], test_file, 1)

    #initialize train and test data
    Train_X, Test_X = train_file['title_final'], test_file['title_final']
    Train_Y, Test_Y = train_file['subreddit'], test_file['subreddit']

    #Encode the labels to a number
    Encoder = LabelEncoder()
    Test_Y = Encoder.fit_transform(Test_Y)
    Train_Y = Encoder.fit_transform(Train_Y)

    #term frequency-inverse doc frequency
    Tfidf_vect = TfidfVectorizer(max_features=5000) #assuming max # of words = 5000
    Tfidf_vect.fit(train_file['title_final'].append(test_file['title_final']))
    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)

    #save trained utility values for later use
    with open('local/utils.txt', 'wb') as utils_file:
        pickle.dump(Tfidf_vect, utils_file)
        pickle.dump(Train_X_Tfidf, utils_file)
        pickle.dump(Test_X_Tfidf, utils_file)

    if(model_type == 'nb'):
        NB = naive_bayes.MultinomialNB()
        NB.fit(Train_X_Tfidf, Train_Y)
        train_NB = NB.predict(Train_X_Tfidf)
        print('Naive Bayes Train Accuracy: ', metrics.accuracy_score(train_NB, Train_Y) * 100)
        test_NB = NB.predict(Test_X_Tfidf)
        print('Naive Bayes Test Accuracy: ', metrics.accuracy_score(test_NB, Test_Y) * 100)

        with open('local/trained_nb.txt', 'wb') as NB_file:
            pickle.dump(NB, NB_file)

    elif(model_type == 'svm'):
        SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
        SVM.fit(Train_X_Tfidf, Train_Y)

        train_svm = SVM.predict(Train_X_Tfidf)
        print('SVM Train Accuracy: ' + str(metrics.accuracy_score(train_svm, Train_Y) * 100))
        test_SVM = SVM.predict(Test_X_Tfidf)
        print("SVM Test score: " + str(metrics.accuracy_score(test_SVM, Test_Y) * 100))

        with open('trained_svm.txt', 'wb') as SVM_file:
            pickle.dump(SVM, SVM_file)

    elif(model_type == 'knn'):
        KNN = KNeighborsClassifier(n_neighbors = 3)
        KNN.fit(Train_X_Tfidf, Train_Y)
        train_KNN = KNN.predict(Train_X_Tfidf)
        print('KNN Train Accuracy: ' + str(metrics.accuracy_score(train_KNN, Train_Y) * 100))
        test_KNN = KNN.predict(Test_X_Tfidf)
        print('KNN Test Accuracy: ' + str(metrics.accuracy_score(test_KNN, Test_Y) * 100))

        with open('local/trained_knn.txt', 'wb') as KNN_file:
            pickle.dump(KNN, KNN_file)

    elif(model_type == 'rocchio'):
        Rocchio = NearestCentroid()
        Rocchio.fit(Train_X_Tfidf, Train_Y)
        print('fitted')
        train_Rocchio = Rocchio.predict(Train_X_Tfidf)
        print('Rocchio Train Accuracy: ' + str(metrics.accuracy_score(train_Rocchio, Train_Y) * 100))
        test_Rocchio = Rocchio.predict(Test_X_Tfidf)
        print('Rocchio Test Accuracy: ' + str(metrics.accuracy_score(test_Rocchio, Test_Y) * 100))

        with open('local/trained_rocchio.txt', 'wb') as Rocchio_file:
            pickle.dump(Rocchio, Rocchio_file)


train_choice = input('Do you want to train again? (y/n): (TAKES A LONG TIME)')

#takes a very long time, train each model individually
#train one, comment the rest
#each time a model is trained again, it is locally saved
#
if(train_choice == 'y'):
    train_model('nb')
    train_model('svm')
    train_model('knn')
    train_model('rocchio')


with open('local/utils.txt', 'rb') as utils_file:
    Tfidf_vect = pickle.load(utils_file)
    Train_X_Tfidf = pickle.load(utils_file)
    Test_X_Tfidf = pickle.load(utils_file)

with open('local/trained_nb.txt', 'rb') as NB_file:
    NB = pickle.load(NB_file)

with open('local/trained_svm.txt', 'rb') as SVM_file:
    SVM = pickle.load(SVM_file)

with open('local/trained_knn.txt', 'rb') as KNN_file:
    KNN = pickle.load(KNN_file)

with open('local/trained_rocchio.txt', 'rb') as Rocchio_file:
    Rocchio = pickle.load(Rocchio_file)


Encoder = LabelEncoder()
Train_Y = train_file['subreddit']
Train_Y = Encoder.fit_transform(Train_Y)

query = ''
while(query != 'exit'):
    query = input("Enter your query('exit' to end): ")
    if(query == 'exit'): break
    query = tokenize_string(pd.Series(query))
    query = lemmatize_string(query, None, 0)

    query_Tfidf = Tfidf_vect.transform(query)
    Rocchio_predict = Rocchio.predict(query_Tfidf)
    KNN_predict = KNN.predict(query_Tfidf)
    SVM_predict = SVM.predict(query_Tfidf)
    NB_predict = NB.predict(query_Tfidf)

    print('NB: ' + str(Encoder.inverse_transform(NB_predict)))
    print('SVM: ' + str(Encoder.inverse_transform(SVM_predict)))
    print('KNN: ' + str(Encoder.inverse_transform(KNN_predict)))
    print('Rocchio: ' + str(Encoder.inverse_transform(Rocchio_predict)))

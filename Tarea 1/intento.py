import pandas as pd
import shutil

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

import os
import numpy as np

# NEW
from nltk.tokenize import TweetTokenizer
from nltk.sentiment.util import mark_negation
from common_functions import *
from nltk.corpus import stopwords

train = {
    'anger': pd.read_csv('https://raw.githubusercontent.com/dccuchile/CC6205/master/assignments/assignment_1/data/train/anger-train.txt', sep='\t', names=['id', 'tweet', 'class', 'sentiment_intensity']),
    'fear': pd.read_csv('https://raw.githubusercontent.com/dccuchile/CC6205/master/assignments/assignment_1/data/train/fear-train.txt', sep='\t', names=['id', 'tweet', 'class', 'sentiment_intensity']),
    'joy': pd.read_csv('https://raw.githubusercontent.com/dccuchile/CC6205/master/assignments/assignment_1/data/train/joy-train.txt', sep='\t', names=['id', 'tweet', 'class', 'sentiment_intensity']),
    'sadness': pd.read_csv('https://raw.githubusercontent.com/dccuchile/CC6205/master/assignments/assignment_1/data/train/sadness-train.txt', sep='\t', names=['id', 'tweet', 'class', 'sentiment_intensity'])
}

def get_group_dist(group_name, train):
    print(group_name, "\n",
        train[group_name].groupby('sentiment_intensity').count())

def print_qtweets(dataset):
    for key in dataset:
        get_group_dist(key, dataset)

#print_qtweets(train)


# CLASIFICADOR y VECTORIZADOR______________________________________________________________

# Inicializamos el Vectorizador para transformar las oraciones a BoW 

# Esto hace los tokens para un tweet (considera emojis y caritas)
# y ademas añade un negado a todo lo que venga después de una negación
def superTokenize(text):
    tokens = TweetTokenizer().tokenize(text)
    tokens = mark_negation(tokens)
    return tokens

vectorizer = CountVectorizer(tokenizer=superTokenize, ngram_range=(1,3), stop_words=stopwords.words('english'))

superTokens = []
for key in train:
    X_train, X_test, y_train, y_test = split_dataset(train[key])
    #print(X_train)
    #print("\n \n")
    tokens = []    
    #for text in y_train:
        #tokens.append(tokenizer.tokenize(text))
        #print(text)
    print(vectorizer.fit(X_train, y_train))

    superTokens.append(tokens)





    
# Inicializamos el Clasificador.
naive_bayes = MultinomialNB()

# Establecer el pipeline.
#text_clf = Pipeline([('token', tokenizer), ('clf', naive_bayes)])
#print(Pipeline)



# # Ejecutar el clasificador por cada dataset

# classifiers = []
# learned_labels_array = []

# # Por cada llave en train ('anger', 'fear', 'joy', 'sadness')
# for key in train:
#     classifier, learned_labels = classify(train[key], key)
#     classifiers.append(classifier)
#     learned_labels_array.append(learned_labels)



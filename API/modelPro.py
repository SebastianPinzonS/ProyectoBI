import numpy as np
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay,
    roc_auc_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
import gensim
from gensim.models import Word2Vec

def predict(datas:string, modelw, tfidf_model):
    dataf = [datas]
    data = pd.DataFrame(dataf, columns=['text'])
    datach = data['text'].apply(lambda x: finalpreprocess(x))
    X_tok= [nltk.word_tokenize(i) for i in datach]
    X_vectors_w2v = modelw.transform(X_tok)
    predictions = tfidf_model.predict(X_vectors_w2v)
    return predictions

# This is a helper function to map NTLK position tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Tokenize the sentence
def lemmatizer(string):
    wl = WordNetLemmatizer()
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(a)

def stopword(string):
    stop_words = stopwords.words("spanish")
    a= [i for i in string.split() if i not in stop_words]
    return ' '.join(a)
    

def finalpreprocess(string):
    return lemmatizer(stopword(preprocess(string)))

def preprocess(text):
    text = text.lower()
    text=text.strip()
    text=re.compile('<.*?>').sub('', text)
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]',' ',text)
    text=re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text)
    text = re.sub(r'\s+',' ',text)
    return text

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.values())))
    def fit(self, X, y):
        return self
    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

def getModel():
    train_df = pd.read_excel("./data/cat_6716.xlsx")
    train_df["Textos_espanol"]= train_df["Textos_espanol"].astype(str)
    train_df["Textos_espanol"]= train_df["Textos_espanol"].astype(str)
    train_df['clean_text'] = train_df['Textos_espanol'].apply(lambda x: finalpreprocess(x))
    train_df['clean_text_tok']=[nltk.word_tokenize(i) for i in train_df['clean_text']]
    model = Word2Vec(train_df['clean_text_tok'],min_count=1)
    w2v = dict(zip(model.wv.index_to_key, model.wv.vectors))
    modelw = MeanEmbeddingVectorizer(w2v)
    return modelw

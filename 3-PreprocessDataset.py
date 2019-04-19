# Get the dataset of news headlines

import numpy as np
np.random.seed(2018)

import pandas as pd

f = "./pickle/filtered_headlines_data.pkl"
documents = pd.read_pickle(f)

# Pre-process the text before the Topic Model can be trained

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

stemmer = SnowballStemmer('english')

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

processed_docs = documents['text'].map(preprocess)

documents.to_pickle("./pickle/documents.pkl")
processed_docs.to_pickle("./pickle/processed_docs.pkl")
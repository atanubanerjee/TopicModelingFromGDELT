import pandas as pd
import gensim
import pickle

documents = pd.read_pickle("./pickle/documents.pkl")
processed_docs = pd.read_pickle("./pickle/processed_docs.pkl")

dictionary = gensim.corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

from gensim import corpora, models

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

with open('./pickle/bow_corpus.pkl', 'wb') as f:
    pickle.dump(bow_corpus, f)

with open('./pickle/corpus_tfidf.pkl', 'wb') as f:
    pickle.dump(corpus_tfidf, f)

with open('./pickle/dictionary.pkl', 'wb') as f:
    pickle.dump(dictionary, f)

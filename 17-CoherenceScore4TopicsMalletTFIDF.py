import pickle
import gensim
from gensim import corpora, models
from multiprocessing import freeze_support

with open('./pickle/dictionary.pkl', 'rb') as f:
    dictionary = pickle.load(f)

import pandas as pd
processed_docs = pd.read_pickle("./pickle/processed_docs.pkl")
from gensim.models.coherencemodel import CoherenceModel

lda_model_4topics_mallet_tfidf = gensim.models.wrappers.LdaMallet.load('./pickle/lda_model_4topics_mallet_tfidf')

# Compute Perplexity
# print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
if __name__ == '__main__':
    freeze_support()
    coherence_model_lda_mallet_tfidf = CoherenceModel(model=lda_model_4topics_mallet_tfidf, texts=processed_docs, dictionary=dictionary, coherence='c_v')
    coherence_lda_mallet_tfidf = coherence_model_lda_mallet_tfidf.get_coherence()
    print('\nCoherence Score: ', coherence_lda_mallet_tfidf)

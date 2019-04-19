import pickle
import gensim
from gensim import corpora, models
from multiprocessing import freeze_support

with open('./pickle/dictionary.pkl', 'rb') as f:
    dictionary = pickle.load(f)

import pandas as pd
processed_docs = pd.read_pickle("./pickle/processed_docs.pkl")
from gensim.models.coherencemodel import CoherenceModel

lda_model_mallet = gensim.models.wrappers.LdaMallet.load('./pickle/lda_model_mallet')

# Compute Perplexity
# print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
if __name__ == '__main__':
    freeze_support()
    coherence_model_lda_mallet = CoherenceModel(model=lda_model_mallet, texts=processed_docs, dictionary=dictionary, coherence='c_v')
    coherence_lda_mallet = coherence_model_lda_mallet.get_coherence()
    print('\nCoherence Score: ', coherence_lda_mallet)


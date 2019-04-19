import pickle
import gensim
from gensim import corpora, models
from multiprocessing import freeze_support

with open('./pickle/corpus_tfidf.pkl', 'rb') as f:
    corpus_tfidf = pickle.load(f)

with open('./pickle/dictionary.pkl', 'rb') as f:
    dictionary = pickle.load(f)

# if __name__ == '__main__':
#    freeze_support()
#    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)

lda_model_tfidf = gensim.models.LdaModel(corpus_tfidf, num_topics=10, iterations=50, id2word=dictionary)

lda_model_tfidf.save('./pickle/lda_model_tfidf')

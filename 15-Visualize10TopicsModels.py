# Visualize the topics
import pyLDAvis
import pyLDAvis.gensim
import gensim
import pickle

with open('./pickle/bow_corpus.pkl', 'rb') as f:
    bow_corpus = pickle.load(f)

with open('./pickle/corpus_tfidf.pkl', 'rb') as f:
    corpus_tfidf = pickle.load(f)

with open('./pickle/dictionary.pkl', 'rb') as f:
    dictionary = pickle.load(f)

lda_model = gensim.models.LdaModel.load('./pickle/lda_model')
vis_10topics = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary)
pyLDAvis.save_html(vis_10topics, './DataViz/Viz10Topics.html')

lda_model_tfidf = gensim.models.LdaModel.load('./pickle/lda_model_tfidf')
vis_10topics_tfidf = pyLDAvis.gensim.prepare(lda_model_tfidf, corpus_tfidf, dictionary)
pyLDAvis.save_html(vis_10topics_tfidf, './DataViz/Viz10TopicsTFIDF.html')

lda_model_mallet_tfidf = gensim.models.wrappers.LdaMallet.load('./pickle/lda_model_mallet_tfidf')
lda_mallet2gensim = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(lda_model_mallet_tfidf, iterations=1000)
vis_mallet_tfidf = pyLDAvis.gensim.prepare(lda_mallet2gensim, corpus_tfidf, dictionary)
pyLDAvis.save_html(vis_mallet_tfidf, './DataViz/Viz10TopicsMalletTFIDF.html')
import pickle
import numpy
import gensim

from gensim.models.coherencemodel import CoherenceModel
from matplotlib import pyplot as plt

import os
os.environ.update({'MALLET_HOME':r'C:/mallet-2.0.8/'})
mallet_path = 'C:/mallet-2.0.8/bin/mallet'

if __name__ == '__main__': 

    with open('./pickle/corpus_tfidf.pkl', 'rb') as f:
        corpus_tfidf = pickle.load(f)

    with open('./pickle/dictionary.pkl', 'rb') as f:
        dictionary = pickle.load(f)

    import pandas as pd
    processed_docs = pd.read_pickle("./pickle/processed_docs.pkl")


    def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
        """
        Compute c_v coherence for various number of topics

        Parameters:
        ----------
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts
        limit : Max num of topics

        Returns:
        -------
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """
        coherence_values = []
        model_list = []

        mallet_path = 'C:/mallet-2.0.8/bin/mallet'

        for num_topics in range(start, limit, step):
            model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_score = coherencemodel.get_coherence()
            coherence_values.append(coherence_score)
            print("Num Topics = ", num_topics, " has Coherence Value of ", coherence_score)

        return model_list, coherence_values

    # Can take a long time to run.
    model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus_tfidf, texts=processed_docs, start=2, limit=20, step=2)

    with open('./pickle/coherence_scores.pkl', 'wb') as f:
        pickle.dump(coherence_values, f)

    # Show graph
    limit=20; start=2; step=2
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.savefig('./DataViz/ElbowChartCoherenceScores.png')
    plt.show()
import os
from gensim.models.wrappers import LdaMallet
os.environ.update({'MALLET_HOME':r'C:/mallet-2.0.8/'})
# os.environ.update({'PATH':r'C:\Program Files (x86)\Common Files\Oracle\Java\javapath;C:\WINDOWS\system32;C:\WINDOWS;C:\WINDOWS\System32\Wbem;C:\WINDOWS\System32\WindowsPowerShell\v1.0\;C:\WINDOWS\System32\OpenSSH\;C:\Program Files\Git\cmd;C:\Users\atanu\AppData\Local\Microsoft\WindowsApps;;C:\Program Files\Microsoft VS Code\bin'})
mallet_path = 'C:/mallet-2.0.8/bin/mallet'

import pickle
import gensim
from gensim import corpora, models
from multiprocessing import freeze_support

with open('./pickle/bow_corpus.pkl', 'rb') as f:
    bow_corpus = pickle.load(f)

with open('./pickle/dictionary.pkl', 'rb') as f:
    dictionary = pickle.load(f)

lda_model_mallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=bow_corpus, num_topics=10, id2word=dictionary)

lda_model_mallet.save('./pickle/lda_model_mallet')

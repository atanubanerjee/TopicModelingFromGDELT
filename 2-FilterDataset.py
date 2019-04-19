import pickle

import numpy as np
np.random.seed(2019)

import pandas as pd

f = "./pickle/corpus_2019-04-17_21-54-32.pkl"
headlines_data = pd.read_pickle(f)
filtered_headlines_data = headlines_data[~headlines_data['text'].isin(['Connection Error','HTTP Error'])].copy()
filtered_headlines_data['index'] = filtered_headlines_data.index

f = "./pickle/filtered_headlines_data.pkl"
filtered_headlines_data.to_pickle(f)

import pickle
import gensim
from gensim import corpora, models
from multiprocessing import freeze_support

lda_model_tfidf = gensim.models.LdaModel.load('./pickle/lda_model_tfidf')

# Wordcloud of Top N words in each topic
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

cols2 = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud2 = WordCloud(stopwords=gensim.parsing.preprocessing.STOPWORDS,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols2[i],
                  prefer_horizontal=1.0)

topics2 = lda_model_tfidf.show_topics(formatted=False)

fig2, axes2 = plt.subplots(5, 2, figsize=(10,10), sharex=True, sharey=True)

for i, ax in enumerate(axes2.flatten()):
    fig2.add_subplot(ax)
    topic_words = dict(topics2[i][1])
    cloud2.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud2)
    plt.gca().set_title('Topic ' + str(i+1), fontdict=dict(size=16))
    plt.gca().axis('off')

plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.savefig('./DataViz/Wordcloud10TopicsTFIDF.png')
plt.show()
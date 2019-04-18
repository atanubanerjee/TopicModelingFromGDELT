import numpy as np
np.random.seed(2019)

import pandas as pd
import httplib2
from bs4 import BeautifulSoup

http = httplib2.Http(timeout=10)

def get_text(url):
    try:
        headers, body = http.request(url)
    except:
        return "Connection Error"

    if headers.status != 200:
        # HTTP Error
        return "HTTP Error"

    soup = BeautifulSoup(body, 'html.parser')

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    import html2text
    h = html2text.HTML2Text()
    text2 = h.handle(text)
    return text2

def get_text_simple(url):
    try:
        headers, body = http.request(url)
    except:
        return "Connection Error"

    if headers.status != 200:
        # HTTP Error
        return "HTTP Error"

    import html2text
    h = html2text.HTML2Text()
    text2 = h.handle(str(body, encoding = "utf-8"))
    return text2

headlines_data = pd.read_csv('data/20190409044732.7114.events.csv', error_bad_lines=False)

series_length = len(headlines_data)
#series_length = 5
corpus = []

for index in range(series_length):
    print(">>>> Retrieving: " + str(index) + "/" + str(series_length))
    html_doc = headlines_data.loc[index]['SOURCEURL']
    text = get_text(html_doc)
    corpus.append(text)
    # print(text)

hd = headlines_data[0:series_length].copy()
hd['text'] = corpus

import datetime
dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
f = "./pickle/corpus_" + dt + ".pkl"
hd.to_pickle(f)

import numpy as np
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
import spacy
#### hacer referencias

from src.features.utils import remove_stopwords, sent_to_words, make_bigrams, make_trigrams, lemmatization
from src.data.prepare_data import read_sample
#######finalizan referencias

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# Obtener datos
data = read_sample()

data_words = list(sent_to_words(data))

# Construimos modelos de bigrams y trigrams
# https://radimrehurek.com/gensim/models/phrases.html#gensim.models.phrases.Phrases
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

# Aplicamos el conjunto de bigrams/trigrams a nuestros documentos
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# Eliminar stopwords
#def remove_stopwords(texts):
 #   return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


# Eliminamos stopwords
data_words_nostops = remove_stopwords(data_words)

# Formamos bigrams
#data_words_bigrams = make_bigrams(data_words_nostops)
data_words_bigrams = make_bigrams(bigram_mod, data_words_nostops)

# python3 -m spacy download en_core_web_lg

# Lematizamos preservando únicamente noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

# Creamos diccionario
id2word = corpora.Dictionary(data_lemmatized)

for key, value in id2word.items():
    print(key, value)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]


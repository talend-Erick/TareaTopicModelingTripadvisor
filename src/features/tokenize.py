import gensim
from gensim.models import CoherenceModel
from src.features.utils import remove_stopwords, sent_to_words, make_bigrams, make_trigrams, lemmatization

def tokenize_classes(data_words: list) -> list:
# Construimos modelos de bigrams y trigrams
# https://radimrehurek.com/gensim/models/phrases.html#gensim.models.phrases.Phrases
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

# Aplicamos el conjunto de bigrams/trigrams a nuestros documentos
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)


# Eliminamos stopwords
    data_words_nostops = remove_stopwords(data_words)

# Formamos bigrams
#data_words_bigrams = make_bigrams(data_words_nostops)
    data_words_bigrams = make_bigrams(bigram_mod, data_words_nostops)

# python3 -m spacy download en_core_web_lg

# Lematizamos preservando únicamente noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    return data_lemmatized
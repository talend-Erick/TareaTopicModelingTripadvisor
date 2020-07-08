import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from nltk.corpus import stopwords

#### hacer referencias

from src.features.utils import  sent_to_words
from src.features.tokenize import tokenize_classes
from src.data.prepare_data import read_sample
#######finalizan referencias

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# Obtener datos
data = read_sample()

data_words = list(sent_to_words(data))

#agregar tokenize
data_lemmatized = tokenize_classes(data_words)
# Creamos diccionario
id2word = corpora.Dictionary(data_lemmatized)

#for key, value in id2word.items():
 #   print(key, value)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
#muestra el id y la frecuencia
#[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]

################## MODELO ###################
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

# Muestra los topics
#pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

#Perplejidad
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Score de coherencia
#coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
#coherence_lda = coherence_model_lda.get_coherence()
#print('\nCoherence Score: ', coherence_lda)

# Visualizamos los temas
#pyLDAvis.enable_notebook()
#vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
#vis
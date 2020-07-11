import pickle

import pyLDAvis
import pyLDAvis.gensim

# MODELO
with open('../../models/model.pkl', 'rb') as input_file:
    lda_model = pickle.load(input_file)

# DICCIONARIO
with open('../../models/id2word.pkl', 'rb') as input_file:
    id2word = pickle.load(input_file)

    # CORPUS
with open('../../models/corpus.pkl', 'rb') as input_file:
    corpus = pickle.load(input_file)

vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
pyLDAvis.show(vis)
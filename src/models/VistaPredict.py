import pickle

import pyLDAvis
import pyLDAvis.gensim

# MODELO
with open('../../models/model.pkl', 'rb') as input_file:
    model = pickle.load(input_file)

# DICCIONARIO
with open('../../models/id2word.pkl', 'rb') as input_file:
    id2word = pickle.load(input_file)

    # new_CORPUS
with open('../../models/new_corpus.pkl', 'rb') as input_file:
    new_corpus = pickle.load(input_file)

vis = pyLDAvis.gensim.prepare(model, new_corpus, id2word)
pyLDAvis.show(vis)
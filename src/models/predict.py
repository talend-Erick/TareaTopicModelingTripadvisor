import pickle
from typing import List

import gensim.corpora as corpora

def predict(document: List[List[str]]):

    with open('../../models/model.pkl', 'rb') as input_file:
        lda_model = pickle.load(input_file)
    #abro el diccionario de datos que genere en el entrenamiento
    with open('../../models/id2word.pkl', 'rb') as input_file:
        id2word = pickle.load(input_file)
    # leo cada registro de la lista ingresada y la comparo con el diccionario, para obtener el nuevo corpus,
    #que sería los id encontrados con su frecuencia según el pkl generado por el entrenamiento
    new_corpus = [id2word.doc2bow(text) for text in document]

    with open('../../models/new_corpus.pkl', 'wb') as output_file:
        pickle.dump(new_corpus, output_file)

    prediction =[]
    for doc in new_corpus:
        result = lda_model[doc]
      #  print (result)
        prediction.append(result)

    new_dic = corpora.Dictionary(document)
    #print(new_dic)
    print(prediction)
    return prediction

text_input = [
    ['hotel', 'bad', 'survey', 'room', 'graph'],
    ['cup', 'door', 'eps'],
    ['friendly', 'beach', 'holidays']]

predict(text_input)
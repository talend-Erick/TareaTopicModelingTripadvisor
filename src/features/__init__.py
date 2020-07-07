import spacy

nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])

# Inicializamos el modelo 'en_core_web_lg' con las componentes de POS únicamente
#nlp = spacy.load('C:\\Users\\ergarciap\\Miniconda3\\Lib\\site-packages\\en_core_web_lg\\en_core_web_lg-2.3.0', disable=['parser', 'ner'])

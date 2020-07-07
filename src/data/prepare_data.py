import re
import pandas as pd
from pandas import DataFrame

def read_sample() -> DataFrame:
    df = pd.read_json('../../data/raw/newsgroups.json')
    data = df.content.values.tolist()

    # Eliminar emailsTodo lo que este antes y despues del arroba
    data = [re.sub(r'\S*@\S*\s?', '', sent) for sent in data]

    # Eliminar newlines
    data = [re.sub(r'\s+', ' ', sent) for sent in data]

    # Eliminar comillas
    data = [re.sub(r"\'", "", sent) for sent in data]

    return data
"""
datasets.py
"""
import numpy as np
import pandas as pd
from collections import namedtuple
from gensim.corpora import Dictionary


# Load two dictionary
dct = Dictionary.load_from_text("vocab.txt")
def doc2bow(morphemes):
    """ Converrt strings with non filtered dictionary to vector
    :param morphemes: morphemes
    :return: BOW vector
    """
    global dct
    # We can obtain all of indices and frequency of a morpheme
    # [(ID, frequency)]
    bow_format = dct.doc2bow(morphemes.split())
    return bow_format

def load_dataset():
    """ Load dataset
    :return: data and label
    """
    global dct
    # Load training dataset
    df = pd.read_csv("livedoor_news.csv")
    Dataset = namedtuple("Dataset", ["news", "data", "target", "target_names", "dct"])
    news = [doc for doc in df["news"]]
    data = [doc2bow(doc) for doc in df["news"]]
    target = np.array([label for label in df["class"]]).astype(np.int64)
    target_names = ["dokujo-tsushin","it-life-hack","kaden-channel","livedoor-homme",
                    "movie-enter","peachy","smax","sports-watch","topic-news"]
    ldcc_dataset = Dataset(news, data, target, target_names, dct)
    return ldcc_dataset

"""
make_dic.py
1. Read csv file
2. Make dictionary
3. Update dictionary
4. Save dictionary into a txt file
"""
import pandas as pd
from gensim.corpora import Dictionary

# Read csv file
df = pd.read_csv("livedoor_news.csv")

# 辞書
dct = Dictionary()
for i, news in enumerate(df["news"]):
    # Update dictionary with new documents
    dct.add_documents([news.split()])

dct.save_as_text("vocab.txt")

"""
make_csv.py
1. Read txt files.
2. Extract only noun in sentence.
3. Write extracted data into csv file.
"""
import codecs
import glob
import MeCab
import pandas as pd


tagger = MeCab.Tagger("-Ochasen")

def get_morphemes(fpath):
    """ Get morphemes
    :param fpath: file path
    :return: result of morphological analysis, -1 indicates error
    """
    data = read_file(fpath)
    morphemes = tokenzier(data)
    return morphemes if morphemes else -1

def read_file(fpath):
    """ Read file
    :param fpath: file path
    :return: file content of live door news corpus
    """
    with codecs.open(fpath, 'r', 'utf-8') as f:
        return "\n".join(f.read().splitlines()[2:])

def tokenzier(sentences):
    """ Morphological analysis
    :param sentences: strings in the article
    :return: morphemes
    """
    tag = tagger.parseToNode(sentences)
    morphemes = []
    while tag:
        features = tag.feature.split(",")
        if features[0] == "名詞":
            morphemes.append(tag.surface.lower())
        tag = tag.next
    return morphemes


path = "text/"
ldcc = ["dokujo-tsushin","it-life-hack","kaden-channel","livedoor-homme",
        "movie-enter","peachy","smax","sports-watch","topic-news"]
ldcc2id = {v: k for k, v in enumerate(ldcc)}
df = pd.DataFrame(columns=["class", "news"])
for d, i in ldcc2id.items():
    flist = glob.glob(path + d + "/*.txt")
    flist.remove(path + d + "\\LICENSE.txt")
    for fpath in flist:
        print(fpath)
        morphemes = get_morphemes(fpath)
        if morphemes == -1:
            continue
        temp = pd.Series([i, " ".join(morphemes)], index=df.columns)
        df = df.append(temp, ignore_index=True)
df.to_csv("livedoor_news.csv", index=False, encoding="utf-8")

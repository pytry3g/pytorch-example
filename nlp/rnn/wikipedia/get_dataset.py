# -*- coding: utf-8 -*-
import MeCab
import codecs
import re
import urllib.parse as par
import urllib.request as req


def write2file(fname, sentences):
    with codecs.open(fname, 'w', 'utf-8') as f:
        f.write("".join(sentences))

def get_morphemes(sentences):
    morphemes = []
    for sent in sentences:
        if len(sent) == 0:
            continue
        temp = tagger.parse(sent).split()
        temp.append("。\n")
        morphemes.append(" ".join(temp))
    return morphemes if morphemes else -1


tagger = MeCab.Tagger("-Owakati")
link = "https://ja.wikipedia.org/wiki/"
fname_list = ["nobunaga", "hideyoshi", "ieyasu"]
word_list = ["織田信長", "豊臣秀吉", "徳川家康"]
for fname, word in zip(fname_list, word_list):
    with req.urlopen(link + par.quote_plus(word)) as response:
        html = response.read().decode('utf-8')
        # <p>タグを取得
        all_p_tag = re.findall("<p>.*</p>", html)
        temp = []
        for p in all_p_tag:
            # 半角文字を削除
            p = re.sub("[\s!-~]*", "", p)
            p = p.split("。")
            # 分かち書き
            morphemes = get_morphemes(p)
            if morphemes == -1:
                continue
            temp = temp + morphemes
        write2file(fname + ".txt", temp)

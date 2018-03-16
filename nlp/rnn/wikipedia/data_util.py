# -*- coding: utf-8 -*-
import codecs


def read_file(fname):
    """ Read file
    :param fname: file name
    :return: word list in the file
    """
    with codecs.open(fname + ".txt", 'r', 'utf-8') as f:
        return f.read().splitlines()

def select_sentences(sentences):
    dataset = []
    for sent in sentences:
        morphemes = sent.split()
        if len(morphemes) > 30:
            continue
        for i in range(len(morphemes)-2):
            if morphemes[i] == morphemes[i+1]:
                break
            if morphemes[i] == morphemes[i+2]:
                break
        else:
            dataset.append(sent)
    return dataset

def make_vocab(sentences):
    """ make dictionary
    :param sentences: word list ex. ["I", "am", "stupid"]
    """
    global word2id

    for sent in sentences:
        for morpheme in sent.split():
            if morpheme in word2id:
                continue
            word2id[morpheme] = len(word2id)

def sent2id(sentences):
    id_list = []
    for sent in sentences:
        temp = []
        for morpheme in sent.split():
            temp.append(word2id[morpheme])
        id_list.append(temp)
    return id_list

def get_dataset():
    fname_list = ["nobunaga", "hideyoshi", "ieyasu"]
    dataset = []
    # make dictionary
    for fname in fname_list:
        sentences = read_file(fname)
        sentences = select_sentences(sentences)
        make_vocab(sentences)
        dataset = dataset + sentences
    id2sent = sent2id(dataset)
    return word2id, id2sent, dataset


word2id = {}

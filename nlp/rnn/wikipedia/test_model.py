import random
import data_util as U
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as O
from torch.autograd import Variable


### Prepare data ###
word2id, id2sent, dataset = U.get_dataset()

### Build the model ###
USE_CUDA = False
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
class RNN(nn.Module):
    def __init__(self, n_vocab, embedding_dim, hidden_dim):
        super(RNN, self).__init__()
        self.encoder = nn.Embedding(n_vocab, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, n_vocab)
        self.embedding_dim = embedding_dim
        self.num_layers = 1
        self.dropout = nn.Dropout(0.1)
        self.init_hidden()

    def init_hidden(self):
        self.hidden = Variable(\
                        FloatTensor(self.num_layers, 1, self.embedding_dim).fill_(0))
        if USE_CUDA:
            self.hidden.cuda()

    def forward(self, x):
        x = self.encoder(x.view(1, -1))
        x = self.dropout(x)
        y, self.hidden = self.gru(x.view(1, 1, -1), self.hidden)
        y = self.decoder(y.view(1, -1))
        return y

def variable(index):
    tensor = torch.LongTensor(index)
    if USE_CUDA:
        return Variable(tensor).cuda()
    return Variable(tensor)



n_vocab = len(word2id)
embedding_dim = 128
hidden_dim = 128
learning_rate = 0.01
model = RNN(n_vocab, embedding_dim, hidden_dim)
if USE_CUDA:
    model.cuda()
id2word = {v: k for k, v in word2id.items()}
model_name = "example_gpu.model"
model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
morpheme = "信長"
sentence = [morpheme]

### Test the model ###
for i in range(30):
    index = word2id[morpheme]
    var = variable([index])
    result = torch.max(model(var), 1)[1].data[0]
    morpheme = id2word[result]
    sentence.append(morpheme)
    if morpheme == "。":
        break
print("".join(sentence))

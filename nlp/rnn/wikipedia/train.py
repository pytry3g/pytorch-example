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
# GPU使わないならFalse
# もしくは、USE_CUDA = torch.cuda.is_available()
USE_CUDA = True
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
    tensor = LongTensor(index)
    if USE_CUDA:
        return Variable(tensor).cuda()
    return Variable(tensor)

### Training ###
n_epochs = 1000
n_vocab = len(word2id)
embedding_dim = 128
hidden_dim = 128
learning_rate = 0.01

model = RNN(n_vocab, embedding_dim, hidden_dim)
if USE_CUDA:
    model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = O.Adam(model.parameters(), lr=learning_rate)

print("USE_CUDA: {}\nn_epochs: {}\nn_vocab: {}\n".format(USE_CUDA, n_epochs, n_vocab))

for epoch in range(n_epochs):
    if (epoch+1) % 100 == 0:
        print("Epoch {}".format(epoch+1))
    random.shuffle(id2sent)
    for indices in id2sent:
        model.init_hidden()
        model.zero_grad()
        source = variable(indices[:-1])
        target = variable(indices[1:])
        loss = 0
        for x, t in zip(source, target):
            y = model(x)
            loss += criterion(y, t)
        loss.backward()
        optimizer.step()

# FloydHubを使うなら
#model_name = "/output/example_gpu.model" if USE_CUDA else "/output/example.model"
# FloydHubを使わないとき
model_name = "example_gpu.model" if USE_CUDA else "example.model"
torch.save(model.state_dict(), model_name)

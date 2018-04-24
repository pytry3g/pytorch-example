import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as O
from torch.autograd import Variable
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#USE_CUDA = torch.cuda.is_available()
USE_CUDA = False
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
class NeuralNetwork(nn.Module):
    def __init__(self, n_in, n_units, n_out):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(n_in, n_units)
        self.l2 = nn.Linear(n_units, n_out)

    def forward(self, x):
        h = F.relu(self.l1(x))
        y = self.l2(h)
        return y


parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', '-b', type=int, default=20,
                    help="Number of minibatch size...")
parser.add_argument('--epochs', '-e', type=int, default=40,
                    help="Number of epochs...")
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1,
                    help="Learning rate...")
args = parser.parse_args()
batch_size = args.batchsize
n_epochs = args.epochs
learning_rate = args.learning_rate

# Load dataset...
iris_dataset = datasets.load_iris()
# data type -> numpy.ndarray
data = iris_dataset.data
# 0 -> setosa
# 1 -> versicolor
# 2 -> virginica
target = iris_dataset.target

# np.float32 -> FloatTensor
# np.int64 -> Longtensor
target = target.astype(np.int64)

# Split data into train and test set...
train_x, test_x, train_t, test_t = train_test_split(data, target, test_size=0.1)

n_in = len(data[0])
n_units = 5
n_out = len(iris_dataset.target_names)
n_batch = len(train_x) // batch_size

# Model + Loss + Optimizer
model = NeuralNetwork(n_in, n_units, n_out)
if USE_CUDA:
    model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = O.Adam(model.parameters(), lr=learning_rate)
print("GPU: {}\nepochs: {}\n".format(USE_CUDA, n_epochs))

# Training...
print("Training...")
for epoch in range(n_epochs):
    train_x, train_t = shuffle(train_x, train_t)
    for i in range(n_batch):
        start = i * batch_size
        end = start + batch_size
        x = Variable(FloatTensor(train_x[start:end]))
        t = Variable(LongTensor(train_t[start:end]))
        if USE_CUDA:
            x = x.cuda()
            t = t.cuda()
        model.zero_grad()
        y = model(x)
        loss = criterion(y, t)
        loss.backward()
        optimizer.step()
print("Done...")

# Save the model
#torch.save(model.state_dict(), "iris.model")

# Test...
test_case = Variable(FloatTensor(test_x))
if USE_CUDA:
    test_case = test_case.cuda()
result = model(test_case)
predicted = torch.max(result, 1)[1].data.numpy().tolist()
print("{:.2f}".format(sum(p == t for p, t in zip(predicted, test_t)) / len(test_x)))

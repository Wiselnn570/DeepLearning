import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

num_inputs = len(vocab)
num_hiddens = 256
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = d2l.RNNModel(gru_layer, len(vocab))
device = 'cuda:0'
model = model.to(device)
num_epochs, lr = 500, 1

d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)

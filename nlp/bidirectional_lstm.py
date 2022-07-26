import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

vocab_size, num_hiddens, num_layers = len(vocab), 256, 2

device = d2l.try_gpu()

lstm_layer = nn.LSTM(vocab_size, num_hiddens, num_layers, bidirectional=True)

model = d2l.RNNModel(lstm_layer, vocab_size)
model = model.to(device)

num_epochs, lr = 500, 1

d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)



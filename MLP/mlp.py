import torch
from torch import isin, nn
from d2l import torch as d2l

input_nums, yc_nums, output_nums = 784, 256, 10

net = nn.Sequential(
    nn.Flatten(), nn.Linear(input_nums, yc_nums), nn.ReLU(), nn.Linear(yc_nums, output_nums)
)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

batch_size, lr, num_epochs = 256, 0.1, 10

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

loss = nn.CrossEntropyLoss(reduction='none')

trainer = torch.optim.SGD(net.parameters(), lr=lr)

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

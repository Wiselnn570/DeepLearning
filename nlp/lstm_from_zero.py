import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
def get_params(vocab_size, num_hiddens, device):
    input_nums = output_nums = vocab_size

    def normal(shape):
        return torch.randn(shape, device=device) * 0.01
    
    def three():
        return (
            normal((vocab_size, num_hiddens)),
            normal((num_hiddens, num_hiddens)),
            torch.zeros(num_hiddens, device=device)
        )
    W_xi, W_hi, b_i = three()
    W_xf, W_hf, b_f = three()
    W_xo, W_ho, b_o = three()
    W_xc, W_hc, b_c = three()
    W_hq, b_q = normal((num_hiddens, output_nums)), torch.zeros(output_nums, device=device)
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q]

    for param in params:
        param.requires_grad_(True)
    return params

def init_lstm_state(vocab_size, num_hiddens, device):
    return (
        torch.zeros((vocab_size, num_hiddens), device=device),
        torch.zeros((vocab_size, num_hiddens), device=device)
    )

def lstm(inputs, state, params):
    W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q = params

    H, C = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid(X @ W_xi + H @ W_hi + b_i)
        F = torch.sigmoid(X @ W_xf + H @ W_hf + b_f)
        O = torch.sigmoid(X @ W_xo + H @ W_ho + b_o)
        C_cond = torch.tanh(X @ W_xc + H @ W_hc + b_c)
        C = (F * C) + (I * C_cond)
        H = O * torch.tanh(C)
        Y = H @ W_hq + b_q
        outputs.append(Y)
    
    return torch.cat(outputs, dim=0), (H, C)

vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_params, init_lstm_state, lstm)

d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)




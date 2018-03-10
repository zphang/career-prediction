import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, batch_size):
        super(LSTM, self).__init__()
        
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_size = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        
        self.linear_f = nn.Linear(embedding_dim + hidden_size, hidden_size)
        self.linear_i = nn.Linear(embedding_dim + hidden_size, hidden_size)
        self.linear_ctilde = nn.Linear(embedding_dim + hidden_size, hidden_size)
        self.linear_o = nn.Linear(embedding_dim + hidden_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.init_weights()
    
    def forward(self, x, hidden, c):
        x_emb = self.embed(x)
        embs = torch.chunk(x_emb, x_emb.size()[1], 1)       
        
        def step(emb, hid, c_t):
            combined = torch.cat((hid, emb), 1)
            f = F.sigmoid(self.linear_f(combined))
            i = F.sigmoid(self.linear_i(combined))
            c_tilde = F.tanh(self.linear_ctilde(combined))
            c_t = f * c_t + i * c_tilde
            o = F.sigmoid(self.linear_o(combined))
            hid = o * F.tanh(c_t)
            return hid, c_t
              
        for i in range(len(embs)):
            hidden, c = step(embs[i].squeeze(), hidden, c)     
            
        output = self.decoder(hidden)
        return output, hidden

    def init_hidden(self):
        h0 = Variable(torch.zeros(self.batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(self.batch_size, self.hidden_size))
        return h0, c0
    
    def init_weights(self):
        initrange = 0.1
        lin_layers = [self.linear_f, self.linear_i, self.linear_ctilde, self.linear_o, self.decoder]
        em_layer = [self.embed]
     
        for layer in lin_layers+em_layer:
            layer.weight.data.uniform_(-initrange, initrange)
            if layer in lin_layers:
                layer.bias.data.fill_(0)
                
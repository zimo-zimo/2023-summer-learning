import torch
from torch import nn 
from torch.nn import functional as F


class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, bidirectional, dropout, rnn_type):
        super(TextClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=embedding_dim, 
                               hidden_size=hidden_dim, 
                               num_layers=num_layers, 
                               bidirectional=bidirectional, 
                               dropout=dropout, 
                               batch_first=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=embedding_dim, 
                              hidden_size=hidden_dim, 
                              num_layers=num_layers, 
                              bidirectional=bidirectional, 
                              dropout=dropout, 
                              batch_first=True)
            
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        output = self.dropout(output)
        last_hidden = output[:, -1, :]

        if self.rnn.bidirectional:
            last_hidden = torch.cat((last_hidden[:, :self.rnn.hidden_size], last_hidden[:, self.rnn.hidden_size:]), dim=1)

        out = self.fc(last_hidden)
        out = torch.sigmoid(out)
        return out
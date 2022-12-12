import math
from numpy import size
import torch.nn as nn
import torch
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import torch
from MSModel import MSModel

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_layer = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=False, dropout=dropout)
        self.classifier = nn.Sequential(nn.BatchNorm1d(hidden_size), nn.ReLU(), nn.Dropout(), nn.Linear(hidden_size, output_size))

    def forward(self, input):
        input = torch.permute(input, (1,0,2))
        output, hidden = self.rnn_layer(input)
        # output = torch.transpose(output, 0, 1)
        # output = torch.flatten(output, 1, -1)
        pred = self.classifier(output[-1])

        return pred

    def initHidden(self, batchsz):
        return torch.zeros(1, batchsz, self.hidden_size)

class CRNN(nn.Module):
    def __init__(self, in_planes, input_size, hidden_size, num_layers, output_size):
        super(CRNN, self).__init__()
        self.extractor = MSModel(in_planes)
        self.rnn_layer = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=False)
        self.classifier = nn.Linear(hidden_size, output_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = torch.permute(x, [1, 0, 2, 3, 4])
        s = []
        for i in range(len(x)):
            s.append(self.extractor(x[i]))
        s = torch.stack(s, dim=0)
        output, hidden = self.rnn_layer(s)
        pred = self.classifier(output[-1])

        return pred

    def initHidden(self, batchsz):
        return torch.zeros(1, batchsz, self.hidden_size)

#@save
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class Transformer(nn.Module):
    def __init__(self, input_size, nhead=8, num_layers=6, output_size=2):
        super().__init__()
        self.inputsize = input_size
        num_hiddens = input_size
        self.num_hiddens = input_size
        # self.press = nn.Linear(input_size, num_hiddens)
        self.relu = nn.ReLU()
        # self.pos_encoding = PositionalEncoding(num_hiddens, 0.1)
        self.pos_encoding = nn.Parameter(torch.rand(1, input_size, num_hiddens))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=num_hiddens, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(num_hiddens, output_size)

    def forward(self, input):
        input = torch.permute(input, (1,0,2))
        # input = self.press(input)
        # input = self.pos_encoding(input * math.sqrt(self.num_hiddens))
        input = input + self.pos_encoding[:, :input.shape[1], :].to(input.device)
        output = self.transformer_encoder(input)
        pred = self.classifier(output[-1])

        return pred

if __name__ == '__main__':
    
    model = CRNN(496, 496, 4, 2)
    x  = torch.rand([1, 6, 3, 224, 224])
    model(x).shape

import torch.nn as nn
import torch
from MSModel import MSModel
from rnn import CRNN


class OCRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, omic_len, output_size):
        super(OCRNN, self).__init__()
        crnn = CRNN(3, input_size, hidden_size, num_layers, output_size)
        crnn.load_state_dict(torch.load(r'论文\exp36\ckpts\scale_model.pth'))
        crnn.classifier = nn.Sequential()
        for param in crnn.parameters():
            param.requires_grad = False
        self.crnn = crnn
        self.linear0 = nn.Sequential(nn.Linear(omic_len, 2*omic_len), nn.LeakyReLU(), nn.Linear(2*omic_len, omic_len), nn.LeakyReLU())
        f_len = omic_len + 496
        self.classifier = nn.Linear(f_len, output_size)
    
    def forward(self, x_crnn, x_omic):
        f_crnn = self.crnn(x_crnn)
        x_omic = self.linear0(x_omic)
        x_domic = torch.cat([f_crnn, x_omic], dim=1)
        out = self.classifier(x_domic)

        return out
    

if __name__ == '__main__':
    ocrnn = OCRNN(496, 496, 4, 37, 2)
    x1 = torch.rand([4, 6, 3, 224, 224])
    x2 = torch.rand([4, 37])
    print(torch.softmax(ocrnn(x1, x2), dim=-1))
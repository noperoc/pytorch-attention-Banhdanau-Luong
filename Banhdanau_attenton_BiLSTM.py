import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
from torch import optim
from torch.utils.data.dataloader import DataLoader

class Encoder(nn.Module):
    """
    """
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        # self.enc_size = enc_size
        # self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, src):
        """src = [batch, len, dim]"""
        # embedded = self.embedding(input).view(1, 1, -1)
        # output = embedded..

        output, hidden = self.lstm(src)
        enc_hidden_h = hidden[0]
        enc_hidden_c = hidden[1]
        s_h = torch.cat((enc_hidden_h[-2,:,:], enc_hidden_h[-1,:,:]), dim=1)
        s_c = torch.cat((enc_hidden_c[-2,:,:], enc_hidden_c[-1,:,:]), dim=1)
        s_h = torch.tanh(self.fc(s_h))
        s_c = torch.tanh(self.fc(s_c))
        #output[batch, len, hidden*2] s[batch, hidden]
        return output, s_h, s_c

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 3, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    def forward(self, enc_output, s):
        srclen = enc_output.shape[1]
        #s[batch, len, hidden_size]
        s = s.unsqueeze(1).repeat(1, srclen, 1)
        #mul[batch, len, hidden_size]
        mul = torch.tanh(self.attn(torch.cat((enc_output, s), dim=2)))
        alpha_hat = self.v(mul).squeeze(2)
        return F.softmax(alpha_hat, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, input_size, dropout, attention):
        super().__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.attention = attention
        self.rnn = nn.LSTM((hidden_size * 2 + input_size), hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, dec_input, s_h, s_c, enc_output):
        # dec_input[batch, input_size]
        dec_input = dec_input.unsqueeze(1)
        # enc_output[batch, len, hidden_size*2]
        # alpha[batch, len]
        s = torch.cat((s_h, s_c), dim=1)
        s = torch.tanh(self.fc(s))
        alpha = self.attention(enc_output, s).unsqueeze(1)
        # attn_weight[batch, 1, hidden_size*2]
        attn_weight = torch.bmm(alpha, enc_output)
        rnn_input = torch.cat((attn_weight, dec_input), dim=2)
        # dec_output[bacth, 1, hidden_szie] dec_hidden[layer, batch, hidden_size] 一个h 一个c
        dec_output, dec_hidden = self.rnn(rnn_input, (s_h.unsqueeze(0), s_c.unsqueeze(0)))
        dec_hidden_h = dec_hidden[0][0]
        dec_hidden_c = dec_hidden[1][0]
        dec_output = self.out(dec_output)
        # dec_output[batch, output_size], dec_hidden_x[batch, hidden_size]
        return dec_output.squeeze(1), dec_hidden_h, dec_hidden_c


class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    def forward(self, src, trg, teaching_force_ratio = 0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        dim = src.shape[2]
        outputs = torch.zeros(batch_size, trg_len, dim).to(self.device)

        enc_output, s_h, s_c = self.encoder(src)

        dec_input = torch.zeros(batch_size, dim)
        for t in range(0, trg_len):
            dec_output, s_h, s_c = self.decoder(dec_input, s_h, s_c, enc_output)
            outputs[:, t, :] = dec_output
            teacher_force = random.random() < teaching_force_ratio
            dec_input = trg[:, t, :] if teacher_force else dec_output

        return outputs


dim = 3
hidden_size = 256

attn = Attention(hidden_size=hidden_size)
encoder = Encoder(input_size=dim, hidden_size=hidden_size)
decoder = Decoder(output_size=dim, hidden_size=hidden_size, input_size=dim, dropout=0.1, attention=attn)
model = Seq2seq(encoder=encoder, decoder=decoder, device='cpu')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epoch = 0

while epoch < 300:
    optimizer.zero_grad()
    src = torch.Tensor([[[1,1,1],[2,2,2],[3,3,3]],[[2,2,2],[3,3,3],[4,4,4]]])
    trg = torch.Tensor([[[2,2,2],[3,3,3],[4,4,4]],[[3,3,3],[4,4,4],[5,5,5]]])
    pred = model(src, trg)
    loss = criterion(pred, trg)
    loss.backward()
    optimizer.step()
    print(pred, loss.item())
    epoch += 1
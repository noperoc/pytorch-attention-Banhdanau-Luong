import torch
import math
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
        self.rnn = nn.GRU(self.input_size, self.hidden_size, batch_first=True)

    def forward(self, src):
        """src = [batch, len, dim]"""
        output, hidden = self.rnn(src)
        # output [batch, len, hidden_size], hidden [layer, batch, hidden_size]
        return output, hidden

class GlobalAttention(nn.Module):
    def __init__(self, hidden_size, method):
        super(GlobalAttention, self).__init__()
        self.hidden_size = hidden_size
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.method = method
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

    def forward(self, enc_output, s):
        batch_size = enc_output.shape[0]
        seq_len = enc_output.shape[1]
        alpha_hat = torch.zeros(batch_size, seq_len)
        if self.method == 'concat':
            srclen = enc_output.shape[1]
            # s[batch, len, hidden_size]
            s = s.unsqueeze(1).repeat(1, srclen, 1)
            # mul[batch, len, hidden_size]
            mul = torch.tanh(self.attn(torch.cat((enc_output, s), dim=2)))
            alpha_hat = self.v(mul).squeeze(2)
        elif self.method == 'general':
            enc_output = self.attn(enc_output)
            s = s.unsqueeze(2)
            alpha_hat = torch.bmm(enc_output, s).squeeze(2)
        elif self.method == 'dot':
            s = s.unsqueeze(2)
            alpha_hat = torch.bmm(enc_output, s).squeeze(2)
        return F.softmax(alpha_hat, dim=1)

class LocalAttention(nn.Module):
    def __init__(self, hidden_size, method, len, windows):
        super(LocalAttention, self).__init__()
        self.hidden_size = hidden_size
        self.method = method
        self.s_len = len
        self.D = windows
        self.omega_w = nn.Linear(self.hidden_size, self.hidden_size)
        self.omega_v = nn.Linear(self.hidden_size, 1)
        self.v = nn.Linear(self.hidden_size, 1)
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
    def forward(self, enc_output, s):
        sigmoid_inp = self.omega_v(F.tanh(self.omega_w(s))).squeeze(1)
        S = self.s_len - 2*self.D - 1
        ps = S * F.sigmoid(sigmoid_inp)
        pt = ps + self.D
        batch_size = enc_output.shape[0]
        enc_output_ = torch.zeros(1, 2*self.D+1, hidden_size)
        for i in range(batch_size):
            enc_output_i = enc_output[i, int(pt[i])-self.D:int(pt[i])+self.D+1, :]
            enc_output_ = torch.cat((enc_output_, enc_output_i.unsqueeze(0)), dim=0)
        enc_output = enc_output_[1:,:,:]
        enc_w_output = enc_output
        seq_len = enc_w_output.size(1)
        sigma = self.D/2
        alpha_hat = torch.zeros(batch_size, seq_len)
        if self.method == 'concat':
            # s[batch, len, hidden_size]
            s = s.unsqueeze(1).repeat(1, seq_len, 1)
            # mul[batch, len, hidden_size]
            mul = torch.tanh(self.attn(torch.cat((enc_w_output, s), dim=2)))
            alpha_hat = self.v(mul).squeeze(2)
        elif self.method == 'general':
            enc_w_output = self.attn(enc_w_output)
            s = s.unsqueeze(2)
            alpha_hat = torch.bmm(enc_w_output, s).squeeze(2)
        elif self.method == 'dot':
            s = s.unsqueeze(2)
            alpha_hat = torch.bmm(enc_w_output, s).squeeze(2)
        gauss = []
        for i in range(batch_size):
            gauss_socre = []
            for j in range(int(pt[i])-self.D, int(pt[i])+self.D+1):
                gauss_socre_i = math.exp(-(pow((j-pt[i].item()),2))/(2*(pow(sigma, 2))))
                gauss_socre.append(gauss_socre_i)
            gauss.append(gauss_socre)
        gauss = torch.Tensor(gauss)
        energies = alpha_hat * gauss
        return F.softmax(energies, dim=1), enc_w_output

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, input_size, dropout, attention):
        super().__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.attention = attention
        self.rnn = nn.GRU((hidden_size+input_size), hidden_size, batch_first=True)
        # self.rnn = nn.LSTM((hidden_size * 2 + input_size), hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, dec_input, lasts, lastc, enc_output):
        # dec_input[batch, input_size]
        # enc_output[batch, len, hidden_size]
        # alpha[batch, len]
        dec_input = dec_input.unsqueeze(1)
        lastc = lastc.unsqueeze(1)
        rnn_input = torch.cat((dec_input, lastc), dim=2)
        dec_output, hidden = self.rnn(rnn_input, lasts)

        alpha = self.attention(enc_output, hidden.squeeze(0)) # Global-attention
        # alpha, enc_w_output = self.attention(enc_output, hidden.squeeze(0)) # Local-attention

        # attn_weight[batch, 1, hidden_size] == context
        attn_weight = torch.bmm(alpha.unsqueeze(1), enc_output) # Global-attention
        # attn_weight = torch.bmm(alpha.unsqueeze(1), enc_w_output) # Local-attention

        dec_output = self.out(dec_output)
        return dec_output.squeeze(1), attn_weight.squeeze(1), hidden

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

        enc_output, s = self.encoder(src)

        dec_input = torch.zeros(batch_size, dim)
        deccontext = torch.zeros(batch_size, hidden_size)

        for t in range(0, trg_len):
            dec_output, deccontext, s = self.decoder(dec_input, s, deccontext, enc_output)
            outputs[:, t, :] = dec_output
            teacher_force = random.random() < teaching_force_ratio
            dec_input = trg[:, t, :] if teacher_force else dec_output

        return outputs

dim = 3
hidden_size = 256

# method = ['concat', 'dot', 'general'] general在样例实验中是最强的
attn = GlobalAttention(hidden_size=hidden_size, method='general')
# attn = LocalAttention(hidden_size=hidden_size, method='general', len=15, windows=2)
encoder = Encoder(input_size=dim, hidden_size=hidden_size)
decoder = Decoder(output_size=dim, hidden_size=hidden_size, input_size=dim, dropout=0.1, attention=attn)
model = Seq2seq(encoder=encoder, decoder=decoder, device='cpu')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epoch = 0

while epoch < 300:
    optimizer.zero_grad()
    src = torch.Tensor([[[1,1,1],[2,2,2],[3,3,3],[1,1,1],[2,2,2],[3,3,3],[1,1,1],[2,2,2],[3,3,3],[1,1,1],[2,2,2],[3,3,3],[1,1,1],[2,2,2],[3,3,3]],[[2,2,2],[3,3,3],[4,4,4],[2,2,2],[3,3,3],[4,4,4],[2,2,2],[3,3,3],[4,4,4],[2,2,2],[3,3,3],[4,4,4],[2,2,2],[3,3,3],[4,4,4]]])
    trg = torch.Tensor([[[2,2,2],[3,3,3],[4,4,4],[2,2,2],[3,3,3],[4,4,4],[2,2,2],[3,3,3],[4,4,4],[2,2,2],[3,3,3],[4,4,4],[2,2,2],[3,3,3],[4,4,4]],[[3,3,3],[4,4,4],[5,5,5],[3,3,3],[4,4,4],[5,5,5],[3,3,3],[4,4,4],[5,5,5],[3,3,3],[4,4,4],[5,5,5],[3,3,3],[4,4,4],[5,5,5]]])
    pred = model(src, trg)
    loss = criterion(pred, trg)
    loss.backward()
    optimizer.step()
    print(pred, loss.item())
    epoch += 1
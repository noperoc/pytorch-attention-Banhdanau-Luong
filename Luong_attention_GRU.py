import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import random
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim import lr_scheduler

import utils
import argparse
import os
import shutil

parser = argparse.ArgumentParser(
    description='Train seq2seq GRU with Luongc attention...')
parser.add_argument('--name', type=str, default='local-attention-500-3D')
parser.add_argument('--dataset_folder', type=str, default='Data/Data_')
parser.add_argument('--dim', type=int, default=3)
parser.add_argument('--max_epoch', type=int, default=40)
parser.add_argument('--dropout', type=float, default=0.02)
parser.add_argument('--device', type=str, default='0')
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--cnts', type=int, default=100000)
parser.add_argument('--track_len', type=int, default=500)
parser.add_argument('--sequence', type=int, default=500)
parser.add_argument('--gamma', default=0.9, type=float)
parser.add_argument('--step', default=10, type=int)

args = parser.parse_args()

log_dir = os.path.join('logs', args.name)
out_dir = os.path.join('outs', args.name)
model_dir = os.path.join('models', args.name)

data_path = 'Data/Data_'+str(args.dim)+'D/'

# if args.del_hist:
#     if os.path.exists(log_dir):
#         print('Deleting history logs: ' + log_dir)
#         shutil.rmtree(log_dir)
#     if os.path.exists(out_dir):
#         print('Deleting history outputs: ' + out_dir)
#         shutil.rmtree(out_dir)
#     if os.path.exists(model_dir):
#         print('Deleting history models: ' + model_dir)
#         shutil.rmtree(model_dir)

try:
    os.mkdir('logs')
except FileExistsError:
    pass
try:
    os.mkdir('outs')
except FileExistsError:
    pass
try:
    os.mkdir('models')
except FileExistsError:
    pass
try:
    os.mkdir(os.path.join('outs', args.name))
except FileExistsError:
    pass
try:
    os.mkdir(os.path.join('models', args.name))
except FileExistsError:
    pass

log = SummaryWriter(f'logs/{args.name}')

os.environ["CUDA_VISIBLE_DEVICES"] = args.device

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

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
        alpha_hat = torch.zeros(batch_size, seq_len, device=device)
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
        enc_output_ = torch.zeros(1, 2*self.D+1, args.hidden_size, device=device)
        for i in range(batch_size):
            enc_output_i = enc_output[i, int(pt[i])-self.D:int(pt[i])+self.D+1, :]
            enc_output_ = torch.cat((enc_output_, enc_output_i.unsqueeze(0)), dim=0)
        enc_output = enc_output_[1:,:,:]
        enc_w_output = enc_output
        seq_len = enc_w_output.size(1)
        sigma = self.D/2
        alpha_hat = torch.zeros(batch_size, seq_len, device=device)
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

        # alpha = self.attention(enc_output, hidden.squeeze(0)) # Global-attention
        alpha, enc_w_output = self.attention(enc_output, hidden.squeeze(0)) # Local-attention

        # attn_weight[batch, 1, hidden_size] == context
        # attn_weight = torch.bmm(alpha.unsqueeze(1), enc_output) # Global-attention
        attn_weight = torch.bmm(alpha.unsqueeze(1), enc_w_output) # Local-attention

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

        dec_input = torch.zeros(batch_size, dim).to(self.device)
        deccontext = torch.zeros(batch_size, args.hidden_size).to(self.device)

        for t in range(0, trg_len):
            dec_output, deccontext, s = self.decoder(dec_input, s, deccontext, enc_output)
            outputs[:, t, :] = dec_output
            teacher_force = random.random() < teaching_force_ratio
            dec_input = trg[:, t, :] if teacher_force else dec_output

        return outputs

def calcRMSE_matlab(test_gt, pre_trk):
    rmses = []
    for i in range(test_gt[0]):
        res = test_gt[i] - pre_trk[i]
        res = pow(res , 2)
        rmse_mean = res.mean(axis=0)
        rmse = rmse_mean.mean()
        rmses.append(rmse)
    return torch.Tensor(rmses).mean()

def train(model, iterator, optimizer, criterion, mean, std, device):
    model.train()
    epoch_loss = 0
    train_batch_bar = tqdm([(id_, batch) for id_, batch in enumerate(train_dl)])
    for _, batch in enumerate(train_batch_bar):
        input_tensor = (batch['src'] - mean) / std
        target_tensor = (batch['trg'] - mean) / std
        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)
        pred = model(input_tensor, target_tensor)
        loss = criterion(target_tensor, pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        train_batch_bar.set_description(
            "train epoch %03i/%03i  loss: %7.4f " % (
                epoch + 1, args.max_epoch, loss.item()))
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, mean, std, device):
    model.eval()
    epoch_loss = 0
    RMSE = 0
    with torch.no_grad():
        val_batch_bar = tqdm([(id_, batch)
                              for id_, batch in enumerate(val_dl)])
        for _, batch in enumerate(val_batch_bar):
            input_tensor = (batch['src'] - mean) / std
            target_tensor = (batch['trg'] - mean) / std
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)
            pred = model(input_tensor, target_tensor, 0)
            loss = criterion(target_tensor, pred)
            epoch_loss += loss.item()
            pred = pred * std + mean
            rmse = calcRMSE_matlab(batch['trg'], pred)
            RMSE += rmse.item()
            val_batch_bar.set_description(
                "val: %03i/%03i loss: %7.4f rmse: %7.4f" % (
                    epoch + 1, args.max_epoch, loss.item(), rmse.item()))
    return epoch_loss / len(iterator), RMSE / len(iterator)

train_dt, val_dt, test_dt = utils.create_dataset(data_path, args.cnts, args.sequence, args.track_len)

train_dl = torch.utils.data.DataLoader(train_dt, batch_size=32, shuffle=True,
                                       num_workers=3)
val_dl = torch.utils.data.DataLoader(val_dt, batch_size=32, shuffle=True,
                                     num_workers=0)
test_dl = torch.utils.data.DataLoader(test_dt, batch_size=32, shuffle=True,
                                     num_workers=0)

mean = torch.cat((train_dt[:]['src'], train_dt[:]['trg']), 1).mean((0, 1))
std = torch.cat((train_dt[:]['src'], train_dt[:]['trg']), 1).std((0, 1))

best_val_loss = float('inf')

# method = ['concat', 'dot', 'general'] general在样例实验中是最强的
# attn = GlobalAttention(hidden_size=args.hidden_size, method='general')
attn = LocalAttention(hidden_size=args.hidden_size, method='general', len=500, windows=6)
encoder = Encoder(input_size=args.dim, hidden_size=args.hidden_size)
decoder = Decoder(output_size=args.dim, hidden_size=args.hidden_size, input_size=args.dim, dropout=args.dropout, attention=attn)
model = Seq2seq(encoder=encoder, decoder=decoder, device=device)
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)

for epoch in range(args.max_epoch):
    train_loss = train(model, train_dl, optimizer, criterion, mean, std, device)
    valid_loss, Rmse = evaluate(model, val_dl, criterion, mean, std, device)
    if valid_loss < best_val_loss:
        best_val_loss = valid_loss
        torch.save(model.state_dict(), os.path.join(model_dir, 'model.pt'))
    log.add_scalar('loss/train', train_loss, epoch)
    log.add_scalar('loss/val', valid_loss, epoch)
    log.add_scalar('Rmse/val', Rmse, epoch)
    print(f'Epoch:{epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} ')
    print(f'\t Val. Loss: {valid_loss:.3f} ')

    if epoch > args.warmup:
        scheduler.step()

model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pt')))
_, RMSE = evaluate(model, test_dl, criterion, mean, std, device)
print(f'RMSE:{RMSE:.5f}')







# while epoch < 300:
#     optimizer.zero_grad()
#     src = torch.Tensor([[[1,1,1],[2,2,2],[3,3,3],[1,1,1],[2,2,2],[3,3,3],[1,1,1],[2,2,2],[3,3,3],[1,1,1],[2,2,2],[3,3,3],[1,1,1],[2,2,2],[3,3,3]],[[2,2,2],[3,3,3],[4,4,4],[2,2,2],[3,3,3],[4,4,4],[2,2,2],[3,3,3],[4,4,4],[2,2,2],[3,3,3],[4,4,4],[2,2,2],[3,3,3],[4,4,4]]])
#     trg = torch.Tensor([[[2,2,2],[3,3,3],[4,4,4],[2,2,2],[3,3,3],[4,4,4],[2,2,2],[3,3,3],[4,4,4],[2,2,2],[3,3,3],[4,4,4],[2,2,2],[3,3,3],[4,4,4]],[[3,3,3],[4,4,4],[5,5,5],[3,3,3],[4,4,4],[5,5,5],[3,3,3],[4,4,4],[5,5,5],[3,3,3],[4,4,4],[5,5,5],[3,3,3],[4,4,4],[5,5,5]]])
#     pred = model(src, trg)
#     loss = criterion(pred, trg)
#     loss.backward()
#     optimizer.step()
#     print(pred, loss.item())
#     epoch += 1
#
# def get_parameter_number(model):
#     total_num = sum(p.numel() for p in model.parameters())
#     trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     return {'Total': total_num, 'Trainable': trainable_num}
#
# print(get_parameter_number(model)) # Global 79w5 Local 86w1
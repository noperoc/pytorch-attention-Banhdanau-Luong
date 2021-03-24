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
parser.add_argument('--name', type=str, default='BiLSTM-attention-500-3D')
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

        dec_input = torch.zeros(batch_size, dim, device=device)
        for t in range(0, trg_len):
            dec_output, s_h, s_c = self.decoder(dec_input, s_h, s_c, enc_output)
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

attn = Attention(hidden_size=args.hidden_size)
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



#
# while epoch < 300:
#     optimizer.zero_grad()
#     src = torch.Tensor([[[1,1,1],[2,2,2],[3,3,3]],[[2,2,2],[3,3,3],[4,4,4]]])
#     trg = torch.Tensor([[[2,2,2],[3,3,3],[4,4,4]],[[3,3,3],[4,4,4],[5,5,5]]])
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
# print(get_parameter_number(model)) #178w
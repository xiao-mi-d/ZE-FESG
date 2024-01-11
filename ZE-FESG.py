from argparse import ArgumentParser
import os
import h5py
import torch
from torch.optim import Adam, lr_scheduler, adamw
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random
from scipy import stats
from tensorboardX import SummaryWriter
import datetime
from transformers import AdamW, WarmupLinearSchedule

class VQADataset(Dataset):
    def __init__(self, features_dir='../feature/', index=None, max_len=240, feat_dim=2385, scale=1):
        super(VQADataset, self).__init__()
        self.features = np.zeros((len(index), max_len, feat_dim))
        self.length = np.zeros((len(index), 1))
        self.mos = np.zeros((len(index), 1))
        for i in range(len(index)):
            features = np.load(features_dir + str(index[i]) + 'clip.npy')
            self.length[i] = features.shape[0]
            self.features[i, :features.shape[0], :] = features
            self.mos[i] = np.load(features_dir + str(index[i]) + '_score.npy')  #
        self.scale = scale  #
        self.label = self.mos / self.scale  # label normalization

    def __len__(self):
        return len(self.mos)

    def __getitem__(self, idx):
        sample = self.features[idx], self.length[idx], self.label[idx]
        return sample


class ANN(nn.Module):
    def __init__(self, input_size=2385, reduced_size=1024, n_ANNlayers=1, dropout_p=0.5):
        super(ANN, self).__init__()
        self.n_ANNlayers = n_ANNlayers
        self.fc0 = nn.Linear(input_size, reduced_size)  #
        self.dropout = nn.Dropout(p=dropout_p)  #
        self.fc = nn.Linear(reduced_size, reduced_size)  #

    def forward(self, input):
        input = self.fc0(input)  # linear
        for i in range(self.n_ANNlayers-1):  # nonlinear
            input = self.fc(self.dropout(F.relu(input)))
        return input


def TP(q, tau=24, beta=0.5):
    """subjectively-inspired temporal pooling"""
    q = torch.unsqueeze(torch.t(q), 0)
    qm = -float('inf')*torch.ones((1, 1, tau-1)).to(q.device)
    qp = 10000.0 * torch.ones((1, 1, tau - 1)).to(q.device)  #
    l = -F.max_pool1d(torch.cat((qm, -q), 2), tau, stride=1)
    m = F.avg_pool1d(torch.cat((q * torch.exp(-q), qp * torch.exp(-qp)), 2), tau, stride=1)
    n = F.avg_pool1d(torch.cat((torch.exp(-q), torch.exp(-qp)), 2), tau, stride=1)
    m = m / n
    return beta * m + (1 - beta) * l


class VSFA(nn.Module):
    def __init__(self, input_size=2385, reduced_size=128, hidden_size=32):

        super(VSFA, self).__init__()
        self.hidden_size = hidden_size
        self.ann = ANN(input_size, reduced_size, 1)
        self.rnn = nn.GRU(reduced_size, hidden_size, batch_first=True)
        self.q = nn.Linear(hidden_size, 1)
    def forward(self, input, input_length):
        input = self.ann(input)  # dimension reduction
        outputs, _ = self.rnn(input, self._get_initial_state(input.size(0), input.device))
        q = self.q(outputs)  # frame quality
        score = torch.zeros_like(input_length, device=q.device)  #
        for i in range(input_length.shape[0]):  #
            qi = q[i, :int(input_length[i].numpy())]   #todo np.int
            qi = TP(qi)
            score[i] = torch.mean(qi)  # video overall quality
        return score

    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h0


if __name__ == "__main__":
    epochs = 2000
    decay_interval = epochs/10
    decay_ratio = 0.8
    seed=19920517
    torch.manual_seed(seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.utils.backcompat.broadcast_warning.enabled = True
    features_dir = '../feature/'  # features dir
    datainfo = '../data/KoNViD-1kinfo.mat'  # database info: video_names, scores; video format, width, height, index, ref_ids, max_len, etc.

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    Info = h5py.File(datainfo, 'r')  # index, ref_ids
    index = Info['index']

    index = index[:, 0 % index.shape[1]]   # np.random.permutation(N)  len(index) = 1200

    ref_ids = Info['ref_ids'][0, :]  #
    max_len = int(Info['max_len'][0])
    trainindex = index[int(np.ceil((1 - 0.6) * len(index))):int(np.ceil(len(index)))]
    testindex = index[0:int(np.ceil((1 - 0.2 - 0.6) * len(index)))]
    train_index, val_index, test_index = [], [], []
    for i in range(len(ref_ids)):

        if (ref_ids[i] in trainindex):
            train_index.append(i)
        elif (ref_ids[i] in testindex):
            test_index.append(i)
        else:
            val_index.append(i)

    scale = Info['scores'][0, :].max()  # label normalization factor

    train_dataset = VQADataset(features_dir, train_index, max_len, scale=scale)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=0)
    for features, length, label in train_loader:
        aa=features
        bb=length
        cc=label
    val_dataset = VQADataset(features_dir, val_index, max_len, scale=scale)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, num_workers=0)
    test_dataset = VQADataset(features_dir, test_index, max_len, scale=scale)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, num_workers=0)
    print("123")
    model = VSFA().to(device)

    if not os.path.exists('../models'):
         os.makedirs('../models')
    trained_model_file = '../models/{}-{}-EXP{}'.format('VSFA', 'KoNViD-1k', 0)
    if not os.path.exists('../results'):
         os.makedirs('../results')
    save_result_file = '../results/{}-{}-EXP{}'.format('VSFA', 'KoNViD-1k', 0)

    writer = SummaryWriter('../My_tensorboard/VQA_model5')
    optimizer = AdamW(model.parameters(), lr=0.00005, eps=1e-6)
    scheduler = WarmupLinearSchedule(optimizer, 15, 200)
    criterion = nn.L1Loss()
    best_val_criterion = -1  # SROCC min
    for epoch in range(200):
         # Train
        model.train()
        L = 0
        for i, (features, length, label) in enumerate(train_loader):
            a= features
            features = features.to(device).float()
            label = label.to(device).float()
            optimizer.zero_grad()  #
            outputs = model(features, length.float())
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            L = L + loss.item()
        train_loss = L / (i + 1)
        scheduler.step()
        model.eval()
        # Val
        y_pred = np.zeros(len(val_index))
        y_val = np.zeros(len(val_index))
        L = 0
        with torch.no_grad():
            for i, (features, length, label) in enumerate(val_loader):
                y_val[i] = scale * label.item()  #
                features = features.to(device).float()
                label = label.to(device).float()
                outputs = model(features, length.float())
                y_pred[i] = scale * outputs.item()
                loss = criterion(outputs, label)
                L = L + loss.item()
        val_loss = L / (i + 1)
        val_PLCC = stats.pearsonr(y_pred, y_val)[0]
        val_SROCC = stats.spearmanr(y_pred, y_val)[0]
        val_RMSE = np.sqrt(((y_pred-y_val) ** 2).mean())
        val_KROCC = stats.stats.kendalltau(y_pred, y_val)[0]

        # Test
        #if args.test_ratio > 0 and not args.notest_during_training:
        y_pred = np.zeros(len(test_index))
        y_test = np.zeros(len(test_index))
        L = 0
        with torch.no_grad():
            for i, (features, length, label) in enumerate(test_loader):
                y_test[i] = scale * label.item()  #
                features = features.to(device).float()
                label = label.to(device).float()
                outputs = model(features, length.float())
                y_pred[i] = scale * outputs.item()
                loss = criterion(outputs, label)
                L = L + loss.item()
        test_loss = L / (i + 1)
        PLCC = stats.pearsonr(y_pred, y_test)[0]
        SROCC = stats.spearmanr(y_pred, y_test)[0]
        RMSE = np.sqrt(((y_pred-y_test) ** 2).mean())
        KROCC = stats.stats.kendalltau(y_pred, y_test)[0]

        #if not args.disable_visualization:  # record training curves
        writer.add_scalar("loss/train", train_loss, epoch)  #
        writer.add_scalar("loss/val", val_loss, epoch)  #
        writer.add_scalar("SROCC/val", val_SROCC, epoch)  #
        writer.add_scalar("KROCC/val", val_KROCC, epoch)  #
        writer.add_scalar("PLCC/val", val_PLCC, epoch)  #
        writer.add_scalar("RMSE/val", val_RMSE, epoch)  #
       # if args.test_ratio > 0 and not args.notest_during_training:
        writer.add_scalar("loss/test", test_loss, epoch)  #
        writer.add_scalar("SROCC/test", SROCC, epoch)  #
        writer.add_scalar("KROCC/test", KROCC, epoch)  #
        writer.add_scalar("PLCC/test", PLCC, epoch)  #
        writer.add_scalar("RMSE/test", RMSE, epoch)  #

        # Update the model with the best val_SROCC
        print("epoch {}".format(epoch))
        if val_SROCC > best_val_criterion:
            print("EXP ID={}: Update best model using best_val_criterion in epoch {}".format(0, epoch))
            print("Val results: val loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                  .format(val_loss, val_SROCC, val_KROCC, val_PLCC, val_RMSE))
            #if args.test_ratio > 0 and not args.notest_during_training:
            print("Test results: test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                      .format(test_loss, SROCC, KROCC, PLCC, RMSE))
            np.save(save_result_file, (y_pred, y_test, test_loss, SROCC, KROCC, PLCC, RMSE, test_index))
            torch.save(model.state_dict(), trained_model_file)
            best_val_criterion = val_SROCC  # update best val SROCC



# -*- coding:utf-8 -*-

import warnings
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

# 数据准备与Dataloader定义
class CustomDataset(Dataset):
    def __init__(self, X_major, X_minor, y_major, y_minor):
        self.X_major = X_major
        self.X_minor = X_minor
        self.y_major = y_major
        self.y_minor = y_minor

    def __len__(self):
        return len(self.X_major)

    def __getitem__(self, index):
        x_major_idx = torch.from_numpy(self.X_major[index]).float()
        x_minor_idx = torch.from_numpy(self.X_minor[index]).float()
        y_major_idx = torch.from_numpy(self.y_major[index]).float()
        y_minor_idx = torch.from_numpy(self.y_minor[index]).float()
        
        return x_major_idx, x_minor_idx, y_major_idx, y_minor_idx
    
def initialize_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


class MLPmodel(nn.Module):
    def __init__(self, input_dim, output_dim, layer_num, hidden_dims, dropout_rates):
        super(MLPmodel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_num = layer_num-1
        self.hidden_dims = hidden_dims
        self.dropout_rates = dropout_rates
        self.model = self._mlp()
        self.model.apply(initialize_weight)
    
    def _mlp(self):     
        layers = [nn.Linear(self.input_dim, self.hidden_dims[0]), nn.ReLU(), nn.Dropout(self.dropout_rates[0])]
        for i in range(0, self.layer_num):
            layers.extend([nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]), nn.ReLU(), nn.Dropout(self.dropout_rates[i+1])])
        layers.append(nn.Linear(self.hidden_dims[-1], self.output_dim))
        layers.append(nn.Softmax(dim=1)) 
        mlp = nn.Sequential(*layers)
        return mlp

    def forward(self, x):
        # x: (n sample, m gene)
        # output: (n sample, k cell proportions)
        return self.model(x)
    
# 损失函数, major
base_loss = nn.L1Loss(reduction='none')
def loss_major(x, y, weights):
    l1_loss = base_loss(x, y)
    # 确保x和y与权重有相同的形状
    if l1_loss.shape[1] != weights.shape[0]:
        raise ValueError("The number of weights does not match the number of features in input.")
    # 将L1损失与权重相乘
    weighted_loss = l1_loss * weights
    return weighted_loss.mean()

# 损失函数, minor_mae
def loss_minor_mae(pred, label, weights, main_props, types_info, lambdaP=0.2):
    mae_loss = (torch.abs(pred - label) * weights).mean()
    
    major_cells = []; minor_cells = [] 
    for key, values in types_info.items():
        major_cells.append(key)
        minor_cells += values
    # 对于每个主类，累加其下所有小类的占比
    recalculated_main_props = torch.zeros((pred.shape[0], len(major_cells)), device=pred.device, dtype=pred.dtype)
    for major_idx, (major, minors) in enumerate(types_info.items()):
        if len(minors) > 0:
            indices = [minor_cells.index(minor) for minor in minors]
            recalculated_main_props[:, major_idx] = torch.sum(pred[:, indices], dim=1)
        else:
            # 如果大类没有对应的小类，则默认比例为0
            recalculated_main_props[:, major_idx] = 0
    # 计算重新计算的主类占比与pred_main之间的MAE作为惩罚项
    penalty = torch.mean(torch.abs(recalculated_main_props - main_props))
    
    # 加入惩罚项
    total_loss = (1-lambdaP) * mae_loss + lambdaP * penalty
    
    return total_loss

# 训练函数
def model_train(model, dataloader, loss_func, weights, optimizer, device, epochs, 
                model_major=None, use_true_labels=False, types_info = dict(), lambdaP = 0.5):
    model.to(device)
    model.train()
    loss = []
    for epoch in tqdm(range(epochs), desc="Epochs"):
        for X_major, X_minor, y_major, y_minor in dataloader:
            X_major, X_minor, y_major, y_minor = X_major.to(device), X_minor.to(device), y_major.to(device), y_minor.to(device)
            optimizer.zero_grad()
            if model_major is not None:
                outputs = model(X_minor)
                if use_true_labels:
                    batch_loss = loss_func(outputs, y_minor, weights, y_major, types_info, lambdaP)
                else:
                    # pred_main = model_major(X_major)
                    with torch.no_grad():
                        pred_main = model_major(X_major)
                    batch_loss = loss_func(outputs, y_minor, weights, pred_main, types_info, lambdaP)
            else:
                outputs = model(X_major)
                batch_loss = loss_func(outputs, y_major, weights)
            batch_loss.backward()
            optimizer.step()
            loss.append(batch_loss.cpu().detach().numpy())
    
    return model, loss

# 预测函数
def model_predict(model, X, device):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X, dtype=torch.float32).to(device)
        pred = model(inputs)
        return pred.cpu().numpy()


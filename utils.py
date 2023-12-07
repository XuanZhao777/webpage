# utils.py
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = 0.1
    if epoch >= 75:
        lr = 0.01
    if epoch >= 90:
        lr = 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_pred(out, labels):
    pred = out.sort(dim=-1, descending=True)[1][:, 0]
    second_pred = out.sort(dim=-1, descending=True)[1][:, 1]
    adv_label = torch.where(pred == labels, second_pred, pred)

    return adv_label

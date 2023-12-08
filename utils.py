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


def get_pred(outputs, labels):
    raw_output = outputs[0]  # Assuming the raw output is the first element of the tuple
    pred = raw_output.max(1)[1]  # Use max instead of sort to get the index of the maximum value
    correct = pred.eq(labels).sum().item()
    return correct, pred

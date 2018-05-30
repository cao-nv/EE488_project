from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd 
class MyDataset(Dataset):
    def __init__(self, phase):
        self.phase = phase
        if phase =='train': 
            data = pd.read_csv('./imbalance_data/train_feat.csv',header=None) 
            label = pd.read_csv('./imbalance_data/train_label.csv',header=None)
            self.data = data.values.T
            self.label = label.values - 1
        elif phase == 'valid': 
            data = pd.read_csv('./imbalance_data/valid_feat.csv',header=None)
            label = pd.read_csv('./imbalance_data/valid_label.csv',header=None)
            self.data = data.values.T
            self.label = label.values - 1
        else:
            data = pd.read_csv('./imbalance_data/test_feat.csv',header=None)
            self.data = data.values.T
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        if self.phase == 'test':
            return torch.Tensor(self.data[idx])
        else:
            return torch.Tensor(self.data[idx]), torch.LongTensor(self.label[idx])


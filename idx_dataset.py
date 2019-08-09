'''
Created on 2019年4月19日

@author: guolixiang
'''
from torch.utils.data import Dataset
import torch
import numpy as np


class IDXDataset(Dataset):
    def __init__(self, idx_list):
        super().__init__()
        self.idx_list = torch.from_numpy(idx_list).long()
        self.cnt = len(idx_list)

    def __len__(self):
        return self.cnt

    def __getitem__(self, idx):
        return self.idx_list[idx]
    
    
    
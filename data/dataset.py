import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

def split_dataset(dataset: np.ndarray, split_rate=0.8):
    '''
    groups=1
    :param dataset: x: (L, N, C)
    :param split_rate:切分数据集的比例
    :return: train: (L, N, C), test: (L, N, C)
    '''
    total_seq_len, num_nodes, _ = dataset.shape
    train_size = int(total_seq_len * split_rate)
    train_dataset, test_dataset = dataset[ 0:train_size, ...],dataset[train_size:, ...]

    return train_dataset, test_dataset


class SubwayDataset(Dataset):
    def __init__(self, dataset,time_dataset, seq_len,pred_len, feature_range=(-1, 1),**kwargs):
        '''
        :param dataset: x:(total_L, N, C)
        :param seq_len: length of split sequence
        :param pred_len:length of pred sequence
        :param feature_range: range of min_max scalar
        '''
        self.feature_range = feature_range
        self.pred_len = pred_len
        self.seq_len=seq_len
        self.mean=kwargs.get("mean")
        self.std=kwargs.get("std")
        self.max_values=kwargs.get("max")
        self.min_values = kwargs.get("min")

        assert len(self.feature_range) == 2 and self.feature_range[1] > self.feature_range[0]

        self.total_seq_len = len(dataset)
        self.dataset=dataset
        _,self.num_features,self.num_nodes=dataset.shape
        self.time_dataset=time_dataset


    def __getitem__(self, item):
        '''
        :param item: index
        :return: x: (C,L,N) and label：(C,N,L)
        '''
        x_end=item+self.seq_len
        y_end=x_end+self.pred_len
        x=self.dataset[item:x_end]
        y=self.dataset[x_end:y_end]
        x_time=self.time_dataset[item:x_end]
        y_time = self.time_dataset[x_end:y_end]
        x,y=torch.FloatTensor(x),torch.FloatTensor(y)
        x_time, y_time = torch.FloatTensor(x_time), torch.FloatTensor(y_time)

        return x,x_time,y,y_time

    def __len__(self):
        return len(self.dataset) - self.seq_len - self.pred_len








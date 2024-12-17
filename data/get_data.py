from data.data_process import load_data,get_time_features
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from torch.utils import data
from data.dataset import split_dataset,SubwayDataset


'''Slicing the dataset and creating the dataloader'''
def build_dataloader(args, test=False):
    '''Training set, validation set, test set = 6:2:2 or 7:1:2, training set and validation set are identically distributed, but training set and test set are not identically distributed.'''
    dataset,Time,adj = load_data(args)  # Get all the data records and the corresponding adjacency matrices
    Time=get_time_features(Time) #(total_len,N=1,C=5)

    total_len, num_nodes, num_features = dataset.shape  # shape(total_len,num_nodes,num_features)
    args.num_nodes = num_nodes
    args.num_features = num_features
    args.time_features = Time.shape[-1]
    print('Number of nodes：', args.num_nodes)
    print('Feature Dimension：', args.num_features)
    print('Temporal Feature Dimension：', args.time_features)

    if args.data_name == "PeMS-Bay" or args.data_name == "METR-LA":
        # Single cut training and test sets (cut on total length dimension)
        train_dataset,test_dataset = split_dataset(dataset,split_rate=0.8)
        Time = Time.reshape(-1, args.time_features, 1)  # (total_len,C=5,N=1)
        train_time_dataset, test_time_dataset = split_dataset(Time, split_rate=0.8)
        # Slice the training set and validation set as follows
        # Training set: validation set: test set = 7:1:2
        train_dataset,val_dataset=split_dataset(train_dataset,split_rate=0.875)
        train_time_dataset, val_time_dataset = split_dataset(train_time_dataset, split_rate=0.875)
    else:
        # Single cut training and test sets (cut on total length dimension)
        train_dataset,test_dataset = split_dataset(dataset,split_rate=0.8)
        Time = Time.reshape(-1, args.time_features, 1)  # (total_len,C=5,N=1)
        train_time_dataset, test_time_dataset = split_dataset(Time, split_rate=0.8)
        # Slice the training set and validation set as follows
        # Training set: validation set: test set = 6:2:2
        train_dataset,val_dataset=split_dataset(train_dataset,split_rate=0.75)
        train_time_dataset, val_time_dataset = split_dataset(train_time_dataset, split_rate=0.75)


    train_len=len(train_dataset)
    val_len=len(val_dataset)
    test_len=len(test_dataset)

    if args.data_name == "PeMS-Bay" or args.data_name == "METR-LA":
        # Normalization
        scaler = StandardScaler(with_mean=True, with_std=True)
        train_dataset = scaler.fit_transform(train_dataset.reshape(-1,num_features))
        val_dataset = scaler.transform(val_dataset.reshape(-1,num_features))
        test_dataset = scaler.transform(test_dataset.reshape(-1,num_features))

        train_dataset = np.transpose(train_dataset.reshape(train_len,num_nodes,num_features),(0,2,1))
        val_dataset = np.transpose(val_dataset.reshape(val_len,num_nodes,num_features),(0,2,1))
        test_dataset = np.transpose(test_dataset.reshape(test_len,num_nodes,num_features),(0,2,1))
        mean = scaler.mean_.reshape(1, num_features,1)
        std = scaler.scale_.reshape(1, num_features,1)

        train_dataset = SubwayDataset(train_dataset, train_time_dataset, args.seq_len, args.pred_len,std=std,mean=mean)
        val_dataset = SubwayDataset(val_dataset, val_time_dataset, args.seq_len, args.pred_len,std=std,mean=mean)
        test_dataset = SubwayDataset(test_dataset, test_time_dataset, args.seq_len, args.pred_len,std=std,mean=mean)

    elif args.data_name=="PEMS04"or args.data_name=="PEMS08":
        scaler = MinMaxScaler(feature_range=(-1, 1))
        train_dataset = scaler.fit_transform(train_dataset.reshape(-1,num_features))
        val_dataset = scaler.transform(val_dataset.reshape(-1,num_features))
        test_dataset = scaler.transform(test_dataset.reshape(-1,num_features))

        train_dataset = np.transpose(train_dataset.reshape(train_len,num_nodes,num_features),(0,2,1))
        val_dataset = np.transpose(val_dataset.reshape(val_len,num_nodes,num_features),(0,2,1))
        test_dataset = np.transpose(test_dataset.reshape(test_len,num_nodes,num_features),(0,2,1))

        min_values = scaler.data_min_.reshape(1, num_features,1)
        max_values = scaler.data_max_.reshape(1, num_features,1)
        train_dataset = SubwayDataset(train_dataset, train_time_dataset, args.seq_len, args.pred_len, max=max_values,min=min_values)
        val_dataset = SubwayDataset(val_dataset, val_time_dataset, args.seq_len, args.pred_len, max=max_values,min=min_values)
        test_dataset = SubwayDataset(test_dataset, test_time_dataset, args.seq_len, args.pred_len, max=max_values,min=min_values)
    else:
        assert print("Dataset normalization undefined")

    if not isinstance(adj, torch.Tensor):
        adj = torch.tensor(adj, dtype=torch.float32)
    adj.cuda()

    train_sampler, val_sampler,test_sampler = None,None,None

    if args.distributed and not test:
        train_sampler = data.DistributedSampler(train_dataset, seed=args.seed)
        train_batch_sampler = data.BatchSampler(train_sampler, args.batch_size, drop_last=True)
        train_dataloader = DataLoader(train_dataset, batch_sampler=train_batch_sampler,
                                           num_workers=args.num_workers, pin_memory=args.pin_memory)

        val_sampler = data.DistributedSampler(val_dataset, seed=args.seed)
        val_dataloader = DataLoader(test_dataset, args.batch_size, sampler=test_sampler,
                                          num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False)

        test_sampler = data.DistributedSampler(test_dataset, seed=args.seed)
        test_dataloader = DataLoader(test_dataset, args.batch_size, sampler=test_sampler,
                                          num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=False)
    else:

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                           drop_last=True,
                                           num_workers=args.num_workers,
                                           pin_memory=args.pin_memory)  # pin_memory默认为False，设置为TRUE会加速读取数据

        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                          drop_last=True,
                                          num_workers=args.num_workers, pin_memory=args.pin_memory)

        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                          drop_last=True,
                                          num_workers=args.num_workers, pin_memory=args.pin_memory)

    if args.data_name=="PeMS-Bay"or args.data_name=="METR-LA":
        mean,std=np.expand_dims(mean,axis=-1),np.expand_dims(std,axis=-1) #(B=1,C,N,L=1)
        train_dataloader.mean,val_dataloader.mean,test_dataloader.mean=mean,mean,mean
        train_dataloader.std, val_dataloader.std, test_dataloader.std = std, std, std

    elif args.data_name == "PEMS04" or args.data_name == "PEMS08":
        min, max = np.expand_dims(min_values, axis=-1), np.expand_dims(max_values, axis=-1)  # (B=1,C,N,L=1)
        train_dataloader.min, val_dataloader.min, test_dataloader.min = min, min, min
        train_dataloader.max, val_dataloader.max, test_dataloader.max = max, max, max
    return adj, train_dataloader, val_dataloader, test_dataloader, train_sampler, val_sampler, test_sampler






import pandas as pd
import numpy as np
import os
import csv
def get_adjacency_matrix(distance_df_filename, num_of_vertices):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''

    with open(distance_df_filename, 'r') as f:
        reader = csv.reader(f)
        header = f.__next__()
        edges = [(int(i[0]), int(i[1])) for i in reader] # Taking the first two from - to - indicates the neighboring information

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    return A

def load_adjacency_matrix(distance_df_filename,num_of_vertices):
    adj = pd.read_pickle(distance_df_filename)[2]
    assert num_of_vertices==adj.shape[0]==adj.shape[1] and len(adj.shape)==2
    # The following deals with the adjacency matrix, since the diagonal of the read adjacency matrix is 0 and is entitled.
    I=np.eye(num_of_vertices) # Diagonal matrix
    adj=np.where(adj!=0,1,adj) # Where it's not 0, just write 1, otherwise it stays the same value
    adj=adj-I # The diagonal becomes 0
    return adj

def process_time_seq_data(dataset_dir,dataset_dir_adj):
    tmp=np.load(dataset_dir,allow_pickle=True)
    data=tmp['data']#TODO Format of feature data (total_len,N,C)
    time=tmp['time']#TODO Data format for the time component (total_len,)
    assert data.shape[0]==time.shape[0]

    #TODO !!! The data is characterized by default with the first dimension being the time dimension, the second dimension being the number of nodes, and the third dimension being the feature dimension
    total_len,node_num,dim=data.shape
    time=time.reshape(-1,1)
    #TODO !!! The adjacency matrix obtained here is an unweighted adjacency matrix
    if dataset_dir_adj.split('.')[-1]=='csv':
        adj=get_adjacency_matrix(dataset_dir_adj,num_of_vertices=node_num)
    elif dataset_dir_adj.split('.')[-1]=='pkl':
        adj = load_adjacency_matrix(dataset_dir_adj, num_of_vertices=node_num)
    else:
        raise print('Path error in the adjacency matrix. There is no path with pathname {0}'.format(dataset_dir_adj))

    assert adj.shape[0]==adj.shape[1]==data.shape[1]

    return data,time,adj

def get_data_(dataset_dir,dataset_dir_adj):
    data,time, adj = process_time_seq_data(dataset_dir,dataset_dir_adj)

    return data,time,adj


'''Here is the read data code, the default input feature data are saved using an npz file'''
def load_data(args):
    tmp=os.path.join(args.dataset_dir,args.data_name)
    dataset_dir = os.path.join(tmp,args.data_name+'.npz')

    # The following code determines the type of storage file for the corresponding adjacency matrix
    if 'distance.csv'in os.listdir(tmp):
        dataset_dir_adj = os.path.join(tmp,'distance.csv')
    elif 'adj.pkl'in os.listdir(tmp):
        dataset_dir_adj = os.path.join(tmp, 'adj.pkl')
    else:
        raise print('No corresponding adj file')
    dataset,time, adj = get_data_(dataset_dir,dataset_dir_adj)
    assert len(dataset.shape)==3 and adj.shape[0]==adj.shape[1]

    return dataset,time, adj

'''Extract temporal features and normalize them'''
def get_time_features(time):
    dt = pd.DataFrame({'dates': time.flatten()})
    # Converting Date Columns to Pandas Timestamp Objects
    dt['dates'] = pd.to_datetime(dt['dates'])
    dayofyear = dt['dates'].dt.dayofyear.values.reshape(-1, 1)  # The n day of the year
    dayofyear = (dayofyear - 1) / (365 - 1) - 0.5  # Normalized to -0.5 to +0.5
    dayofmonth = dt['dates'].dt.day.values.reshape(-1, 1)  # The n day of the mouth.
    dayofmonth = (dayofmonth - 1) / (31 - 1) - 0.5  # Normalized to -0.5 to +0.5
    dayofweek = dt['dates'].dt.dayofweek.values.reshape(-1, 1)
    dayofweek = (dayofweek - 0) / (6 - 0) - 0.5  # Normalized to -0.5 to +0.5
    hourofday = dt['dates'].dt.hour.values.reshape(-1, 1)  # The n hour of the day.
    hourofday = (hourofday - 0) / (23 - 0) - 0.5  # Normalized to -0.5 to +0.5
    minofhour = dt['dates'].dt.minute.values.reshape(-1, 1)
    minofhour = (minofhour - 0) / (59 - 0) - 0.5  # Normalized to -0.5 to +0.5
    Time = np.concatenate((dayofyear, dayofmonth, dayofweek, hourofday, minofhour), axis=-1)  # TODO Time feature dimension = 5
    time_feature = Time.shape[-1]
    Time = Time.reshape((-1, 1, time_feature))  # (total_len,N=1,C=5)，
    return Time





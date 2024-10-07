import torch
from torch.utils.data import DataLoader
from data.dataset import split_dataset,SubwayDataset
from data.data_process import *
import numpy as np
from fastdtw import fastdtw
from tqdm import tqdm
import torch.nn as nn
from tslearn.clustering import TimeSeriesKMeans, KShape
from sklearn.preprocessing import StandardScaler
class get_dtw(nn.Module):
    def __init__(self, config,dtw=True,pattern_keys=True):
        super().__init__()
        # self.parameters_str = \
        #     str(config.data_name) + '_' + str(config.seq_len) + '_' + str(config.pred_len) + '_' \
        #     + str(self.train_rate) + '_' + str(self.part_train_rate) + '_' + str(self.eval_rate) + '_' + str(
        #         self.scaler_type) + '_' \
        #     + str(self.batch_size) + '_' + str(self.load_external) + '_' + str(self.add_time_in_day) + '_' \
        #     + str(self.add_day_in_week) + '_' + str(self.pad_with_last_sample) + '_' + str("".join(self.data_col))
        # self.type_short_path = 'hop'
        # self.cache_file_name = os.path.join('./libcity/cache/',
        #                                     'pdformer_point_based_{}.npz'.format(self.parameters_str))
        self.config=config
        df,Time, _ = load_data(config)
        Time = get_time_features(Time)  # (total_len,N=1,C=5)，与dataset的形状一样
        Time = Time.reshape(-1, config.time_features, 1)  # (total_len,C=5,N=1)
        self.df = df
        self.time=Time
        self.output_dim=config.output_dim
        self.points_per_hour = config.points_per_hour
        self.time_intervals=3600//config.points_per_hour # 一个小时有3600秒除于一个小时有多少个记录点=记录点间秒数差距多少
        if dtw==True:
            self.dtw_matrix = self._get_dtw() # 得到节点间DTW的距离矩阵
        self.points_per_day = 24 * 3600 // self.time_intervals #一天有24*3600秒除于记录点秒数差距=一天内有多少的记录点
        self.cand_key_days=config.cand_key_days = 14
        self.s_attn_size =config.s_attn_size=  3
        self.n_cluster =config.n_cluster=  16
        self.cluster_max_iter=config.cluster_max_iter = 5
        self.cluster_method =config.cluster_method="kshape"
        self.dataset=config.data_name
        if pattern_keys==True:
            self.pattern_keys=self._get_pattern_key() # 得到质心

    '''The distance matrix of DTW is obtained'''
    def _get_dtw(self):
        cache_path = './datasets/cache/dtw_' + self.config.data_name + '.npy'
        if not os.path.exists(cache_path):
            print(f'Calculate the inter-node dtw distance since there is no file with path corresponding to {cache_path}')
            df=self.df
            data_mean = np.mean(  # TODO this is done on the whole dataset to calculate the distance between nodes
                [df[24 * self.points_per_hour * i: 24 * self.points_per_hour * (i + 1)]
                 # Here the average of the features corresponding to each site in a day is calculated
                 for i in range(df.shape[0] // (24 * self.points_per_hour))], axis=0)  # data_mean(total,N,C)
            _, self.num_nodes, self.feature = df.shape
            dtw_distance = np.zeros((self.num_nodes, self.num_nodes))
            for i in tqdm(range(self.num_nodes)):
                for j in range(i, self.num_nodes):
                    dtw_distance[i][j], _ = fastdtw(data_mean[:, i, :], data_mean[:, j, :], radius=6) # Calculate the corresponding dtw distance between each site day
            for i in range(self.num_nodes):  # This is equivalent to constructing a symmetrical array
                for j in range(i):  # Calculate the dtw distance between each station day.
                    dtw_distance[i][j] = dtw_distance[j][i]
            np.save(cache_path, dtw_distance)

        dtw_matrix = np.load(cache_path)
        print('Load DTW matrix from {}'.format(cache_path))
        return dtw_matrix

    def get_seq_traindata(self):
        train_dataset, _ = split_dataset(self.df, split_rate=0.8)   # Slicing training and test sets alone (slicing in time-step dimension)
        train_time_dataset, _ = split_dataset(self.time, split_rate=0.8)

        _, num_nodes, num_features = train_dataset.shape
        scaler = StandardScaler(with_mean=True, with_std=True)
        train_dataset = scaler.fit_transform(train_dataset.reshape(len(train_dataset), -1))  # Integrate sites and features
        train_dataset = train_dataset.reshape(len(train_dataset), num_features, num_nodes, )
        train_dataset = SubwayDataset(train_dataset,train_time_dataset, self.config.seq_len, self.config.pred_len)
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False,drop_last=False)
        self.train_dataset=[]
        for batch_x,_,batch_y,_ in train_dataloader:
            self.train_dataset.append(batch_x)
        self.train_dataset=torch.concat(self.train_dataset,dim=0)
        return self.train_dataset

    def _get_pattern_key(self):
        self.pattern_key_file = os.path.join(
            # This data should be the data corresponding to each cluster center after clustering
            '. /datasets/cache/', 'pattern_keys_{}_{}_{}_{}_{}_{}'.format(
                self.cluster_method, self.dataset, self.cand_key_days, self.s_attn_size, self.n_cluster,
                self.cluster_max_iter))
        if not os.path.exists(self.pattern_key_file + '.npy'):
            print(f'Since the file with address {self.pattern_key_file} does not exist, the corresponding clustered center of mass data is computed')

            self.train_dataset = self.get_seq_traindata()  # get training set train_dataset(total_len,Len,dim,N)

            cand_key_time_steps = self.cand_key_days * self.points_per_day  # 14* how many record points per day
            pattern_cand_keys = (
                self.train_dataset[:cand_key_time_steps, :self.s_attn_size, :self.output_dim, :].permute(0, 3, 1,
                                                                                                         2)
                .reshape(-1, self.s_attn_size,
                         self.output_dim))  # TODO This is clustering on the training set only, and the training set is sliced by seq_len.
            print('Clustering...')
            if self.cluster_method == 'kshape':  # This uses autocorrelation to calculate distance, otherwise it is the same as Kmeans.

                km = KShape(n_clusters=self.n_cluster, max_iter=self.cluster_max_iter).fit(pattern_cand_keys)
            else: # Kmeans
                km = TimeSeriesKMeans(n_clusters=self.n_cluster, metric="softdtw", max_iter=self.cluster_max_iter).fit(
                    pattern_cand_keys)
            self.pattern_keys = km.cluster_centers_
            np.save(self.pattern_key_file, self.pattern_keys)
            print("Saved at file " + self.pattern_key_file + ".npy")
        else:
            self.pattern_keys = np.load(self.pattern_key_file + ".npy")  # (16,3,1),16-->簇数，3-->Attention的类别数，1-->特征数
            print("Loaded file " + self.pattern_key_file + ".npy")

        return self.pattern_keys







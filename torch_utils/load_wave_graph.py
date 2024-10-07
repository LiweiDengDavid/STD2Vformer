import os
import math
import numpy as np
import torch
from tqdm import tqdm
import scipy.sparse as sp
from fastdtw import fastdtw
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

def laplacian(W):
    """Return the Laplacian of the weight matrix."""
    # Degree matrix.
    if isinstance(W,torch.Tensor):
        W=np.array(W.detach().cpu())
    d = W.sum(axis=0)
    # Laplacian matrix.
    d = 1 / np.sqrt(d)
    D = sp.diags(d, 0)
    I = sp.identity(d.size, dtype=W.dtype)
    L = I - D * W * D
    return L

def largest_k_lamb(L, k):
    lamb, U = sp.linalg.eigsh(L, k=k, which='LM')
    return (lamb, U)

def get_eigv(adj,k):
    L = laplacian(adj)
    eig = largest_k_lamb(L,k)
    return eig

# Construct adj matrix based on similarity of time series (measure DTW distance)
def construct_tem_adj(data, num_node):
    data_mean = np.mean([data[24*12*i: 24*12*(i+1)] for i in range(data.shape[0]//(24*12))], axis=0) # This one calculates the average for the day #
    data_mean = data_mean.squeeze().T
    dtw_distance = np.zeros((num_node, num_node))
    for i in tqdm(range(num_node)):
        for j in range(i, num_node):
            dtw_distance[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0] # Calculate the dtw distance between node i and node j
    for i in range(num_node):
        for j in range(i):
            dtw_distance[i][j] = dtw_distance[j][i] # 这个距离是对称的

    nth = np.sort(dtw_distance.reshape(-1))[
        int(np.log2(dtw_distance.shape[0])*dtw_distance.shape[0]):
        int(np.log2(dtw_distance.shape[0])*dtw_distance.shape[0])+1] # NlogN edges to get thresholds
    tem_matrix = np.zeros_like(dtw_distance)
    tem_matrix[dtw_distance <= nth] = 1 # 1 for values less than this threshold, 0 for values greater than this threshold.
    tem_matrix = np.logical_or(tem_matrix, tem_matrix.T) # Logic and change into symmetric matrices
    return tem_matrix

def loadGraph(adj, temporal_graph, dims, data):
    # calculate spatial and temporal graph wavelets
    adj = adj + np.eye(adj.shape[0])
    if os.path.exists(temporal_graph+".npy"):
        tem_adj = np.load(temporal_graph+".npy")
    else:
        tem_adj = construct_tem_adj(data, adj.shape[0])
        np.save(temporal_graph, tem_adj)
    spawave = get_eigv(adj, dims) # This is the adj of the diagram of space
    temwave = get_eigv(tem_adj, dims) # This one is constructed from the dtw distance of the time series-adj

    # derive neighbors
    sampled_nodes_number = int(math.log(adj.shape[0], 2))
    graph = csr_matrix(adj) # Representation of graph sparsification
    dist_matrix = dijkstra(csgraph=graph) # Calculate the shortest path of the graph and return a 2D matrix
    dist_matrix[dist_matrix==0] = dist_matrix.max() + 10  # A shortest path of 0 indicates a possible non-connection, so a large value is appended
    localadj = np.argpartition(dist_matrix, sampled_nodes_number, -1)[:, :sampled_nodes_number] # Get the first sampled_nodes_number of points with the smallest distance in each row

    return localadj, spawave, temwave

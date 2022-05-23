import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from scipy.signal import butter,filtfilt, zpk2sos, sosfilt
from scipy.stats.stats import pearsonr
from Utils.CRASH_loader import *

import ipdb

class StandardScaler():
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(
        d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(
        d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            pickle_data = u.load()
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def mod_adj(adj_mx, adjtype):
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(
                                                  np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return adj

def load_adj(pkl_filename, adjtype):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    adj = mod_adj(adj_mx, adjtype)
    return sensor_ids, sensor_id_to_ind, adj

def inverse_sliding_window(li, K=None):
    '''
    takes in a list, each with dimension [num_window, num_nodes, window_width]
    with stride (each window's starting index discrepancy) K
    return a list, each with dimension [num_nodes, num_timesteps]
    ***** The overlapped portion are averaged *****
    '''
    def _rev(a, _K):
        assert len(a.shape) == 3
        num_window, num_nodes, width = a.shape
        num_t = width + (num_window - 1) * _K

        a = a.transpose(0, 2, 1)
        idxer = np.arange(width)[None, :] + np.arange(0,
                                    num_t-width+1, _K)[:, None]
        rev = np.zeros((num_nodes, num_t))
        for l in range(num_t):
            rev[:, l] = a[idxer == l].mean(0)
        return rev

    ret = []
    if K is None:
        K = [1]*len(li)
    else:
        assert len(li) == len(K)
    for i in range(len(li)):
        ret.append(_rev(li[i], K[i]))
    return ret

def butter_lowpass_filter(data, cutoff, fs, order=6):
    nyq = 0.5*fs
    normal_cutoff = cutoff / nyq

    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def get_cc(pred, real):
    assert pred.shape == real.shape
    if len(pred.shape) == 2: # batch 1 case
        pred = pred[None,...]
        real = real[None,...]
    cc = []
    p_min = 1
    p_max = 0
    for i in range(len(pred)):
        cur_cc = []
        for node_i in range(pred.shape[2]):
                r, pval = pearsonr(pred[i, :, node_i].detach().cpu().numpy(),
                                   real[i, :, node_i].detach().cpu().numpy())        
                cur_cc.append(r)
                if pval < p_min:
                    p_min = pval
                if pval > p_max:
                    p_max = pval
        cc.append(sum(cur_cc) / len(cur_cc))
    cc = sum(cc) / len(cc)
    return cc, p_min, p_max

def entropy(_S):
    '''for a more clearly defined cluster
    assign each node only to few clusters (preferablly 1)
    '''
    out = 0
    for S in _S: #bcnt
        p = torch.softmax(S, dim=1)
        p = p.permute(1,2,3,0)
        p = p.reshape(p.shape[0],-1)
        p = p.transpose(0,1) # p: (bnt,k)
        out += torch.distributions.Categorical(probs=p).entropy().mean()
    return out/len(S)

def entropy2(_S):
    '''to avoid a same assignment pattern of all the nodes
    maximize the "cluster-mass" entropy across nodes --> minimize -H
    '''
    out = 0
    for S in _S: #bcnt
        p = torch.softmax(S, dim=1).sum(2)
        p = p.transpose(1,2)
        p = p.reshape(-1, p.shape[-1]) # p: (bt, k)
        out -= torch.distributions.Categorical(probs=p).entropy().mean()
    return out/len(S)

def L2(S, A): 
    ''' Minimize A-SS^T, nearby nodes should be pooled together
       - S: a list of [b,cluster,node,t]
    '''
    out = 0
    for p in S:
        p = torch.softmax(p, dim=1)
        out += (torch.einsum('bcnt,bckt->bnkt',(p,p)) 
                - A[...,None]).pow(2.0).sum() / len(p)
    return out/len(S)

def T_constraint_out(y, frame = 5): 
    '''
       - y: [b, t, n]
       - frame: window length to calculate the std to restrict difference 
                in T dim
    '''
    t_len = y.shape[1]
    out = 0
    y = y.squeeze()
    for i in range(t_len - frame):       
        v = y[:, i + np.arange(frame), :].std()
        out += v
    return out/(t_len - frame)

def T_constraint(S, window_ratio=8): 
    '''
       - y: [b, c, n, t]
    '''
    ret = 0.0
    for y in S:
        t_len = y.shape[-1]
        # window length for penalization
        frame = t_len // window_ratio
        if frame:
            out = 0.0
            for i in range(t_len - frame):
                v = y[..., i + np.arange(frame)].std()
                out += v
            ret += out/(t_len - frame)
    return ret

def fourier_mapping(x, B):
    '''Fourier feature mapping'''
    if B is None:
        return x
    else:
        x_proj = torch.einsum('ncvl,wc->nwvl',(2.*math.pi*x,B))
        # x_proj = (2.*math.pi*x) @ B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=1)

def batch_corrcoef(x):
    ''' 
    Take in input with size (B, N, p), 
    return batch correlation coefficient matrix with size (B, N, N)
    '''
    # calculate covariance matrix of rows
    mean_x = torch.mean(x, -1, keepdim=True)
    xm = x.sub(mean_x.expand_as(x))
    c = torch.matmul(xm, xm.transpose(-1,-2))
    c = c / (x.size(-1) - 1)
    # normalize covariance matrix
    d = torch.diagonal(c, dim1=-2, dim2=-1)
    stddev = torch.pow(d, 0.5)[...,None]
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).transpose(-1,-2))

    # clamp between -1 and 1
    # probably not necessary but numpy does it
    c = torch.clamp(c, -1.0, 1.0)
    # normalize per row to serve as a transition matrix
    # c = c / c.sum(2,keepdim=True)
    return c * 0.01

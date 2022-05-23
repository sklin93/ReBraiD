import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
from Utils.util import *
import ipdb

EPS = 1e-15

def CausalConv2d(in_channels, out_channels, kernel_size,
                 dilation=(1,1), **kwargs):
    pad = (kernel_size[1] - 1) * dilation[1]
    return nn.Conv1d(in_channels, out_channels, kernel_size,
                     padding=(0, pad), dilation=dilation, **kwargs)

class in_cluster_smoothing(nn.Module):
    """ inner cluster smoothing, with t axis. """
    def __init__(self, noT=False):
        super().__init__()
        self.noT = noT

    def forward(self, x, s, viz=False, support=None): 
        '''
        - x: bdnt
        - s: bcnt
        - support: (b, num_supports, n, n)
        '''
        s = torch.softmax(s, dim=1)
        if self.noT:
            x = torch.einsum('bfnt,bcn->bfct', x, s).contiguous()
        else:
            x = torch.einsum('bfnt,bcnt->bfct', x, s).contiguous() # bdct 
        
        _s = torch.softmax(s, dim=2) # bcnt
        if self.noT:
            out = torch.einsum('bfct,bcn->bfnt', x, _s).contiguous()
        else:
            out = torch.einsum('bfct,bcnt->bfnt', x, _s).contiguous() # bdnt

        return out


class diff_pool(nn.Module):
    def __init__(self, noT=False):
        super().__init__()
        self.noT = noT

    def forward(self, x, s, viz=False, support=None): 
        s = torch.softmax(s, dim=1)
        if self.noT:
            x = torch.einsum('bfnt,bcn->bfct', x, s).contiguous()
            # coarsen supports
            _a = torch.einsum('bcn,bwnm,bkm->bwck', s, support, s)
        else:
            x = torch.einsum('bfnt,bcnt->bfct', x, s).contiguous() # bdnt
            # coarsen supports
            _a = torch.einsum('bcnt,bwnm,bkmt->bwck', s, support, s)

        return x, _a


class nconv2(nn.Module):
    def __init__(self):
        super(nconv2,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,nwv->ncwl',(x,A))
        return x.contiguous()


class gcn2(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super().__init__()
        self.nconv = nconv2()
        c_in = (order * support_len + 1) * c_in
        self.mlp = torch.nn.Conv2d(c_in, c_out,
            kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        '''
        - support: (b, num_supports, n, n)
        - x: bdnt'''
        out = [x]
        for support_i in range(support.shape[1]):
            a = support[:, support_i]
            x1 = self.nconv(x, a)
            out.append(x1)
            # k-hop neighbors
            for k in range(2, self.order + 1):
                x2 = F.relu(self.nconv(x1, a))
                out.append(x2)
                x1 = x2
        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, device,
        batch_size=16, concat=True):
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(batch_size, in_features,
            out_features))).to(device)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(batch_size,
            2*out_features, 1))).to(device)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.einsum('bfnt, bfc ->bcnt',(h,self.W))
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.einsum('bfnmt, bfa -> banmt',
            a_input, self.a).squeeze(1))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj.unsqueeze(3) > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.einsum('bnmt, bcmt ->bcnt',(attention,Wh))

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        '''
        - Wh: bdnt

        Below, two matrices are created that contain embeddings in their rows
        in different orders. (e stands for embedding)
        These are the rows of the first matrix (Wh_repeated_in_chunks): 
        e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        
        These are the rows of the second matrix (Wh_repeated_alternating): 
        e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN 
        '----------------------------------------------------' -> N times
        '''        
        B = Wh.size()[0]
        T = Wh.size()[-1]
        N = Wh.size()[2]


        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=2)
        Wh_repeated_alternating = Wh.repeat(1, 1, N, 1)

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks,
            Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (B, 2 * out_features, N*N, T)

        return all_combinations_matrix.view(B, 2 * self.out_features, N, N, T)

    def __repr__(self):
        return (self.__class__.__name__ + ' (' + str(self.in_features) + ' -> '
               + str(self.out_features) + ')')


class gcn3(nn.Module): #GCN module with GAT layer mechanism
    def __init__(self, c_in, c_out, dropout, device, support_len=3, order=2,
        alpha=0.2, batch_size=16):
        super().__init__()

        c_in = (order * support_len + 1) * c_in
        self.mlp = torch.nn.Conv2d(c_in, c_out,
            kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.dropout = dropout
        self.order = order
        self.attentions = [GraphAttentionLayer(c_in, c_in, dropout=dropout,
            alpha=alpha, batch_size = batch_size, concat=True, device=device
            ) for _ in range(support_len)]
        self.attentionsK = [GraphAttentionLayer(c_in, c_in, dropout=dropout,
            alpha=alpha, batch_size = batch_size, concat=True, device=device
            ) for _ in range(support_len*(self.order-1))]

    def forward(self,x, *support):
        count = 0
        out = [x]
        for idx, a in enumerate(support):
            # a: [64, 80, 80], x:[64, 32, 80, 15]
            x1 = self.attentions[idx](x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = F.relu(self.attentionsK[count](x1,a))
                out.append(x2)
                x1 = x2
                count += 1
        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class TemporalAttention(nn.Module): 
    """attn for temporal features along temp axis.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1,
            (1, kernel_size), padding=(0, padding), bias=False)

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avgout, maxout], dim=1)
        scale = self.conv(scale)
        return torch.sigmoid(scale)*x


class ChannelAttention(nn.Module):
    """attn for temporal features along channel axis.
    """
    def __init__(self, in_channels, reduction_num, out_channels):
        super().__init__()
        self.mlp1 = nn.Linear(in_channels, int(in_channels/reduction_num))
        self.mlp2 = nn.Linear(int(in_channels/reduction_num), out_channels)
    
    def forward(self, x):
        avg_x = x.mean((2,3))
        avg_x = self.mlp1(avg_x)
        avg_x = self.mlp2(avg_x)
        
        max_x = x.amax((2,3))
        max_x = self.mlp1(max_x)
        max_x = self.mlp2(max_x)

        return torch.sigmoid(avg_x + max_x)[...,None,None] * x


class pool(torch.nn.Module): # strictly pool at the end
    def __init__(self,in_channels,num_nodes_eeg,dropout,support_len,
                 non_linearity=torch.tanh):
        super().__init__()
        self.in_channels = in_channels
        self.score_layer = gcn2(in_channels, 1, dropout, support_len)
        self.num_nodes_eeg = num_nodes_eeg
        self.non_linearity = non_linearity
    def forward(self, x, *support):
        #x = x.unsqueeze(-1) if x.dim() == 1 else x
        score = self.score_layer(x,*support)
        _,perm = torch.topk(score.squeeze(), self.num_nodes_eeg)
        x = x.permute(0,2,1,3)
        perm = torch.unsqueeze(perm, 2)
        perm = torch.unsqueeze(perm, 3)
        x = torch.gather(x, 1, perm.expand(-1,-1,x.size(2),x.size(3)))
        x = x.permute(0,2,1,3)
        perm = perm.permute(0,2,1,3)
        score = torch.gather(score, 2, perm)
        #find way to index topk nodes from x and from score layer
        x = x * self.non_linearity(score)
        return x

################ MODELS ################

class mls_gat_classifier(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports_len=0,
                 batch_size=32, gcn_bool=True, addaptadj=True, in_dim=2,
                 seq_len=12, residual_channels=32,
                 dilation_channels=32, skip_channels=256, end_channels=512,
                 kernel_size=3, blocks=4, layers=2, hidden_dim=64,
                 noT=False, pool=True, use_corrcoef=False, out_class=6):

        super().__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.device = device
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.noT = noT
        self.pool = pool
        self.use_corrcoef = use_corrcoef
        self.out_class = out_class

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.temporalAttn = nn.ModuleList()
        # self.channelAttn = nn.ModuleList()

        if pool:
            self.pool_bn = nn.ModuleList()
            self.gconv_pool = nn.ModuleList()
            # self.pool_bn2 = nn.ModuleList()
            if noT:
                self.pool_conv = nn.ModuleList()
                self.pool_bn_t = nn.ModuleList()
        
        if self.gcn_bool and self.addaptadj:
            self.nodevec = nn.Parameter(torch.randn(seq_len, 5).to(self.device), 
                                    requires_grad=True).to(self.device)

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))

        receptive_field = 1
        multi_factor = kernel_size #2
        clusters = [100,50,17,7]

        for b in range(blocks):
            additional_scope = kernel_size - 1
            for i in range(layers):
                self.filter_convs.append(CausalConv2d(
                    in_channels=residual_channels,
                    out_channels=dilation_channels, 
                    kernel_size=(1, kernel_size), stride=(1, 2), dilation=(1, 1)
                    ))

                self.gate_convs.append(CausalConv2d(
                    in_channels=residual_channels,
                    out_channels=dilation_channels,
                    kernel_size=(1, kernel_size), stride=(1, 2), dilation=(1, 1)
                    ))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(
                    in_channels=dilation_channels,
                    out_channels=residual_channels,
                    kernel_size=(1, 1), stride=(1, 2)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))

                # self.temporalAttn.append(TemporalAttention(kernel_size=7))
                # self.channelAttn.append(ChannelAttention(residual_channels,
                                    # 8, dilation_channels))

                receptive_field += additional_scope
                additional_scope *= multi_factor
                if self.gcn_bool and i == self.layers-1: 
                    # 1 gcn per block
                    self.gconv.append(gcn2(dilation_channels, residual_channels,
                                           dropout, support_len=supports_len))
                    self.bn.append(nn.BatchNorm2d(residual_channels, eps=EPS))

                    if pool and i == self.layers-1:
                        if b == blocks-1: 
                            # only apply nodes attn in last gconv pooling block
                            self.gconv_pool.append(gcn3(
                                dilation_channels, clusters[b], dropout,
                                batch_size=batch_size, support_len=supports_len,
                                device=device))
                        else:
                            self.gconv_pool.append(gcn2(dilation_channels,
                                clusters[b], dropout, support_len=supports_len))
                        self.pool_bn.append(nn.BatchNorm2d(clusters[b],
                                                           eps=EPS))
                        if noT:
                            self.pool_conv.append(nn.Conv2d(t_list[b], 1, 1,
                                                            bias=False))
                        self.in_cluster_smoothing = in_cluster_smoothing(noT)
      
        self.receptive_field = receptive_field

        self.cl1 = nn.Linear(skip_channels, skip_channels//4)
        self.cl2 = nn.Linear(skip_channels//4, out_class)
        self.cl_node1 = nn.Linear(num_nodes, num_nodes//2)
        self.cl_node2 = nn.Linear(num_nodes//2, 1)
        # self.lstm = nn.LSTM(skip_channels//4, out_class, num_layers=2,
                            # bidirectional=True, dropout=dropout)

    def get_adp(self, input, supports=None, viz=False):
        # generate adaptive adjacency matrix based on current inputs
        nodevec = torch.einsum('ncl,lv->ncv', (input[:, 0, ...], self.nodevec))
        adp = F.softmax(F.relu(torch.matmul(nodevec,
                                            nodevec.transpose(1, 2))), dim=2)
        if viz: 
            plt.imshow(adp[0].detach().cpu().numpy(), cmap='Blues')
            plt.show()
            # plot learned theta
            plt.imshow(self.nodevec.detach().cpu().numpy())
            plt.show()
            ipdb.set_trace()
            # adp.sum(1)
            # _, idx = torch.sort(adp.sum(1)) 
            # top10 = idx.cpu().numpy()[:,::-1][:,:10]

        if len(supports) > 0:
            new_supports = supports + [adp]
        else:
            new_supports = [adp]
        if self.use_corrcoef:
            new_supports = new_supports + [batch_corrcoef(input.squeeze())]
        return new_supports

    def tcn(self, i, x):
        x = self.channelAttn[i](x)
        x = self.temporalAttn[i](x)
        # dilated causal convolution
        _filter = self.filter_convs[i](x)
        _filter = F.relu(_filter[..., :-self.filter_convs[i].padding[1]])

        return _filter

    def gcn(self, i, x, supports, viz=False):
        # # gcn part, also handles diffpool for each block
        # x1 = self.gconv[i](x, *supports)
        # x1 = self.bn[i](x1) # bdnt

        # 1 gcn per block (to reduce #gcn layers)
        if i % self.layers == self.layers-1:
            x1 = self.gconv[i // self.layers](x, *supports)
            x1 = self.bn[i // self.layers](x1) # bdnt
        else:
            return x

        if self.pool and i % self.layers == self.layers-1:
            S = self.gconv_pool[i // self.layers](x, *supports)
            S = self.pool_bn[i // self.layers](S)
            if self.noT:
                S = self.pool_conv[i // self.layers](
                    S.transpose(1,3)).squeeze().transpose(1,2)
                # S = self.pool_bn_t[i // self.layers](S)
            self.S.append(S)
            return self.in_cluster_smoothing(x1, S, viz, *supports)
            # x = self.pool_bn2[i // self.layers](x)
            # supports.append(new_adj)
            # self._new_adj = new_adj
        else:
            return x1

    def forward(self, input, supports=None, aptinit=None, viz=False):
        if self.gcn_bool and self.addaptadj:
            if supports is None:
                supports = []
        x = self.start_conv(input)
        skip = 0
        self.S = []

        # calculate the current adaptive adj matrix once per iteration
        if self.gcn_bool and self.addaptadj:
            supports = self.get_adp(input, supports, viz)
        self.supports = supports

        for i in range(self.blocks * self.layers):       
            residual = x
            # tcn
            x = self.tcn(i, x)
            # t_rep = x
            # gcn
            if self.gcn_bool:
                x = self.gcn(i, x, supports, viz)
            
            residual = self.residual_convs[i](residual)
            x = x + residual #+t_rep

            # parametrized skip connection
            s = self.skip_convs[i](x)
            try:
                if s.size(-1)*2 == skip.size(-1):
                    skip = F.max_pool2d(skip,(1,2))
                else:
                    skip = skip[..., -s.size(-1):]
            except:
                skip = 0
            skip = s + skip

        x = F.relu(skip)

        x = self.cl_node1(x.squeeze())
        x = self.cl_node2(F.relu(x)).transpose(1,2)
        x = F.relu(self.cl1(x))
        # x, hidden = self.lstm(x) # [16 33 128]
        # x = F.relu(x)
        # x = F.max_pool2d(x, (12,1))
        # x, hidden = self.lstm(x)
        x = self.cl2(x)
        return x.squeeze()


class mls_classifier(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports_len=0,
                 batch_size=32, gcn_bool=True, addaptadj=True, in_dim=2,
                 seq_len=12, residual_channels=32,
                 dilation_channels=32, skip_channels=256, end_channels=512,
                 kernel_size=3, total_layers=8, block_layers=2,
                 hidden_dim=64, noT=False, pool=True,
                 use_corrcoef=False, out_class=6, parametrized_residual=True):

        super().__init__()
        self.dropout = dropout
        self.total_layers = total_layers
        self.block_layers = block_layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.device = device
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.noT = noT
        self.pool = pool
        self.use_corrcoef = use_corrcoef
        self.out_class = out_class
        self.parametrized_residual = parametrized_residual

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        if self.parametrized_residual:
            self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        if pool:
            self.pool_bn = nn.ModuleList()
            self.gconv_pool = nn.ModuleList()
            if noT:
                self.pool_conv = nn.ModuleList()
                self.pool_bn_t = nn.ModuleList()
        
        if self.gcn_bool and self.addaptadj:
            self.nodevec = nn.Parameter(torch.randn(seq_len, 5).to(self.device), 
                                    requires_grad=True).to(self.device)

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))

        receptive_field = 1
        multi_factor = kernel_size #2
        # hard coded cluster numbers
        if seq_len == 32:
            clusters = [120, 80, 60, 40, 17, 7, 3]
        else:
            clusters = [100,50,17,7]

        # maker sure number of clusters is >= number of blocks
        if pool:
            assert len(clusters) >= total_layers // block_layers, (
                                        'need more cluster number')

        for i in range(total_layers):
            b = i // block_layers
            # if i % block_layers == 0:
            #     additional_scope = kernel_size - 1
            # # dilated convolutions
            # self.filter_convs.append(CausalConv2d(
            #     in_channels=residual_channels, out_channels=dilation_channels,
            #     kernel_size=(1, kernel_size), stride=(1, kernel_size),
            #     dilation=(1, 1)))

            # self.gate_convs.append(CausalConv2d(
            #     in_channels=residual_channels, out_channels=dilation_channels,
            #     kernel_size=(1, kernel_size), stride=(1, kernel_size),
            #     dilation=(1, 1)))
            
            # tcn
            self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                               out_channels=dilation_channels,
                                               kernel_size=(1, kernel_size),
                                               stride=(1, kernel_size)))

            self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                             out_channels=dilation_channels,
                                             kernel_size=(1, kernel_size), 
                                             stride=(1, kernel_size)))                

            if self.parametrized_residual:
                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(
                    in_channels=dilation_channels,
                    out_channels=residual_channels,
                    kernel_size=(1, kernel_size), stride=(1, kernel_size)))
            else:
                assert dilation_channels == residual_channels, (
                    'dilation_channels and residual_channels not the same, ' +
                    'need an extra linear layer.')

            # 1x1 convolution for skip connection
            self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                             out_channels=skip_channels,
                                             kernel_size=(1, 1)))
            # receptive_field += additional_scope
            # additional_scope *= multi_factor
            if self.gcn_bool and i % block_layers == block_layers-1:
                self.gconv.append(gcn2(dilation_channels, residual_channels,
                                       dropout, support_len=supports_len))
                self.bn.append(nn.BatchNorm2d(residual_channels, eps=EPS))

                if pool and i % block_layers == block_layers - 1:
                    self.gconv_pool.append(gcn2(dilation_channels, clusters[b],
                                            dropout, support_len=supports_len))
                    self.pool_bn.append(nn.BatchNorm2d(clusters[b], eps=EPS))
                    if noT:
                        self.pool_conv.append(nn.Conv2d(t_list[b], 1, 1,
                                                        bias=False))
                    self.in_cluster_smoothing = in_cluster_smoothing(noT)
        # self.receptive_field = receptive_field

        self.cl1 = nn.Linear(skip_channels, skip_channels//4)
        self.cl2 = nn.Linear(skip_channels//4, out_class)
        self.cl_node1 = nn.Linear(num_nodes, num_nodes//2)
        self.cl_node2 = nn.Linear(num_nodes//2, 1)

    def get_adp(self, input, supports=None, viz=False):
        if self.addaptadj:
            # generate adaptive adjacency matrix based on current inputs
            nodevec = torch.einsum('ncl,lv->ncv', (input[:, 0, ...],
                                    self.nodevec))
            adp = F.softmax(F.relu(torch.matmul(nodevec,
                        nodevec.transpose(1, 2))), dim=2)
            if viz: 
                plt.imshow(adp[0].detach().cpu().numpy(), cmap='Blues')
                plt.show()
                # plot learned theta
                plt.imshow(self.nodevec.detach().cpu().numpy())
                plt.show()
                ipdb.set_trace()
                # adp.sum(1)
                # _, idx = torch.sort(adp.sum(1)) 
                # top10 = idx.cpu().numpy()[:,::-1][:,:10]

            if supports is not None:
                new_supports = torch.cat([supports, adp[:, None, ...]], axis=1)
            else:
                new_supports = adp[:, None, ...]
            if self.use_corrcoef:
                new_supports = torch.cat([new_supports, batch_corrcoef(
                               input.squeeze())[:, None, ...]], axis=1)
        else:
            if supports is not None:
                new_supports = torch.cat([supports, batch_corrcoef(
                           input.squeeze())[:, None, ...]], axis=1)
            else:
                new_supports = batch_corrcoef(input.squeeze())[:, None, ...]
        return new_supports

    def tcn(self, i, residual):
        # gated TCN
        _filter = self.filter_convs[i](residual)
        # _filter = torch.tanh(_filter[..., :-self.filter_convs[i].padding[1]])
        _filter = torch.tanh(_filter)
        gate = self.gate_convs[i](residual)
        # gate = torch.sigmoid(gate[..., :-self.gate_convs[i].padding[1]])
        gate = torch.sigmoid(gate)

        # if i % self.block_layers == self.block_layers-1:
        #     return F.max_pool2d(_filter * gate, (1,2))

        return _filter * gate

    def gcn(self, i, x, supports, viz=False):
        # # gcn part, also handles diffpool for each block
        # x1 = self.gconv[i](x, supports)
        # x1 = self.bn[i](x1) # bdnt

        # 1 gcn every block_layers
        if i % self.block_layers == self.block_layers-1:
            x1 = self.gconv[i // self.block_layers](x, supports)
            x1 = self.bn[i // self.block_layers](x1) # bdnt
        else:
            return x

        if self.pool and i % self.block_layers == self.block_layers-1:
            S = self.gconv_pool[i // self.block_layers](x, supports)
            S = self.pool_bn[i // self.block_layers](S)
            if self.noT:
                S = self.pool_conv[i // self.block_layers](
                    S.transpose(1,3)).squeeze().transpose(1,2)
            self.S.append(S)
            return self.in_cluster_smoothing(x1, S, viz, supports)
        else:
            return x1

    def forward(self, input, supports=None, aptinit=None, viz=False):
        x = self.start_conv(input)
        skip = 0
        self.S = []

        # calculate current adaptive (latent) adj matrix
        if self.gcn_bool:
            if self.addaptadj or self.use_corrcoef:
                supports = self.get_adp(input, supports, viz)
        self.supports = supports

        for i in range(self.total_layers):       
            residual = x
           
            # tcn
            x = self.tcn(i, x)
            # t_rep = x

            # gcn
            if self.gcn_bool:
                x = self.gcn(i, x, supports, viz)
            
            if self.parametrized_residual:
                residual = self.residual_convs[i](residual)
            else:
                residual = F.max_pool2d(residual, (1, 2))
            x = x + residual #+t_rep

            # parametrized skip connection
            s = self.skip_convs[i](x)
            try:
                if s.size(-1)*2 == skip.size(-1):
                    skip = F.max_pool2d(skip, (1, 2))
                else:
                    skip = skip[..., -s.size(-1):]
            except:
                skip = 0
            skip = s + skip
        x = F.relu(skip)

        x = self.cl_node1(x.squeeze())
        x = self.cl_node2(F.relu(x)).transpose(1,2)
        x = F.relu(self.cl1(x))
        x = self.cl2(x)
        return x.squeeze()


class mls_coarsened_classifier(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports_len=0,
                 batch_size=32, gcn_bool=True, addaptadj=True, in_dim=2,
                 seq_len=12, residual_channels=32,
                 dilation_channels=32, skip_channels=256, end_channels=512,
                 kernel_size=3, total_layers=8, block_layers=2,
                 hidden_dim=64, noT=False, pool=True,
                 use_corrcoef=False, out_class=6, parametrized_residual=True):

        super().__init__()
        self.dropout = dropout
        self.total_layers = total_layers
        self.block_layers = block_layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.device = device
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.noT = noT
        self.pool = pool
        self.use_corrcoef = use_corrcoef
        self.out_class = out_class
        self.parametrized_residual = parametrized_residual

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        if self.parametrized_residual:
            self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        if pool:
            self.pool_bn = nn.ModuleList()
            self.gconv_pool = nn.ModuleList()
            if noT:
                self.pool_conv = nn.ModuleList()
                self.pool_bn_t = nn.ModuleList()
        
        if self.gcn_bool and self.addaptadj:
            self.nodevec = nn.Parameter(torch.randn(seq_len, 5).to(self.device), 
                                    requires_grad=True).to(self.device)

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))

        receptive_field = 1
        multi_factor = kernel_size #2

        num_layers = math.ceil(math.log(seq_len, 2))
        ## hard coded cluster number:
        _clusters = {3:[80, 30, 1],
                     4:[60, 1],
                     5:[100, 50, 25, 10, 1],
                     6:[80, 30, 1],
                     7:[80, 30, 1],
                     8:[100, 50, 20, 1]}
        clusters = _clusters[num_layers]

        # maker sure number of clusters is >= number of blocks
        if pool:
            assert len(clusters) >= total_layers // block_layers, (
                                        'need more cluster number')

            
        for i in range(total_layers):
            b = i // block_layers
            # tcn
            self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                               out_channels=dilation_channels,
                                               kernel_size=(1, kernel_size),
                                               stride=(1, kernel_size)))

            self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                             out_channels=dilation_channels,
                                             kernel_size=(1, kernel_size), 
                                             stride=(1, kernel_size)))                

            if self.parametrized_residual:
                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(
                    in_channels=dilation_channels,
                    out_channels=residual_channels,
                    kernel_size=(1, kernel_size), stride=(1, kernel_size)))
            else:
                assert dilation_channels == residual_channels, (
                    'dilation_channels and residual_channels not the same, ' +
                    'need an extra linear layer.')

            # 1x1 convolution for skip connection
            self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                             out_channels=skip_channels,
                                             kernel_size=(1, 1)))

            if self.gcn_bool and i % block_layers == block_layers-1:
                self.gconv.append(gcn2(dilation_channels, residual_channels,
                                       dropout, support_len=supports_len))
                self.bn.append(nn.BatchNorm2d(residual_channels, eps=EPS))

                if pool and i % block_layers == block_layers-1:
                    self.gconv_pool.append(gcn2(dilation_channels, clusters[b],
                                            dropout, support_len=supports_len))
                    self.pool_bn.append(nn.BatchNorm2d(clusters[b], eps=EPS))
                    if noT:
                        self.pool_conv.append(nn.Conv2d(t_list[b], 1, 1,
                                                        bias=False))
                    self.in_cluster_smoothing = diff_pool(noT)
                    self.in_cluster_smoothing_skip = diff_pool(noT)

        self.cl1 = nn.Linear(skip_channels, skip_channels//4)
        self.cl2 = nn.Linear(skip_channels//4, out_class)

    def get_adp(self, input, supports=None, viz=False):
        if self.addaptadj:
            # generate adaptive adjacency matrix based on current inputs
            nodevec = torch.einsum('ncl,lv->ncv', (input[:, 0, ...],
                                    self.nodevec))
            adp = F.softmax(F.relu(torch.matmul(nodevec,
                        nodevec.transpose(1, 2))), dim=2)
            if viz: 
                plt.imshow(adp[0].detach().cpu().numpy(), cmap='Blues')
                plt.show()
                # plot learned theta
                plt.imshow(self.nodevec.detach().cpu().numpy())
                plt.show()
                ipdb.set_trace()
                # adp.sum(1)
                # _, idx = torch.sort(adp.sum(1)) 
                # top10 = idx.cpu().numpy()[:,::-1][:,:10]

            if supports is not None:
                new_supports = torch.cat([supports, adp[:, None, ...]], axis=1)
            else:
                new_supports = adp[:, None, ...]
            if self.use_corrcoef:
                new_supports = torch.cat([new_supports, batch_corrcoef(
                               input.squeeze())[:, None, ...]], axis=1)
        else:
            if supports is not None:
                new_supports = torch.cat([supports, batch_corrcoef(
                           input.squeeze())[:, None, ...]], axis=1)
            else:
                new_supports = batch_corrcoef(input.squeeze())[:, None, ...]
        return new_supports        

    def tcn(self, i, residual):
        # gated TCN
        _filter = self.filter_convs[i](residual)
        _filter = torch.tanh(_filter)
        gate = self.gate_convs[i](residual)
        gate = torch.sigmoid(gate)

        return _filter * gate

    def gcn(self, i, x, supports, viz=False):
        # 1 gcn every block_layers
        if i % self.block_layers == self.block_layers-1:
            x1 = self.gconv[i // self.block_layers](x, supports)
            x1 = self.bn[i // self.block_layers](x1)
        else:
            return x, supports

        if self.pool and i % self.block_layers == self.block_layers - 1:
            S = self.gconv_pool[i // self.block_layers](x, supports)
            S = self.pool_bn[i // self.block_layers](S)
            if self.noT:
                S = self.pool_conv[i // self.block_layers](
                    S.transpose(1,3)).squeeze().transpose(1,2)
            self.S.append(S)
            return self.in_cluster_smoothing(x1, S, viz, supports)
        else:
            return x1, supports

    def forward(self, input, supports=None, aptinit=None, viz=False):
        x = self.start_conv(input)
        skip = 0
        self.S = []

        # calculate current adaptive (latent) adj matrix
        if self.gcn_bool:
            if self.addaptadj or self.use_corrcoef:
                supports = self.get_adp(input, supports, viz)
        self.supports = supports

        for i in range(self.total_layers):  
            residual = x
            # tcn
            x = self.tcn(i, x)
            # t_rep = x
            # gcn
            if self.gcn_bool:
                x, _supports = self.gcn(i, x, supports, viz)
            
            if self.parametrized_residual:
                residual = self.residual_convs[i](residual)
                if i % self.block_layers == self.block_layers - 1:
                    residual, _ = self.in_cluster_smoothing(residual,
                                self.S[-1], False, supports)
            else:
                residual = F.max_pool2d(residual, (1, 2))
            x = x + residual #+t_rep

            # parametrized skip connection
            s = self.skip_convs[i](x)
            try:
                if s.size(-1)*2 == skip.size(-1):
                    skip = F.max_pool2d(skip, (1, 2))
                else:
                    skip = skip[..., -s.size(-1):]
                if i % self.block_layers == self.block_layers - 1:
                    skip, _ = self.in_cluster_smoothing_skip(skip, self.S[-1],
                                                        False, supports)
                    supports = _supports
            except:
                skip = 0
                if self.block_layers == 1:
                    supports = _supports
            skip = s + skip
        x = F.relu(skip)

        x = F.relu(self.cl1(x.squeeze()))
        x = self.cl2(x)
        return x.squeeze()

import torch
import torch.nn as nn

from dgl.nn.pytorch import GraphConv
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from layers import GraphAttentionLayer
import numpy as np
import scipy.sparse as sp
from utils import normalize_adj
class MatGRUCell(torch.nn.Module):
    """
    GRU cell for matrix, similar to the official code.
    Please refer to section 3.4 of the paper for the formula.
    """

    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.update = MatGRUGate(in_feats, out_feats, torch.nn.Sigmoid())

        self.reset = MatGRUGate(in_feats, out_feats, torch.nn.Sigmoid())

        self.htilda = MatGRUGate(in_feats, out_feats, torch.nn.Tanh())

    def forward(self, prev_Q, z_topk=None):
        if z_topk is None:
            z_topk = prev_Q

        update = self.update(z_topk, prev_Q)
        reset = self.reset(z_topk, prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q


class MatGRUGate(torch.nn.Module):
    """
    GRU gate for matrix, similar to the official code.
    Please refer to section 3.4 of the paper for the formula.
    """

    def __init__(self, rows, cols, activation):
        super().__init__()
        self.activation = activation
        self.W = Parameter(torch.Tensor(rows, rows))
        self.U = Parameter(torch.Tensor(rows, rows))
        self.bias = Parameter(torch.Tensor(rows, cols))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.W)
        init.xavier_uniform_(self.U)
        init.zeros_(self.bias)

    def forward(self, x, hidden):
        out = self.activation(
            self.W.matmul(x) + self.U.matmul(hidden) + self.bias
        )

        return out


class TopK(torch.nn.Module):
    """
    Similar to the official `egcn_h.py`. We only consider the node in a timestamp based subgraph,
    so we need to pay attention to `K` should be less than the min node numbers in all subgraph.
    Please refer to section 3.4 of the paper for the formula.
    """

    def __init__(self, feats, k):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats, 1))
        self.reset_parameters()

        self.k = k

    def reset_parameters(self):
        init.xavier_uniform_(self.scorer)

    def forward(self, node_embs):
        scores = node_embs.matmul(self.scorer) / self.scorer.norm().clamp(
            min=1e-6
        )
        vals, topk_indices = scores.view(-1).topk(self.k)
        out = node_embs[topk_indices] * torch.tanh(
            scores[topk_indices].view(-1, 1)
        )
        # we need to transpose the output
        return out.t()


class EvolveGCNH(nn.Module):
    def __init__(
        self,
        in_feats=166,
        n_hidden=76,
        num_layers=2,
        n_classes=2,
        classifier_hidden=510,
    ):
        # default parameters follow the official config
        super(EvolveGCNH, self).__init__()
        self.num_layers = num_layers
        self.pooling_layers = nn.ModuleList()
        self.recurrent_layers = nn.ModuleList()
        self.gnn_convs = nn.ModuleList()
        self.gcn_weights_list = nn.ParameterList()

        self.pooling_layers.append(TopK(in_feats, n_hidden))
        # similar to EvolveGCNO
        self.recurrent_layers.append(
            MatGRUCell(in_feats=in_feats, out_feats=n_hidden)
        )
        self.gcn_weights_list.append(
            Parameter(torch.Tensor(in_feats, n_hidden))
        )
        self.gnn_convs.append(
            GraphConv(
                in_feats=in_feats,
                out_feats=n_hidden,
                bias=False,
                activation=nn.RReLU(),
                weight=False,
            )
        )
        for _ in range(num_layers - 1):
            self.pooling_layers.append(TopK(n_hidden, n_hidden))
            self.recurrent_layers.append(
                MatGRUCell(in_feats=n_hidden, out_feats=n_hidden)
            )
            self.gcn_weights_list.append(
                Parameter(torch.Tensor(n_hidden, n_hidden))
            )
            self.gnn_convs.append(
                GraphConv(
                    in_feats=n_hidden,
                    out_feats=n_hidden,
                    bias=False,
                    activation=nn.RReLU(),
                    weight=False,
                )
            )

        self.mlp = nn.Sequential(
            nn.Linear(n_hidden, classifier_hidden),
            nn.ReLU(),
            nn.Linear(classifier_hidden, n_classes),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for gcn_weight in self.gcn_weights_list:
            init.xavier_uniform_(gcn_weight)

    def forward(self, g_list):
        feature_list = []
        for g in g_list:
            feature_list.append(g.ndata["feat"])
        for i in range(self.num_layers):
            W = self.gcn_weights_list[i]
            for j, g in enumerate(g_list):
                X_tilde = self.pooling_layers[i](feature_list[j])
                W = self.recurrent_layers[i](W, X_tilde)
                feature_list[j] = self.gnn_convs[i](
                    g, feature_list[j], weight=W
                )
        return self.mlp(feature_list[-1])


class EvolveGATO(nn.Module):
    def __init__(
        self,
        in_feats=166,
        n_hidden=256,
        num_layers=2,
        n_classes=2,
        classifier_hidden=307,
        dropout = 0.6,
        alpha =0.2,
        nheads = 8
    ):
        # default parameters follow the official config
        super(EvolveGATO, self).__init__()
        self.num_layers = num_layers
        # 跟新W的rnn
        self.recurrent_layers = nn.ModuleList()
        # 更新a的rnn
        self.recurrent_a_layers = nn.ModuleList()
        # self.gnn_convs = nn.ModuleList()
        self.gnn_gats = nn.ModuleList()
        # self.gcn_weights_list = nn.ParameterList()
        self.gat_weights_list = nn.ParameterList()
        self.gat_weights_a_list = nn.ParameterList()
        self.dropout=dropout

        # In the paper, EvolveGCN-O use LSTM as RNN layer. According to the official code,
        # EvolveGCN-O use GRU as RNN layer. Here we follow the official code.
        # See: https://github.com/IBM/EvolveGCN/blob/90869062bbc98d56935e3d92e1d9b1b4c25be593/egcn_o.py#L53
        # PS: I try to use torch.nn.LSTM directly,
        #     like [pyg_temporal](github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/torch_geometric_temporal/nn/recurrent/evolvegcno.py)
        #     but the performance is worse than use torch.nn.GRU.
        # PPS: I think torch.nn.GRU can't match the manually implemented GRU cell in the official code,
        #      we follow the official code here.
        self.recurrent_layers.append(
            MatGRUCell(in_feats=in_feats, out_feats=n_hidden)
        )
        self.recurrent_a_layers.append(
            nn.GRUCell(input_size=2*n_hidden,hidden_size=2*n_hidden)
        )
        self.gat_weights_list.append(
            Parameter(torch.Tensor(in_feats, n_hidden))
        )
        #GAT的a参数
        self.gat_weights_a_list.append(
            Parameter(torch.Tensor(2 * n_hidden, 1))
        )
        self.gnn_gats.append(
            GraphAttentionLayer(
                in_features=in_feats,
                out_features=n_hidden,
                dropout=dropout,
                alpha=alpha,
                concat=False

            )

        )
        for _ in range(num_layers - 1):
            self.recurrent_layers.append(
                MatGRUCell(in_feats=n_hidden, out_feats=n_hidden)
            )
            self.recurrent_a_layers.append(
                nn.GRUCell(input_size=2 * n_hidden, hidden_size=2 * n_hidden)
            )
            self.gat_weights_list.append(
                Parameter(torch.Tensor(n_hidden, n_hidden))
            )
            self.gat_weights_a_list.append(
                Parameter(torch.Tensor(2 * n_hidden, 1))
            )
            self.gnn_gats.append(
                GraphAttentionLayer(
                    in_features=n_hidden,
                    out_features= n_hidden,
                    dropout=dropout,
                    alpha=alpha,
                    concat=False)
            )

        self.mlp = nn.Sequential(
            nn.Linear(n_hidden, classifier_hidden),
            nn.ReLU(),
            nn.Linear(classifier_hidden, n_classes),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for gat_weight in self.gat_weights_list:
            init.xavier_uniform_(gat_weight)

        for gat_a_weight in self.gat_weights_a_list:
            init.xavier_uniform_(gat_a_weight)


    def forward(self, g_list):
        feature_list = []
        for g in g_list:
            feature_list.append(g.ndata["feat"])

        W = self.gat_weights_list[0]
        a = self.gat_weights_a_list[0]
        a=a.view(1, -1)
        for j, g in enumerate(g_list):
            g = g.adjacency_matrix()
            g = normalize_adj(g)
            g = g.cuda()

            a=self.recurrent_a_layers[0](a)
            a=a.view(-1,1)
            W = self.recurrent_layers[0](W)

            x=feature_list[j]
            x = F.dropout(x, self.dropout, training=self.training)
            feature_list[j] = self.gnn_gats[0](
                g, x, W,a
            )
            a = a.view(1,-1)
        W = self.gat_weights_list[1]
        a = self.gat_weights_a_list[1]
        a = a.view(1,-1)
        for j, g in enumerate(g_list):
            g = g.adjacency_matrix()
            g = normalize_adj(g)
            g = g.cuda()

            a = self.recurrent_a_layers[1](a)
            a=a.view(-1,1)
            W = self.recurrent_layers[1](W)
            x=feature_list[j]
            x = F.dropout(x, self.dropout, training=self.training)
            feature_list[j] = F.elu(self.gnn_gats[1](
                g, x, W,a
            ))
            a = a.view(1,-1)
        return self.mlp(feature_list[-1])
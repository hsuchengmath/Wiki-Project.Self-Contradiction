
import torch
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing
import torch.nn.functional as F
from inits import uniform
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import torch.nn as nn

class RAGCNConv(MessagePassing):

    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True, **kwargs):
        super(RAGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.basis = Param(torch.Tensor(num_bases, in_channels, out_channels))
        self.att_r = Param(torch.Tensor(num_relations, num_bases))
        self.heads = 1
        self.att = Param(torch.Tensor(1, self.heads, 2 * out_channels))
        self.gate_layer = nn.Linear(4*out_channels, 1)
        self.relu = nn.ReLU()
        self.negative_slope = 0.2

        if root_weight:
            self.root = Param(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.dropout = 0
 
    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att_r)
        uniform(size, self.root)
        uniform(size, self.bias)
        uniform(size, self.att)


    def forward(self, x, edge_index, edge_type, edge_norm=None, size=None):
        """"""
        # if size is None and torch.is_tensor(x):
        #     edge_index, _ = remove_self_loops(edge_index)
        #     edge_index, _ = add_self_loops(edge_index,
        #                                    num_nodes=x.size(self.node_dim))

        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_norm=edge_norm)

    def message(self, x_i, edge_index_i,edge_index_j, x_j,edge_type,size_i, edge_norm,return_attention_weights=True):

        '''
        For exmaple:
        edge_index = [[0,0,0,1,2,3],
                      [1,2,3,0,0,0]]
        x_i = [v0,v0,v0,v1,v2,v3]
        x_j = [v1,v2,v3,v0,v0,v0]
        '''
        ## thelta_r * x_j
        w = torch.matmul(self.att_r, self.basis.view(self.num_bases, -1))

        if x_j is None:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, index)
        else:
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            w = torch.index_select(w, 0, edge_type)
            x_j = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        ## thelta_root * x_i
        if self.root is not None:
            if x_i is None:
                x_i = self.root
            else:
                x_i = torch.matmul(x_i, self.root)

        ## attention_ij
        x_j = x_j.view(-1, self.heads, self.out_channels)
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            
        alpha = softmax(alpha, edge_index_i,size_i)

        if return_attention_weights:
            self.alpha = alpha

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = x_j * alpha.view(-1, self.heads, 1)
        out = out.view(-1,self.out_channels)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        '''
        x shape:(4,16)
        aggr_out shape:(4,32)
        '''
        if self.root is not None:
            if x is None:
                aggr_out = aggr_out + self.root
            else:
                aggr_out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        return aggr_out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(self.__class__.__name__,
                                                     self.in_channels,
                                                     self.out_channels,
                                                     self.num_relations)



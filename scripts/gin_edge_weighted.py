from typing import Callable, Union

import numpy as np
import torch
from torch import nn 
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from dgl.nn.pytorch.glob import AvgPooling, MaxPooling, SumPooling
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp


class SELayer(nn.Module):
    """Squeeze-and-excitation networks"""

    def __init__(self, in_channels, se_channels):
        super(SELayer, self).__init__()

        self.in_channels = in_channels
        self.se_channels = se_channels

        self.encoder_decoder = nn.Sequential(
            nn.Linear(in_channels, se_channels),
            nn.ELU(),
            nn.Linear(se_channels, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """"""
        # Aggregate input representation
        x_global = torch.mean(x, dim=0)
        # Compute reweighting vector s
        s = self.encoder_decoder(x_global)

        return x * s

class MLP(nn.Module):
    """MLP with linear output"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, use_selayer):
        """MLP layers construction

        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction

        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(
                    SELayer(hidden_dim, int(np.sqrt(hidden_dim)))
                    if use_selayer
                    else nn.BatchNorm1d(hidden_dim)
                )

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)

class GINEConv(MessagePassing):
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GINEConv, self).__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        # self.reset_parameters()

    def forward(self, x, edge_index,
                edge_weight, size= None):

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)

        out += (1 + self.eps) * x

        return self.nn(out)

    def message(self, x_j,  edge_weight):
        return F.relu(x_j) if edge_weight is None else F.relu(x_j) * edge_weight.view(-1, 1)



    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

class GIN_Edge_Weighted(nn.Module):
    def __init__(
        self,
        num_layers,
        num_mlp_layers,
        input_dim,
        hidden_dim,
        output_dim,
        final_dropout,
        learn_eps,
        graph_pooling_type,
        use_selayer,
    ):
        super(GIN_Edge_Weighted, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps

        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(
                    num_mlp_layers, input_dim, hidden_dim, hidden_dim, use_selayer
                )
            else:
                mlp = MLP(
                    num_mlp_layers, hidden_dim, hidden_dim, hidden_dim, use_selayer
                )

            self.ginlayers.append(
                GINEConv(
                    mlp,
                    train_eps=self.learn_eps
                )
            )
            self.batch_norms.append(
                SELayer(hidden_dim, int(np.sqrt(hidden_dim)))
                if use_selayer
                else nn.BatchNorm1d(hidden_dim)
            )

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == "sum":
            self.pool = SumPooling()
        elif graph_pooling_type == "mean":
            self.pool = AvgPooling()
        elif graph_pooling_type == "max":
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, g, h, efeat, edge_weight=None):
        nodes_representation = []
        edge_index = torch.stack(g.edges()).to(h.device)

        node_connectivity_mask = self._get_node_connectivity_mask(g, g.number_of_nodes(), edge_index, edge_weight, h.device)
        hidden_rep = [h * node_connectivity_mask]

        for i in range(self.num_layers - 1):
            h = self.ginlayers[i](h, edge_index, edge_weight=edge_weight)
            nodes_representation.append(h)
            h = self.batch_norms[i](h)
            hidden_rep.append(h * node_connectivity_mask)
            h = F.relu(h)
        
        score_over_layer = 0
        
        for i, h in list(enumerate(hidden_rep)):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))

        return score_over_layer, nodes_representation

    def _get_node_connectivity_mask(self, g, node_num, edge_index, edge_weight, device):
        """
        get the node mask such that the node disconnceted from the "center" node is masked.
        """
        if edge_weight is None:
            return torch.ones((node_num, 1), dtype=torch.float32, device=device)
        remained_edge_ind = (edge_weight >=  0.9).nonzero(as_tuple=False).squeeze()
        edge_index_remained = edge_index.index_select(dim=-1, index=remained_edge_ind)
        adj_remained = to_scipy_sparse_matrix(edge_index_remained, num_nodes=node_num)
        _, components = sp.csgraph.connected_components(adj_remained) 
        center_node_ind = (g.ndata['seed'] == 1).nonzero().squeeze().cpu().numpy()
        components_remained = components[center_node_ind]
        node_mask = torch.zeros(node_num, dtype=torch.float32, device=device)
        components = torch.tensor(components, device=device)
        for comp_ind in components_remained:
            node_mask = node_mask + torch.where(components==comp_ind, 1.0, 0.0)
        return node_mask[:, np.newaxis]
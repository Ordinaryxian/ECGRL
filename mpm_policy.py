import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import LayerNorm
from torch_geometric.nn import global_add_pool

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Gate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.v = nn.Parameter(torch.zeros(dim))
    def forward(self):
        g = F.softplus(self.v)
        g = g / (g.sum() + 1e-8)
        return g

class graph_nn(nn.Module):
    def __init__(self, action_space=None, input_dim=2, hidden_dim1=128, hidden_dim2=64):
        super(graph_nn, self).__init__()
        self.gate = Gate(input_dim)
        self.conv1 = GCNConv(input_dim, hidden_dim1)
        self.linear1 = nn.Linear(hidden_dim1, hidden_dim2)
        self.linear2 = nn.Linear(hidden_dim2, action_space)
        self.layer_norm = LayerNorm(hidden_dim1)
    def forward(self, x, edge_index, edge_weight):
        mask = self.gate()
        x = x * mask
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = global_add_pool(self.layer_norm(x), torch.LongTensor(torch.zeros(len(edge_index[0])+1).tolist()).to(device))
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        output = x
        return output, mask
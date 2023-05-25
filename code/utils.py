import torch
from torch_scatter import scatter_add
from torch_geometric.utils import get_laplacian, add_self_loops


def normalize_adj_tensor(adj):
    """Symmetrically normalize adjacency tensor."""
    rowsum = torch.sum(adj,1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[d_inv_sqrt == float("Inf")] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(adj,d_mat_inv_sqrt).transpose(0,1),d_mat_inv_sqrt)

def normalize_adj_tensor_sp(adj):
    """Symmetrically normalize sparse adjacency tensor."""
    device = adj.device
    adj = adj.to("cpu")
    rowsum = torch.spmm(adj, torch.ones((adj.size(0),1))).reshape(-1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[d_inv_sqrt == float("Inf")] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    adj = torch.mm(torch.smm(adj.transpose(0,1),d_mat_inv_sqrt.transpose(0,1)),d_mat_inv_sqrt)
    return adj.to(device)

def edge2adj(x, edge_index):
    """Convert edge index to adjacency matrix"""
    num_nodes = x.shape[0]
    tmp, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    edge_weight = torch.ones(tmp.size(1), dtype=None,
                                     device=edge_index.device)

    row, col = tmp[0], tmp[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return torch.sparse.FloatTensor(tmp, edge_weight,torch.Size((num_nodes, num_nodes)))
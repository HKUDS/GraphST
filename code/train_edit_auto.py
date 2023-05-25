import argparse
import os.path as osp
import random
from time import perf_counter as t
import yaml
import numpy as np
from yaml import SafeLoader
from scipy.linalg import fractional_matrix_power, inv
from torch.utils.data import random_split
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from layers import GCNConv
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid, CitationFull, Amazon, Coauthor, GitHub, FacebookPagePage, LastFMAsia, DeezerEurope
from torch_geometric.utils import dropout_adj
from model import Encoder, Model, drop_feature
from utils import normalize_adj_tensor, normalize_adj_tensor_sp, edge2adj
from attack import PGD_attack_graph
from eval import label_classification
import pickle
from torch_geometric.nn import global_mean_pool, global_add_pool
from model_gcn import GNN
import torch.optim as optim
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # for remindering of the minmatch of shape
import warnings
warnings.filterwarnings('ignore')
def load_data(file):
        data_load_file = []
        file_1 = open(file, "rb")
        data_load_file = pickle.load(file_1)
        return data_load_file
hy = load_data("./data/hy_new_aaai_2.pickle")
# hy = load_data("./data/hy_aaai_chi_1.pickle")
# print(len(list(hy.nodes())))
# println

class vgae(nn.Module):
    def __init__(self, gnn, emb_dim):
        super(vgae, self).__init__()
        self.encoder = gnn
        self.encoder_mean = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(inplace=True), nn.Linear(emb_dim, emb_dim))
        # make sure std is positive
        self.encoder_std = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(inplace=True), nn.Linear(emb_dim, emb_dim), nn.Softplus())
        # only reconstruct first 7-dim, please refer to https://github.com/snap-stanford/pretrain-gnns/issues/30
        self.decoder = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(emb_dim, emb_dim), nn.ReLU(inplace=True), nn.Linear(emb_dim, 4), nn.Sigmoid())
        self.decoder_edge = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, 1))

        self.bceloss = nn.BCELoss(reduction='none')
        self.pool = global_mean_pool
        self.add_pool = global_add_pool
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)

        # reconstruct 4-class & 3-class edge_attr for 1st & 2nd dimension
        self.decoder_1 = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, 4))
        self.decoder_2 = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, 4))
        self.crossentropyloss = nn.CrossEntropyLoss(reduction='none')

    def forward_encoder(self, x, edge_index):
        x = self.encoder(x, edge_index)
        x_mean = self.encoder_mean(x)
        x_std = self.encoder_std(x)
        gaussian_noise = torch.randn(x_mean.shape).to(x.device)
        x = gaussian_noise * x_std + x_mean
        return x.detach(), x_mean, x_std

    # def forward_decoder(self, x, edge_index, edge_index_neg):
    def forward_decoder(self, x, edge_index):
        eleWise_mul = x[edge_index[0]] * x[edge_index[1]]
        edge_attr_pred = self.decoder(eleWise_mul)
        edge_pos = self.sigmoid( self.decoder_edge(eleWise_mul) ).squeeze()
        # edge_neg = self.sigmoid( self.decoder_edge(x[edge_index_neg[0]] * x[edge_index_neg[1]]) ).squeeze()
        # return edge_attr_pred, edge_pos, edge_neg
        return edge_pos

    def loss_vgae(self, edge_pos_pred, edge_index_batch, x_mean, x_std, reward=None):
        # evaluate p(A|Z)
        # num_edge, _ = edge_attr_pred.shape
        # loss_rec = self.bceloss(edge_attr_pred.reshape(-1), edge_attr[:, :4].reshape(-1))
        # loss_rec = loss_rec.reshape((num_edge, -1)).sum(dim=1)

        loss_edge_pos = self.bceloss(edge_pos_pred, torch.ones(edge_pos_pred.shape).to(edge_pos_pred.device))
        # loss_edge_neg = self.bceloss(edge_neg_pred, torch.zeros(edge_neg_pred.shape).to(edge_neg_pred.device))
        # loss_pos = loss_rec + loss_edge_pos
        loss_pos = loss_edge_pos
        loss_pos_cat = torch.cat((loss_pos, loss_pos), 0).view(2, -1)
        # print("loss_pos:", loss_pos_cat.size())
        # print("edge_index_batch:", edge_index_batch.size())
        # println()
        loss_pos = self.pool(loss_pos_cat, edge_index_batch)
        # loss_neg = self.pool(loss_edge_neg, edge_index_neg_batch)
        # loss_rec = loss_pos + loss_neg
        loss_rec = loss_pos
        #print('loss_pos + loss_neg', loss_pos, loss_neg)
        if not reward is None:
            loss_rec = loss_rec * reward
            #print("reward:", reward)
            #print("loss_rec:", loss_rec)

        # evaluate p(Z|X,A)
        kl_divergence = - 0.5 * (1 + 2 * torch.log(x_std+ 1e-6) - x_mean**2 - x_std**2).sum(dim=1)
        kl_ones = torch.ones(kl_divergence.shape).to(kl_divergence.device)
        # kl_divergence = self.pool(kl_divergence, batch)
        # kl_double_norm = 1 / self.add_pool(kl_ones, batch)
        # kl_divergence = kl_divergence * kl_double_norm
        # print("loss_rec:",loss_rec.mean())
        # print("kl_divergence:",kl_divergence.size())
        # println()
        loss = (loss_rec.mean(axis=1) + kl_divergence).mean()
        '''
        # link prediction for sanity check
        from sklearn.metrics import roc_auc_score
        from sklearn.metrics import average_precision_score
        print(roc_auc_score(edge_attr.cpu().numpy(), edge_attr_pred.detach().cpu().numpy()), average_precision_score(edge_attr.cpu().numpy(), edge_attr_pred.detach().cpu().numpy()))
        '''
        return loss, loss_edge_pos.mean().item()
        # return loss, (loss_edge_pos.mean()+loss_edge_neg.mean()).item()/2

    def generate(self, data):
        x, _, _ = self.forward_encoder(data.x, data.edge_index)
        eleWise_mul = torch.einsum('nd,md->nmd', x, x)
        # calculate softmax probability
        prob = self.decoder_edge(eleWise_mul).squeeze()
        # print("prob:", prob.size())
        # pritnl()
        prob = torch.exp(prob)
        prob[torch.isinf(prob)] = 1e10
        prob[list(range(x.shape[0])), list(range(x.shape[0]))] = 0
        prob = torch.einsum('nm,n->nm', prob, 1 / prob.sum(dim=1))

        # sparsify
        
        prob[prob < 1e-1] = 0
        prob[prob.sum(dim=1) == 0] = 1
        prob[list(range(x.shape[0])), list(range(x.shape[0]))] = 0
        prob = torch.einsum('nm,n->nm', prob, 1 / prob.sum(dim=1))

        # predict 4-class & 3-class edge_attr for 1st & 2nd dimension
        edge_attr_prob_1 = self.softmax(self.decoder_1(eleWise_mul))
        edge_attr_rand_1 = torch.rand((edge_attr_prob_1.shape[0], edge_attr_prob_1.shape[1]))
        edge_attr_pred_1 = torch.zeros((edge_attr_prob_1.shape[0], edge_attr_prob_1.shape[1]), dtype=torch.int64)
        for n in range(3):
            edge_attr_pred_1[edge_attr_rand_1 >= edge_attr_prob_1[:, :, n]] = n + 1
            edge_attr_rand_1 -= edge_attr_prob_1[:, :, n]

        edge_attr_prob_2 = self.softmax(self.decoder_2(eleWise_mul))
        edge_attr_rand_2 = torch.rand((edge_attr_prob_2.shape[0], edge_attr_prob_2.shape[1]))
        edge_attr_pred_2 = torch.zeros((edge_attr_prob_2.shape[0], edge_attr_prob_2.shape[1]), dtype=torch.int64)
        for n in range(2):
            edge_attr_pred_2[edge_attr_rand_2 >= edge_attr_prob_2[:, :, n]] = n + 1
            edge_attr_rand_2 -= edge_attr_prob_2[:, :, n]

        edge_attr_pred = torch.cat((edge_attr_pred_1.reshape((edge_attr_prob_1.shape[0], edge_attr_prob_1.shape[1], 1)),
                                    edge_attr_pred_2.reshape(
                                        (edge_attr_prob_2.shape[0], edge_attr_prob_2.shape[1], 1)),edge_attr_pred_2.reshape(
                                        (edge_attr_prob_2.shape[0], edge_attr_prob_2.shape[1], 1)),edge_attr_pred_2.reshape(
                                        (edge_attr_prob_2.shape[0], edge_attr_prob_2.shape[1], 1))), dim=2)
        

        return prob, edge_attr_pred

def train(model: Model, x, edge_index, eps, model_1, optimizer_1,model_2, optimizer_2,lamb, alpha, beta, steps, node_ratio):
    optimizer.zero_grad()
    adj = edge2adj(x, edge_index)
    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
    # print("***:", x.size())
    # println()
    x_1 = drop_feature(x, drop_feature_rate_1)
    # print("x_1", x_1)
    x_2 = drop_feature(x, drop_feature_rate_2)  
    
    # adj_1 = edge2adj(x_1, edge_index_1)
    adj_2 = edge2adj(x_2, edge_index_2)
    # print("adj_1:", adj_1)
    # print("adj_1_shape:", adj_1.size())
    'learning to sample'
    x_1, x_mean, x_std = model_1.forward_encoder(x, edge_index)
    # print("x_1", x_1)
    # println()
    edge_pos_pred = model_1.forward_decoder(x_1,edge_index)
    # print("edge_index:", edge_index)
    # print("edge_pos_pred:", edge_pos_pred.size())
    s = torch.sparse_coo_tensor(edge_index,edge_pos_pred, (adj.size()[0],adj.size()[1]))
    adj_1 = s.to_dense()
    # print("adj_vgae:", adj_vgae)
    # print("x_1:",x_1.size())
    # print(edge_pos_pred)
    # print("x_3:", x_3.size())
    # print(edge_pos_pred.size())
    # println()
    
    x_2, x_mean, x_std = model_2.forward_encoder(x, edge_index)
    # print("x_1", x_1)
    # println()
    edge_pos_pred = model_2.forward_decoder(x_2,edge_index)
    # print("edge_index:", edge_index)
    # print("edge_pos_pred:", edge_pos_pred.size())
 
    s = torch.sparse_coo_tensor(edge_index,edge_pos_pred, (adj.size()[0],adj.size()[1]))
    adj_2 = s.to_dense()
        
    
    if eps > 0:
        print("x_1:", x_1.size())
        print("x:", x.size())
        file=open(r"./data/tmp_case_before.pickle","wb")
        pickle.dump(x,file) #storing_list
        file.close()
        adj_3, x_3 = PGD_attack_graph(model, edge_index_1, edge_index, x_1, x, steps, node_ratio, alpha, beta)
        print("x_3:", x_3.size())
        file=open(r"./data/tmp_case_after.pickle","wb")
        pickle.dump(x_3,file) #storing_list
        file.close()
        # println()
    z = model(x, adj)
    z_1 = model(x_1, adj_1)
    z_2 = model(x_2, adj_2)
    # print("x:", x)
    # print("edge_index:", edge_index)
    
    '''adding cross-view contrastive learning'''
    node_list = list(hy.nodes)
    # print(node_list)
    # println()
    poi_view = []
    spatial_view = []
    flow_view = []
    for item in node_list:
        if item.endswith("s"):
            spatial_view.append(node_list.index(item))
        elif item.endswith("p"):
            poi_view.append(node_list.index(item))
        else:
            flow_view.append(node_list.index(item))
 
    # '''cross-view conhtarstive learning'''
    # linear = nn.Linear(len(spatial_view), 180).to(device)
    # linear_1 = nn.Linear(len(flow_view), 180).to(device)
    # poi_view_tensor = torch.tensor(np.array([z_2[item].tolist() for item in poi_view]),requires_grad=True).to(device)
    # spatial_view_tensor = torch.tensor(np.array([z_2[item].tolist() for item in spatial_view]),requires_grad=True).to(device)
    # flow_view_tensor = torch.tensor(np.array([z_2[item].tolist() for item in flow_view]),requires_grad=True).to(device)
    # flow_out = linear_1(flow_view_tensor.view(128,len(flow_view)).float())
    # flow_trans = flow_out.view(180,128)
    # spatial_out = linear(spatial_view_tensor.view(128,len(spatial_view)).float())
    # spatial_trans = spatial_out.view(180,128).float()
    # # print(spatial_trans.size())
    # loss_v1, simi_v1 = model.loss(flow_trans.float(),spatial_trans.float(),batch_size=0)
    # loss_v2, simi_v2 = model.loss(flow_trans.float(),poi_view_tensor.float(),batch_size=0)
    # loss_v3, simi_v3 = model.loss(spatial_trans.float(),poi_view_tensor.float(),batch_size=0)
    # # print(loss_v1.mean(), loss_v2.mean(), loss_v3.mean())
    # '''adaptative weight for cross-view loss'''
    # model_fs = nn.Sequential(nn.Linear(360, 1),nn.ReLU()).to(device)
    # # mlp = nn.Linear(in_features = 360, out_features = 1).to(device)
    # flow_spatial = torch.cat((flow_trans,spatial_trans),0).to(device)
    # flow_poi = torch.cat((flow_trans,poi_view_tensor),0).to(device)
    # spatial_poi = torch.cat((spatial_trans, poi_view_tensor),0).to(device)
    # fs_w = model_fs(flow_spatial.view(128,-1).float()).mean()
    # fp_w = model_fs(flow_poi.view(128,-1).float()).mean()
    # sp_w = model_fs(spatial_poi.view(128,-1).float()).mean()
    # # print("fs_w:",fs_w.item())
    # # print("fp_w:",fp_w.item())
    # # print("sp_w:",sp_w.item())
    # loss_view = fs_w.item()*loss_v1+fp_w.item()*loss_v2+sp_w.item()*loss_v3
    '''cross-view conhtarstive learning'''
    reg_num = 180
    linear = nn.Linear(len(spatial_view), reg_num).to(device)
    linear_1 = nn.Linear(len(flow_view), reg_num).to(device)
    poi_view_tensor = torch.tensor(np.array([z_2[item].tolist() for item in poi_view]),requires_grad=True).to(device)
    spatial_view_tensor = torch.tensor(np.array([z_2[item].tolist() for item in spatial_view]),requires_grad=True).to(device)
    flow_view_tensor = torch.tensor(np.array([z_2[item].tolist() for item in flow_view]),requires_grad=True).to(device)
    flow_out = linear_1(flow_view_tensor.view(128,len(flow_view)).float())
    flow_trans = flow_out.view(reg_num,128)
    spatial_out = linear(spatial_view_tensor.view(128,len(spatial_view)).float())
    spatial_trans = spatial_out.view(reg_num,128).float()
    # print(spatial_trans.size())
    loss_v1, simi_v1 = model.loss(flow_trans.float(),spatial_trans.float(),batch_size=0)
    loss_v2, simi_v2 = model.loss(flow_trans.float(),poi_view_tensor.float(),batch_size=0)
    loss_v3, simi_v3 = model.loss(spatial_trans.float(),poi_view_tensor.float(),batch_size=0)
    # print(loss_v1.mean(), loss_v2.mean(), loss_v3.mean())
    '''adaptative weight for cross-view loss'''
    model_fs = nn.Sequential(nn.Linear(reg_num*2, 1),nn.ReLU()).to(device)
    # mlp = nn.Linear(in_features = 360, out_features = 1).to(device)
    flow_spatial = torch.cat((flow_trans,spatial_trans),0).to(device)
    flow_poi = torch.cat((flow_trans,poi_view_tensor),0).to(device)
    spatial_poi = torch.cat((spatial_trans, poi_view_tensor),0).to(device)
    fs_w = model_fs(flow_spatial.view(128,-1).float()).mean()
    fp_w = model_fs(flow_poi.view(128,-1).float()).mean()
    sp_w = model_fs(spatial_poi.view(128,-1).float()).mean()
    # print("fs_w:",fs_w.item())
    # print("fp_w:",fp_w.item())
    # print("sp_w:",sp_w.item())
    loss_view = fs_w.item()*loss_v1+fp_w.item()*loss_v2+sp_w.item()*loss_v3
    
    
    loss1, simi1 = model.loss(z_1,z_2,batch_size=0)
    loss2, simi2 = model.loss(z_1,z,batch_size=0)
    loss3, simi3 = model.loss(z_2,z,batch_size=0)
    loss3 = loss3 -loss3.mean()
    # loss3 = loss3.mean()
    # print("loss3:", loss3)
    loss3[loss3 > 0] = 1
    loss3[loss3 <= 0] = 0.01 # weaken the reward for low cl loss
    
    loss1 = loss1.mean() + lamb*torch.clamp(simi1*2 - simi2.detach()-simi3.detach(), 0).mean()
    # loss_vage = loss3
    # loss_vage = loss3*loss1
    # print("loss_vage:",loss_vage)
    # print("loss_vage:",loss_vage.size())
    # println()
    if eps > 0:  
        z_3 = model(x_3,adj_3)
        loss2, _ = model.loss(z_1,z_3)
        loss2 = loss2.mean()
        loss = (loss1 + eps*loss2+0.05*loss_view.mean())
    else: 
        loss = loss1+0.05*loss_view.mean()
        loss2 = loss1
    '''Adding loss for VGAE'''
    loss_3, link_loss_2 = model_2.loss_vgae(edge_pos_pred,edge_index, x_mean, x_std, reward=loss3.mean().item())
    # loss_2 =  loss_2
    # print("loss_vage:",loss_vage)
    # print("loss1:", loss1)
    # print("loss:", loss)
    # println()
    
    loss.backward(retain_graph=True)
    # loss_vage = (loss3*loss).mean()
    loss_vage = loss_3+link_loss_2
    # print("loss_vage:",loss_vage)
    # println()
    loss_vage.backward(retain_graph=True)
    optimizer.step()
    optimizer_1.step()
 
    return loss1.item(), loss2.item(),loss_vage.item()

def test(model: Model, x, edge_index, model_1,y, final=False, task ="node"):   
    model.eval()
    adj = edge2adj(x, edge_index)
    x = x.to(device)
    adj = adj.to(device)
    # print("adj.size():", adj.size())
    # z = model(x, adj)
    # print("test:", z.size())
    # file=open(r"./data/tmp_vector.pickle","wb")
    # pickle.dump(z,file) #storing_list
    # file.close()
    
    x_1, x_mean, x_std = model_1.forward_encoder(x, edge_index.to(device))
    x_1 = x_1.to(device)
    # print("x_1", x_1)
    # println()
    edge_pos_pred = model_1.forward_decoder(x,edge_index.to(device))
    # print("x:", x.size())
    # print("edge_index:", edge_index)
    # print("edge_pos_pred:", edge_pos_pred.size())
    s = torch.sparse_coo_tensor(edge_index.to(device),edge_pos_pred.to(device), (adj.size()[0],adj.size()[1]))
    adj_1 = s.to_dense()
    adj_1 = adj_1.to(device)
    z_1 = model(x_1, adj_1)
    print("test z_1:", z_1.size())
    
    
    file=open(r"./data/tmp_vector_chi_3.pickle","wb")
    pickle.dump(z_1,file) #storing_list
    file.close()
    
    # return label_classification(z, y, ratio=0.1),label_classification(z_1, y, ratio=0.1)
    return label_classification(z_1, y, ratio=0.1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--log', type=str, default='results/Cora/')
    parser.add_argument('--seed', type=int, default=39788)
    parser.add_argument('--eps', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--lamb', type=float, default=0.05)
    args = parser.parse_args()
 

    assert args.gpu_id in range(0, 8)

    
    config = yaml.load(open(args.config), Loader=SafeLoader)
    if args.dataset in config:
        config = config[args.dataset]
    else:
        config = {
        'learning_rate': 0.001,
        'num_hidden': 256,
        'num_proj_hidden': 256,
        'activation': 'prelu',
        'base_model': 'GCNConv',
        'num_layers': 2,
        'drop_edge_rate_1': 0.3,
        'drop_edge_rate_2': 0.4,
        'drop_feature_rate_1': 0.1,
        'drop_feature_rate_2': 0.0,
        'tau': 0.4,
        'num_epochs': 1000,
        'weight_decay': 1e-5,
        'drop_scheme': 'degree',
    }
    
        
    torch.manual_seed(config["seed"])
    random.seed(12345)
    np.random.seed(config["seed"])
    
    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU(), 'rrelu': nn.RReLU()})[config['activation']]
    base_model = GCNConv
    num_layers = config['num_layers']

    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']
    # switch to the customer inputs by using args.{}
    eps = config["eps"] # args.eps
    lamb = config["lamb"] # args.lamb
    alpha = config["alpha"] # args.alpha
    beta = config["beta"] # arg.sbeta
    
    
    sample_size = 1388 # new york(1388)
    # sample_size = 2234 #chicago
    
    def get_dataset(path, name):
        assert name in ['Cora', 'CiteSeer', "AmazonC", "AmazonP", 'CoauthorC', 'CoauthorP',\
                        "DBLP", "PubMed", "GitHub", "Facebook", "LastFMAsia", "DeezerEurope"]
        if name =="DBLP":
            name = "dblp"
        if name == "AmazonC":
            return Amazon(path, "Computers", T.NormalizeFeatures())
        if name == "AmazonP":
            return Amazon(path, "Photo", T.NormalizeFeatures())
        if name == 'CoauthorC':
            return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())
        if name == 'CoauthorP':
            return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())
        if name == "GitHub":
            return GitHub(root=path,transform=T.NormalizeFeatures())
        if name == "Facebook":
            return FacebookPagePage(root=path,transform=T.NormalizeFeatures())    
        if name == "LastFMAsia":
            return LastFMAsia(root=path,transform=T.NormalizeFeatures())
        if name == "DeezerEurope":
            return DeezerEurope(root=path,transform=T.NormalizeFeatures())

        return (CitationFull if name == 'dblp' else Planetoid)(
            path,
            name,
            "public",
            T.NormalizeFeatures())
        
    path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)
    # print("path:", path)
    # println
    dataset = get_dataset(path, args.dataset)
    # print("dataset:", dataset)
    data = dataset.data  
    # print(data.num_features)
    # println()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = Encoder(data.num_features, num_hidden, activation,
                      base_model=base_model, k=num_layers).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = t()
    prev = start    
    G = nx.Graph()
    G.add_edges_from(list(zip(data.edge_index.numpy()[0],data.edge_index.numpy()[1])))
    
    
    gnn_generative_1 = GNN(3, 96, JK="last", drop_ratio=0, gnn_type= "gcn")
    model_generative_1 = vgae(gnn_generative_1, 96)
    model_generative_1.to(device)
    optimizer_generative_1 = optim.Adam(model_generative_1.parameters(), lr=learning_rate, weight_decay=weight_decay)
    gnn_generative_2 = GNN(3, 96, JK="last", drop_ratio=0, gnn_type= "gcn")
    model_generative_2 = vgae(gnn_generative_2, 96)
    model_generative_2.to(device)
    optimizer_generative_2 = optim.Adam(model_generative_2.parameters(), lr=learning_rate, weight_decay=weight_decay)
 
    '''set training'''
    model.train(),model_generative_1.train()
    # import time
    # start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        # uncomment to increase the eps every T epochs
        #if epoch%20 ==0:
        #    eps = eps*1.1
        # sample a subgraph from the original one

        S = G.subgraph(np.random.permutation(G.number_of_nodes())[:sample_size])
        x = data.x[np.array(S.nodes())].to(device)
        # print("S.nodes():", S.nodes())
        # println()
        S = nx.relabel.convert_node_labels_to_integers(S, first_label=0, ordering='default')
        edge_index = np.array(S.edges()).T
        # print("S.edges():", S.edges())
        edge_index = torch.LongTensor(np.hstack([edge_index,edge_index[::-1]])).to(device)
        
        # println()
        # edge_attr = np.array(S.edges()).T
        # edge_index = torch.LongTensor(np.hstack([edge_index,edge_index[::-1]])).to(device)

        loss1, loss2, loss3 = train(model, x, edge_index, eps, model_generative_1,optimizer_generative_1,model_generative_2,optimizer_generative_2, lamb, alpha, beta, 5, 0.2)
             
        now = t()                                     
        print(f'(T) | Epoch={epoch:03d}, loss1={loss1:.4f}, loss2={loss2:.4f}'
              f' this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now
    # end_time = time.time()
    # print("during time:", (end_time-start_time)/300)
    # printnl()
    print("=== Final ===")
    results_1 = test(model, data.x, data.edge_index, model_generative_1,data.y, final=True)
    print(results_1)
    with open(osp.join(args.log, "progress.csv"), "w") as f:
        f.write(str(results_1))
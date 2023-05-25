# -*- coding: utf-8 -*-

import torch
from torch_geometric.data import Data
from itertools import product
import numpy as np
import pandas as pd
from torch import nn
import pickle

import warnings
warnings.filterwarnings("ignore")

def load_data(file):
        data_load_file = []
        file_1 = open(file, "rb")
        data_load_file = pickle.load(file_1)
        return data_load_file
        
# tmp_vector = load_data("./data/tmp_vector.pickle")

# tmp_vector = load_data("./data/tmp_vector_vgae_au.pickle")

# tmp_vector = load_data("./data/tmp_vector_vgae_3.pickle")
tmp_vector = load_data("./data/tmp_vector_chi_3.pickle")
linear = nn.Linear(128, 16)
ten = torch.tensor(tmp_vector.tolist())
# print(ten.size())
# final = []
# for item in ten:
#     tmp = linear(item).tolist()
#     final.append(tmp)
# final_vector = torch.tensor(final)
# print(final_vector.size())

# hy = load_data("./data/hy_new_aaai_2.pickle")

hy = load_data("./data/hy_aaai_chi_1.pickle")
region = ten

nn_1 = nn.Linear(128,96)
hy_nodes_dict={}
for n,n_vec in zip(hy.nodes(),region):
    tp = n.split("_")[1]
    if tp not in hy_nodes_dict.keys():
        hy_nodes_dict[int(tp)] = []
        hy_nodes_dict[int(tp)].append(n_vec.tolist())
    else:
        hy_nodes_dict[int(tp)].append(n_vec.tolist())

 
hy_com  = {}
for key,value in hy_nodes_dict.items():
    tmp = np.mean(value, axis=0).tolist()
    tmp_ = torch.tensor(tmp).tolist()
    hy_com[int(key)]  = tmp_

linear = nn.Linear(128, 16)
hycom_vec = []
for key,value in hy_com.items():
    hycom_vec.append(linear(torch.tensor(value)).tolist())
vec_final_ = np.reshape(np.tile(np.array(hycom_vec),(30,4)),(234,30,4,16))

print(np.array(hycom_vec).shape)
print("vec_final_:",vec_final_.shape)
file=open(r"./data/tmp_7.pickle","wb")
pickle.dump(vec_final_,file) #storing_list
file.close()
file=open(r"./data/tmp_house.pickle","wb")
pickle.dump(np.array(hycom_vec),file) #storing_list
file.close()



'''transform for traffi prediction--32 dimension vector'''
trans_1 = np.reshape(np.array(hycom_vec), 234*16)
linear_traffic = nn.Linear(234*16, 32)
traffic_vec = linear_traffic(torch.tensor(trans_1).float()).detach().numpy()
# print(traffic_vec.size())
file=open(r"./data/traff_vec.pickle","wb")
pickle.dump(traffic_vec,file) #storing_list
file.close()

print("---finish---")








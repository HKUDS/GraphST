# -*- coding: utf-8 -*-
import pandas as pd
import pickle
from shapely.geometry import Point, LineString
from shapely.geometry import Polygon,MultiPoint  #多边形
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim


def load_data(file):
    data_load_file = []
    file_1 = open(file, "rb")
    data_load_file = pickle.load(file_1)
    return data_load_file
# poi_list = ['drinking_water', 'toilets', 'school', 'hospital', 'arts_centre', 'fire_station', 'police', 'bicycle_parking', 'fountain', 'ferry_terminal', 'bench', 'cinema', 'cafe', 'pub', 'waste_basket', 'parking_entrance', 'parking', 'fast_food', 'bank', 'restaurant', 'ice_cream', 'pharmacy', 'taxi', 'post_box', 'atm', 'nightclub', 'social_facility', 'bar', 'biergarten', 'clock', 'bicycle_rental', 'community_centre', 'watering_place', 'ranger_station', 'boat_rental', 'recycling', 'payment_terminal', 'bicycle_repair_station', 'place_of_worship', 'shelter', 'telephone', 'clinic', 'dentist', 'vending_machine', 'theatre', 'charging_station', 'public_bookcase', 'post_office', 'fuel', 'doctors']
# poi_list_1 = ['drinking_water', 'toilets', 'school', 'hospital', 'arts_centre', 'fire_station', 'police', 'bicycle_parking', 'fountain', 'ferry_terminal', 'bench', 'cinema', 'cafe', 'pub', 'waste_basket', 'parking_entrance', 'parking', 'fast_food', 'bank', 'restaurant', 'ice_cream', 'pharmacy', 'taxi', 'post_box', 'atm', 'nightclub', 'social_facility', 'bar', 'biergarten', 'clock', 'bicycle_rental', 'community_centre', 'watering_place', 'ranger_station', 'boat_rental', 'recycling', 'payment_terminal', 'bicycle_repair_station', 'place_of_worship', 'shelter', 'telephone', 'clinic', 'dentist', 'vending_machine', 'theatre', 'charging_station', 'public_bookcase', 'post_office', 'fuel', 'doctors','drinking_water', 'toilets']
region_back = load_data("../data/region_back.pickle")
reg_poi = load_data("../data/reg_incld_poi_new.pickle")
poi_skip_vec = load_data("../data/poi_skip_vec.pickle")
reg_spatial = load_data("../data/region_spatial_refine_1.pickle")
flow = load_data("../data/flow_graph.pickle")
check_in_label = load_data("../data/checkin_label.pickle")
flow_list = list(flow.edges(data=True))
def normalization(data):
    _range = np.max(abs(data))
    return data / _range
label_norm = normalization(check_in_label)
# print(label_norm)
# final_vec =[]
# connected_layer = nn.Linear(in_features = 200, out_features = 96)
# emb = nn.Embedding(200, 200)
# for key,value in reg_poi.items():
#     output = np.mean([connected_layer(emb(torch.tensor(uu)).float()).tolist() for uu in value],axis=0).tolist()
#     final_vec.append(output)
# print(np.array(final_vec).shape)

# file=open(r"../data/reg_poi_vec.pickle","wb")
# pickle.dump(final_vec,file) #storing_list
# file.close()

# println()
# reg_flow = {}
# for item in flow_list:
#     # print(item)
#     # print(item[2]['weight'])
#     # println()
#     r1 = item[0].split("_")[1]
#     r2 = item[1].split("_")[1]
#     if int(r1) not in reg_flow.keys():
#         reg_flow[int(r1)] = 0
#     if int(r2) not in reg_flow.keys():
#         reg_flow[int(r2)] = 0
#     reg_flow[int(r1)]+= item[2]['weight']
#     reg_flow[int(r2)]+= item[2]['weight']


# println()

reg_idx = [key for key in reg_poi.keys() if len(reg_poi[key])>0]
# print(reg_idx)
# prirntln()
file=open(r"../data/reg_poi_idx_1.pickle","wb")
pickle.dump(reg_idx,file) #storing_list
file.close()



max_len= 0
for key,value in reg_poi.items():
    if max_len< len(value):
        max_len = len(value)
# print("max_len:",max_len)

reg_poi_t = {}
reg_poi_list = []

embedding_cat = torch.nn.Embedding(11, 96)  # spatial
linear = nn.Linear(96*3, 512)
linear_trans = nn.Linear(512, 96)
region_com_list = []
region_poi_gram_dict  = {}
for iii in range(172):
    if iii not in reg_idx:
        # reg_poi_t[key] = np.array([0.0]*96)
        tmp_1 = np.array([0.0]*96)
        region_poi_gram_dict[iii] = tmp_1.tolist()
        tmp_2 = embedding_cat(torch.tensor(0))
        # tmp_3 = torch.squeeze(reg_spatial[iii],0).tolist()
        com = np.concatenate((tmp_1,tmp_2.tolist(),[label_norm[iii]]*96),axis = 0)
        # print("com.shape:",com.shape)
        com_reshape = linear(torch.tensor(com).float()).tolist()
        region_com_list.append(com_reshape)    
    else:
        tmp_g = []
        # print(reg_poi[iii])
        for sub_poi in reg_poi[iii]:
            tmp_g.append(poi_skip_vec[sub_poi].tolist())
        tmp_1 = np.mean(tmp_g, axis =0)
        # region_poi_gram.append(tmp_1.tolist())
        region_poi_gram_dict[iii] = tmp_1.tolist()
        tmp_2 = embedding_cat(torch.tensor(len(reg_poi[iii])))
        # tmp_3 = torch.squeeze(reg_spatial[iii],0).tolist()
        com = np.concatenate((tmp_1,tmp_2.tolist(),[label_norm[iii]]*96),axis = 0)
        com_reshape = linear(torch.tensor(com).float()).tolist()
        region_com_list.append(com_reshape)
        
region_poi_gram = []
for key,value in region_poi_gram_dict.items():
    # print(value)
    region_poi_gram.append(value)
file=open(r"../data/reg_poi_vec_2.pickle","wb")
pickle.dump(torch.tensor(region_poi_gram),file) #storing_list
file.close()
# println()        
  
region_com_array = np.array(region_com_list)
# print("region_com_array:",region_com_array.shape)       
# reg_idx= [key for key in reg_poi_.keys()]
from torch import nn
encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8 )
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
# src = torch.rand(1, 172, 512)
src =  torch.unsqueeze(torch.tensor(region_com_array),0)
out = transformer_encoder(src.float())
out_ = torch.squeeze(out,0)
# print(out_.size())
# pritnln()
out_ = torch.tensor([linear_trans(item).tolist() for item in out_])
# print(out_.size())

# reg_poi_vec = {}
# for idx,vec in zip(reg_idx,out_):
#     reg_poi_vec[idx] = vec

file=open(r"../data/reg_com_poi_cat_spatial.pickle","wb")
pickle.dump(out_,file) #storing_list
file.close()


         
        
println()


# reg_poi_={}
# s = 0
# emb = nn.Embedding(50, 512)
# embedding_spatial = torch.nn.Embedding(15, 512)  # spatial
# for key,value in reg_poi.items():
#     # print("value:",value)
#     if value!=[]:
#         reg_poi_[key]=[]
#         # print("value:",value)
#         if len(value)>s:
#             s = len(value)
#         for item in value:
#             reg_poi_[key].append(emb(torch.tensor(item)).tolist())
# spa_vec= embedding_spatial(torch.tensor(reg_spatial[idx]))
# reg_poi_t = {}
# reg_poi_list = []
# for iii in range(172):
# # for key,value in reg_poi_.items():
#     if iii not in reg_poi_.keys():
#         # print("&&&:", np.array([0.0]*512).shape)
#         reg_poi_t[key] = np.array([0.0]*512)
#         # spa_vec= embedding_spatial(torch.tensor(reg_spatial[iii])).tolist()
#         # ci = np.concatenate((spa_vec,[0.0]*512),axis = 0)
#         reg_poi_list.append(np.array([0.0]*512))
#         # reg_poi_list.append(ci)
#     else:
#         # print("value:",value)
#         tp = np.mean(reg_poi_[key],axis=0)
#         # spa_vec= embedding_spatial(torch.tensor(reg_spatial[iii])).tolist()
#         # ci = np.concatenate((spa_vec,tp.tolist()),axis = 0)
#         # reg_poi_list.append(np.array([0.0]*512))
#         reg_poi_list.append(tp)
#         # print("***:",tp.shape)
#         reg_poi_t[key] = tp
        # reg_poi_list.append(tp)
# print(np.array(reg_poi_list).shape)
# for key,value in reg_poi.items():
#     # print("value:",value)
#     if value!=[]:
#         reg_poi_[key]=[]
#         # print("value:",value)
#         if len(value)>s:
#             s = len(value)
#         for item in value:
#             reg_poi_[key].append(emb(torch.tensor(item)).tolist())
# # spa_vec= embedding_spatial(torch.tensor(reg_spatial[idx]))
# # reg_poi_t = {}
# reg_poi_list = []
# for iii in range(172):
# # for key,value in reg_poi_.items():
#     if iii not in reg_poi_.keys():
#         # print("&&&:", np.array([0.0]*512).shape)
#         # reg_poi_t[key] = np.array([0.0]*512)
#         spa_vec= embedding_spatial(torch.tensor(reg_spatial[iii])).tolist()
#         ci = np.concatenate((spa_vec,[0.0]*512),axis = 0)
#         # reg_poi_list.append(np.array([0.0]*512))
#         reg_poi_list.append(ci)
#     else:
#         # print("value:",value)
#         tp = np.mean(reg_poi_[key],axis=0)
#         spa_vec= embedding_spatial(torch.tensor(reg_spatial[iii])).tolist()
#         ci = np.concatenate((spa_vec,tp.tolist()),axis = 0)
#         # reg_poi_list.append(np.array([0.0]*512))
#         reg_poi_list.append(ci)
#         # print("***:",tp.shape)
#         # reg_poi_t[key] = tp
#         # reg_poi_list.append(tp)
# # print(np.array(reg_poi_list).shape)
# fully_layer = nn.Linear(1024,512)
# reg_poi_list_ = fully_layer(torch.tensor(np.array(reg_poi_list)).float())
reg_poi_list_ = torch.tensor(np.array(reg_poi_list)).float()
reg_poi_list_tensor = torch.unsqueeze(reg_poi_list_,0)
print(reg_poi_list_tensor.size())

reg_idx= [key for key in reg_poi_.keys()]
from torch import nn
encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8 )
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
# src = torch.rand(1, 172, 512)
src = reg_poi_list_tensor
out = transformer_encoder(src)
# print(out.size())
out_ = torch.squeeze(out,0)
print(out_.size())
print(reg_idx)
print(len(reg_idx))
# reg_poi_vec = {}
# for idx,vec in zip(reg_idx,out_):
#     reg_poi_vec[idx] = vec

file=open(r"../data/reg_poi_vec_1.pickle","wb")
pickle.dump(out_,file) #storing_list
file.close()

file=open(r"../data/reg_poi_idx_1.pickle","wb")
pickle.dump(reg_idx,file) #storing_list
file.close()
        






    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


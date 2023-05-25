# -*- coding: utf-8 -*-

import pickle
import pandas as pd
import numpy as np
import copy
from shapely.geometry import Point, LineString
from shapely.geometry import Polygon,MultiPoint  #多边形
import torch
import networkx as nx
import matplotlib.pyplot as pl




def load_data(file):
        data_load_file = []
        file_1 = open(file, "rb")
        data_load_file = pickle.load(file_1)
        return data_load_file


reg_vec_sort = load_data("../data/reg_poi_vec_2.pickle")
region_que = load_data("../data/reg_poi_idx_1.pickle")

region_attr_edges=[]
# region_que
# for idx in range(len(reg_vec_sort)):
# 	for idt in range(idx+1, len(reg_vec_sort)):
# 		output = torch.cosine_similarity(reg_vec_sort[idx], reg_vec_sort[idt], eps=1e-08).mean()
# 		if output>=0.8:
# 			tmp_1 = "r" + '_' + str(idx)
# 			tmp_2 = "r" + '_' + str(idt)
# 	        # sim_dict[key] = [tmp_1, tmp_2, value]
# 			region_attr_edges.append([tmp_1, tmp_2, output.item()])
# print("reg_vec_sort:",len(reg_vec_sort))
# print("region_que:",region_que)
# print(reg_vec_sort[170])

# pritnnl()

for idx in region_que:
    for idt in range(idx+1, len(reg_vec_sort)):
        # print("^^:",reg_vec_sort[idx].size())
        # print("**:",reg_vec_sort[idx+1].size())
        # pritnln()
        output = torch.cosine_similarity(torch.unsqueeze(reg_vec_sort[idx],0), torch.unsqueeze(reg_vec_sort[idt],0), eps=1e-08).mean()
        # print("output:", output.item())
        # pritnln()
        if output.item()>=0.850:
            tmp_1 = "r" + '_' + str(idx)
            tmp_2 = "r" + '_' + str(idt)
            # sim_dict[key] = [tmp_1, tmp_2, value]
            region_attr_edges.append([tmp_1, tmp_2, output.item()])
# print(len(region_attr_edges))
# println()
G = nx.Graph()
# for edge in edges:
#     G.add_edge(edge[0],edge[1],weight= edge[2])

[G.add_edge(edge[0],edge[1],weight= edge[2], date = "1", start = edge[0], end = edge[1] ) for edge in region_attr_edges]
# print(len(G.adj))
# nx.draw(G, with_labels=True)
# plt.show()


file=open(r"../data/region_attr_graph.pickle","wb")
pickle.dump(G,file) #storing_list
file.close()

print("attr_region:", G)
# similarity_dict = {}
# similarity_list = []
# for ii in range(emb.size()[0]):
#     # print(emb[ii])
#     # print(emb[ii].shape)
#     # print(ii.shape)
#     for jj in range(ii+1, emb.size()[0]):
#         # print(emb[jj])
#         # print(emb[jj].shape)
#         output = torch.cosine_similarity(emb[ii], emb[jj], eps=1e-08).mean()
#         # print("similarity:", output.item())
#         # println()
#         similarity_list.append(output.item())
#         tmp = 'r_{}_{}'.format(ii, jj)
#         similarity_dict[tmp] = output.item()
# similarity_list.sort(reverse = True)
# # print(similarity_list)
# print(len(similarity_list))
# sum_1 = 0
# for item in similarity_list:
#     if item>=1.0:
#         sum_1+=1
# print(sum_1)
# print(sum_1/len(similarity_list))
# print(similarity_dict)
# sim_dict = {}
# edges = []
# for key,value in similarity_dict.items():
#     tmp = key.split('_')
#     # print("tmp:", tmp)
#     # print(tmp[0] + '_' + tmp[1])
#     # print(tmp[0] + '_' + tmp[2])
#     if value >=0.8:
#         tmp_1 = tmp[0] + '_' + tmp[1]
#         tmp_2 = tmp[0] + '_' + tmp[2]
#         sim_dict[key] = [tmp_1, tmp_2, value]
#         edges.append([tmp_1, tmp_2, value])

# print(len(edges))
# # println()
# G = nx.Graph()
# # for edge in edges:
# #     G.add_edge(edge[0],edge[1],weight= edge[2])

# [G.add_edge(edge[0],edge[1],weight= edge[2], date = "1", start = edge[0], end = edge[1] ) for edge in edges]
# # print(len(G.adj))
# nx.draw(G, with_labels=True)
# plt.show()


# file=open(r"../data/region_attr_sim_graph_{}.pickle".format(resolution),"wb")
# pickle.dump(G,file) #storing_list
# file.close()
































































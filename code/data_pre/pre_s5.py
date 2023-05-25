# -*- coding: utf-8 -*-

import pickle
import pandas as pd
from itertools import chain
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt


def load_data(file):
    data_load_file = []
    file_1 = open(file, "rb")
    data_load_file = pickle.load(file_1)
    return data_load_file
from scipy.sparse import csr_matrix

flow_g = load_data('../data/flow_graph.pickle')
spatial_g = load_data('../data/spatial_graph.pickle')
region_attr_g = load_data('../data/region_attr_graph.pickle')
# adj_matrix = csr_matrix((np.ones(len(row)), (row, col)), shape=(len(node_map), len(node_map)))
# print(np.array(nx.adjacency_matrix(region_attr_g).todense()))
# adj_= np.array(nx.adjacency_matrix(region_attr_g).todense()).tolist()
# adj_gat = csr_matrix((np.ones(), np.array(nx.adjacency_matrix(region_attr_g).todense())), shape=(62, 62))
# print(adj_gat)
# print(type(adj_gat))

# println
# feature = np.random.uniform(-1, 1, size=(62, 62))
# feature = feature[np.newaxis]
# print(feature.shape)
# println()

# file=open(r"../data/adj_gat.pickle","wb")
# pickle.dump(adj_gat,file) #storing_list
# file.close()
# file=open(r"../data/fea_gat.pickle","wb")
# pickle.dump(feature,file) #storing_list
# file.close()
# println()
##only get region attrbutes matrix
# f = open('../data/poi_edgelist.txt','a')
# for item in region_attr_g.edges():
#     print(item[0].split("_")[1]," ", item[1].split("_")[1])
#     f.write('\n')
#     f.write(str(item[0].split("_")[1]))
#     f.write(" ")
#     f.write(str(item[1].split("_")[1]))
# f.close()
# adj=np.array(nx.adjacency_matrix(region_attr_g).todense())
# # # print(adj)
# f = open('../data/adjlist.txt','a')
# for item in adj:
#     # print(item)
#     # print(item.shape)
#     # print(item)
    
#     f.write('\n')
#     for sub in item:
#         # print(sub)
#         f.write(str(0))
#         f.write(" ")
# f.close()
# f = open('../data/labels.txt','a')
# for idx, ir in enumerate(region_attr_g.nodes()):
#     # print(idx, ir)
#     f.write('\n')
#     f.write(ir.split("_")[1])
#     f.write(" ")
#     f.write("1")
# f.close()
# f = open('../data/features.txt','a')
# for idx, ir in enumerate(region_attr_g.nodes()):
#     # print(idx, ir)
#     f.write('\n')
#     f.write(ir.split("_")[1])
#     f.write(" ")
#     f.write("1")
# f.close()
# println()


# print(flow_g,spatial_g,region_attr_g)

# print(flow_g.edges())

flow_nodes = list(flow_g.nodes)
spatial_nodes = list(spatial_g.nodes)
regat_nodes = list(region_attr_g.nodes)
flow_edges = list(flow_g.edges(data=True))
# print(flow_edges)
# println()
spatial_edges = list(spatial_g.edges(data=True))
# print(spatial_edges)
# println()
regat_edges = list(region_attr_g.edges(data=True))
# print(regat_edges)
# println()

part_f = flow_nodes
part_s = spatial_nodes
part_r = regat_nodes
# print(part_f)
# print("--------------------------")
# print(part_s)
# print("--------------------------")
# print(part_r)

hy_edges = []
for sub in regat_nodes:
    for ss in spatial_nodes:
        tmp = ss.split("_")
        tmp_c = tmp[0]+'_'+tmp[1]
        if sub == tmp_c:
            pair = (sub, ss,{"weight":0, "date": tmp[2], "start":sub, "end":ss})
            # print("pair:", pair)
            hy_edges.append(pair)

for ss in spatial_nodes:
    for ff in flow_nodes:
        tps = ss.split("_")
        # tps_c = tps[0]+'_'+tps[1]
        # tpf = ff.split("_")
        # tpf_c = tpf[0]+'_'+tpf[1]
        if ss == ff:
            pair = (ss, ff,{"weight":0, "date":tps[2] , "start":ss, "end":ff})
            # print("pair:", pair)
            hy_edges.append(pair)
      

# print("hy_edges:",hy_edges)
# hy_edges.extend(flow_edges)
# hy_edges.extend(spatial_edges)
# hy_edges.extend(regat_edges)
 
G_hy = nx.Graph()
G_hy.add_edges_from(hy_edges)
G_hy.add_edges_from(flow_edges)
G_hy.add_edges_from(spatial_edges)
G_hy.add_edges_from(regat_edges)
# nx.draw(G_hy)
# plt.show()
print("hyper_grapgh:", G_hy)
# println()
# fl = nx.Graph()
# fl.add_edges_from(flow_edges[])
# fl.add_edges_from(spatial_edges[:30])
# print(fl.nodes())

# sum_1=0
# for node_1, node_2, attr_dict in G_hy.edges(data=True):
#     if attr_dict=={}:
#         sum_1+=1
        # print(attr_dict)
# print("sum_1:", sum_1)
# printlist(G_hy.edges)      
# println()
 
nodes_num = 3
file=open(r"../data/hy_{}.pickle".format(8),"wb")
pickle.dump(G_hy,file) #storing_list
file.close()
# file=open(r"../data/fl_sp.pickle","wb")
# pickle.dump(fl,file) #storing_list
# file.close()

# adj_fl=np.array(nx.adjacency_matrix(fl).todense())
# f = open('../data/fl_edges.txt','a')
# for item in fl.edges:
#     # print(item)
#     # print(item[0].split("_")[1]," ", item[1].split("_")[1])
#     f.write(str(item[0].split("_")[1]))
#     f.write(" ")
#     f.write(str(item[1].split("_")[1]))
#     f.write('\n')
# f.close()
# f = open('../data/fl_labels.txt','a')
# for idx, ir in enumerate(fl.nodes()):
#     # print(idx, ir)
#     # println
#     f.write(ir.split("_")[1])
#     f.write(" ")
#     f.write("1")
#     f.write('\n')
# f.close()


#for mvure
# adj=np.array(nx.adjacency_matrix(region_attr_g).todense())
# mob_adj = adj[np.newaxis,:]
# # print("t_adj:", adj.shape)
# np.save("../data/mvure_data/mob-adj.npy", mob_adj)

# np.save("../data/mvure_data/s_adj.npy", adj)


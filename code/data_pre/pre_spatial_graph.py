#!/usr/bin/env python3

import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString
from shapely.geometry import Polygon,MultiPoint  #多边形
import matplotlib.pyplot as plt
import json
from urllib.request import urlopen, quote
import requests
import geopy
from geopy.geocoders import Nominatim
import copy
import pickle
from datetime import datetime
from itertools import chain
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt
from sklearn.cluster import DBSCAN

def haversine(lon1, lat1, lon2, lat2): # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # 地球平均半径，单位为公里
    return c * r * 1000

def load_data(file):
        data_load_file = []
        file_1 = open(file, "rb")
        data_load_file = pickle.load(file_1)
        return data_load_file
    
region_back = load_data("../data/region_back_merge.pickle")
region_fea = load_data("../data/region_fea.pickle")
spatial_edges = []
# spatial_edges.extend(flow_edges) # add edges in flow graph
# sim_num=0
X = [[list(value.centroid.coords)[0][0],list(value.centroid.coords)[0][1]] for key,value in region_back.items()]
# y_pred = DBSCAN(eps = 0.01, min_samples = 7).fit_predict(X)
# from sklearn.cluster import SpectralClustering
# sc = SpectralClustering(3, affinity='precomputed', n_init=100,assign_labels='discretize')
# y_pred = sc.fit_predict(X)  
# print(y_pred)
# println()
# # print(check_index)
node=[]
# # reg={}
# for i in range(12):
#     reg[i] = 0
# # print(reg)
# tmp=[]
# for key,value in region_fea.items():
#     if value==0:
#         # print(key,value, list(region_back[key].centroid.coords))
#     #     tmp.append([list(region_back[key].centroid.coords)[0],key])
#         print(key)
#         # shapely.ops.unary_union(polygons)
#         reg[value]+=1
#     # println()
# print(reg)
# def takeSecond(elem):
#     return elem[0][0]
 
# random = tmp
# # 指定第二个元素排序
# random.sort(key=takeSecond)
# print(random)
# print(len(random))

# println()

reg_spatial={}
for ii in range(180):
    for jj in range(ii+1, 180):
        # time = flow_nodes[ii].split("_")[2]
        # t_1 = flow_nodes[ii].split("_")
        # t_2 = flow_nodes[jj].split("_")
        t_1 = ii
        t_2 = jj
        # print("t_1:",t_1)
        # print("t_2:",t_2)
        if int(t_1) not in node:
            node.append(int(t_1))
        if int(t_2) not in node:
            node.append(int(t_2))
        t_1_pos = list(region_back[int(t_1)].centroid.coords)[0]
        t_2_pos = list(region_back[int(t_2)].centroid.coords)[0]
        value = haversine(t_1_pos[0], t_1_pos[1], t_2_pos[0], t_2_pos[1])
        if value<= 2900:  #小于5公里
            n1 = "r"+"_"+str(t_1)
            n2 = "r"+"_"+str(t_2)
            # pair = (n1,n2, {"weight":value, "date":int(1), "start":n1, "end":n2})
            pair = (n1,n2, {"weight":1, "date":int(1), "start":n1, "end":n2})
            # print(pair)
            if pair not in spatial_edges:
                spatial_edges.append(pair)
# region_fea[283]=12
# for ii in range(296):
#     for jj in range(ii+1, 296):
#         t_1 = ii
#         t_2 = jj
#         if int(t_1) not in node:
#             node.append(int(t_1))
#         if int(t_2) not in node:
#             node.append(int(t_2))
#         if region_fea[ii]==region_fea[jj]:
#             n1 = "r"+"_"+str(t_1)
#             n2 = "r"+"_"+str(t_2)
#             pair = (n1,n2, {"weight":1, "date":int(1), "start":n1, "end":n2})
#             if pair not in spatial_edges:
#                 spatial_edges.append(pair)
#         if region_fea[ii]==region_fea[jj]+1 or region_fea[ii]==region_fea[jj]+2 or region_fea[ii]==region_fea[jj]+3 or region_fea[ii]==region_fea[jj]+4 or region_fea[ii]==region_fea[jj]+5 or region_fea[ii]==region_fea[jj]+6 or region_fea[ii]==region_fea[jj]+7 or region_fea[ii]==region_fea[jj]+8 or region_fea[ii]==region_fea[jj]+9 or region_fea[ii]==region_fea[jj]+10 or region_fea[ii]==region_fea[jj]+11:
#             # print("ii:",region_fea[ii])
#             # print("jj:",region_fea[jj])
#             n1 = "r"+"_"+str(t_1)
#             n2 = "r"+"_"+str(t_2)
#             pair = (n1,n2, {"weight":1, "date":int(1), "start":n1, "end":n2})
#             if pair not in spatial_edges:
#                 spatial_edges.append(pair)
#                 continue
# print(spatial_edges)
# print(len(spatial_edges))
# println()


# println()
print("spatial_edges:",spatial_edges)
print(len(spatial_edges))
print("finish spatial graph")


# #spatial graph
G_spatial = nx.Graph()
G_spatial.add_edges_from(spatial_edges[:])
# nx.draw(G_spatial, with_labels=True)
# plt.show()
print("G_spatial:",G_spatial)

file=open(r"../data/spatial_graph_baseline.pickle","wb")
pickle.dump(G_spatial,file) #storing_list
file.close()
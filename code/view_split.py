# -*- coding: utf-8 -*-

import pickle
import numpy as np
import pandas as pd

def load_data(file):
        data_load_file = []
        file_1 = open(file, "rb")
        data_load_file = pickle.load(file_1)
        return data_load_file
hy = load_data("./data/hy_new_aaai_2.pickle")

node_list = list(hy.nodes)
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
print(len(poi_view))
print(len(spatial_view)) 
print(len(flow_view)) 
print(len(poi_view)+len(spatial_view)+len(flow_view))     
# print(poi_view)
# print(flow_view)
























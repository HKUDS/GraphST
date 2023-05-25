# -*- coding: utf-8 -*-

import pandas as pd
import pickle
from shapely.geometry import Point, LineString
from shapely.geometry import Polygon,MultiPoint  #多边形
import torch
from torch import nn
import numpy as np

def load_data(file):
    data_load_file = []
    file_1 = open(file, "rb")
    data_load_file = pickle.load(file_1)
    return data_load_file

view_graph = load_data("./data/hy_new_aaai_2.pickle")
region_lab = load_data("./data/region_label.pickle")
# print(region_lab)
# print(region_lab.keys())
# print(type(region_lab.keys()))
# pritnln()
nodes_list = view_graph.nodes()
nodes_lab = []
for item in nodes_list:
    # print("item:", item)
    idx = int(item.split("_")[1])
    if idx in region_lab.keys():
        tmp_lab = region_lab[idx]
        nodes_lab.append(tmp_lab)
    else:
        nodes_lab.append(5)
# print("nodes_lab:",nodes_lab)
data_num = 1388
file=open(r"./data/nodes_lab.pickle","wb")
pickle.dump(nodes_lab,file) #storing_list
file.close()
'''train mask'''
l1 = [True]*int(0.051698670605613*data_num)
l2 = [False]*(data_num-int(0.051698670605613*data_num))
l1.extend(l2)
tmp_len = int(0.051698670605613*data_num)
train_mask = l1
# print(tmp_len)
file=open(r"./data/train_mask.pickle","wb")
pickle.dump(train_mask,file) #storing_list
file.close()
'''val mask'''
l3 = [True]*int(0.18463810930576072*data_num)
l4 = [False]*(data_num-int(0.18463810930576072*data_num)-tmp_len)
l5 = [False]* tmp_len
l5.extend(l3)
l5.extend(l4)
tmp_len_val = len(l3)
val_mask = l5
file=open(r"./data/val_mask.pickle","wb")
pickle.dump(val_mask,file) #storing_list
file.close()
# print(len(val_mask))
'''test mask'''
l6 = [True]*int(0.36927621861152143*data_num)
l7 = [False]*(data_num-int(0.36927621861152143*data_num)-tmp_len-tmp_len_val)
l8 = [False]* (tmp_len+tmp_len_val)
l8.extend(l6)
l8.extend(l7)
test_mask = l8
file=open(r"./data/test_mask.pickle","wb")
pickle.dump(test_mask,file) #storing_list
file.close()

print("---Done---")























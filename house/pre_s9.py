# -*- coding: utf-8 -*-
"""
Created on Thu May 19 19:33:25 2022

@author: User
"""
# this file is to predict hosue price
import pandas as pd
import pickle
from shapely.geometry import Point, LineString
from shapely.geometry import Polygon,MultiPoint  #多边形
import torch
from torch import nn
import networkx as nx
import numpy as np
from sklearn.linear_model import Lasso,Ridge
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
import numpy as np
# import mglearn
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn import svm
from sklearn import metrics


def load_data(file):
    data_load_file = []
    file_1 = open(file, "rb")
    data_load_file = pickle.load(file_1)
    return data_load_file

# region_vec = load_data("../data/hy_com_dict.pickle")
# gf_vec = load_data("../data/vec/gf.pickle")
# grarep_vec = load_data("../data/vec/grarep.pickle")
# hope_vec = load_data("../data/vec/hope.pickle")
# hdge_vec = load_data("../data/vec/hdge.pickle")
# n2v_vec = load_data("../data/vec/n2v.pickle")
# sdne_vec = load_data("../data/vec/sdne.pickle")
# poi_vec = load_data("../data/vec/poi.pickle")
train_set = load_data("../data/train_house.pickle") 
test_set = load_data("../data/test_house.pickle")
emb_dim = 16
embedding_1 = nn.Embedding(1000000, emb_dim)
linear = nn.Linear(144, 16)
# tmp_vec = load_data("../data/baseline/gae_vector.pickle")
# print(type(tmp_vec))
# printljn()
# for name in ['gcn','gae','gat','graphsage','cgail','mgfn','mvpn','mvure','zemob']:
# for name in ['hdge','n2v','zemob','gae']:
# for name in ['gcn','gae','gat','graphsage','cgail','mgfn','mvpn']:    
for name in ['mvure']:
    tmp_vec = load_data("../data/baseline/{}_vector.pickle".format(name))
    # print(tmp_vec.shape)
    # println()
    train_vec = []
    train_y =[]
    for item in train_set:
        # print(item)
        # printl()
        tmp=[]
        # tmp.extend(linear(torch.tensor(tmp_vec[item[0]])).tolist())
        tmp.extend(linear(torch.tensor(tmp_vec[item[0]]).float()).tolist())
        tmp.extend(embedding_1(torch.tensor(item[1])).tolist())
        train_y.append(item[-1])
        train_vec.append(tmp)
    # print(train_vec)
    # print(train_y)
    # println()
    test_vec= []
    test_y =[]
    for item in test_set:
        tmp=[]
        tmp.extend(linear(torch.tensor(tmp_vec[item[0]]).float()).tolist())
        tmp.extend(embedding_1(torch.tensor(item[1])).tolist())
        # tmp.append(item[0])
        # tmp.append(item[1])
        test_y.append(item[-1])
        test_vec.append(tmp)
    
    # lasso00001 = Lasso(alpha=0.00001).fit(test_vec,test_y)
    lasso01 = Lasso(alpha=0.00001).fit(test_vec,test_y)
    y_pred_lasso=lasso01.fit(train_vec,train_y).predict(test_vec)
    # print(len(y_pred_lasso))
    # print(test_y)
    r2_score_lasso=r2_score(test_y,y_pred_lasso)
    # print(mean_absolute_error(test_y,y_pred_lasso))
    # print(r2_score_lasso)
    test_y_ = []
    y_pred_lasso_= []
    for i,j in zip(test_y,y_pred_lasso):
        if i!=0:
            test_y_.append(i)
            y_pred_lasso_.append(j)
    
    # y = np.array([1,1])
    # y_hat = np.array([2,3])
    MSE = metrics.mean_squared_error(test_y,y_pred_lasso)
    RMSE = metrics.mean_squared_error(test_y,y_pred_lasso)**0.5
    MAE = metrics.mean_absolute_error(test_y,y_pred_lasso)
    MAPE = metrics.mean_absolute_percentage_error(test_y_,y_pred_lasso_)
    print("{}:".format(name), MSE,RMSE,MAE,MAPE,r2_score_lasso)
    # print(test_y)
    print('**********************************')
println()
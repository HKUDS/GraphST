import torch
import torch
# import networkx as nx
import matplotlib.pyplot as pl
import pickle
import pandas as pd
import numpy as np
import math
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
import time


def load_data(file):
        data_load_file = []
        file_1 = open(file, "rb")
        data_load_file = pickle.load(file_1)
        return data_load_file
house =  load_data("../data/NY_house.pickle")
region_old = load_data("../data/NY_region.pickle")
region_new = load_data("../data/region_back_merge.pickle")

region_map = {}
for key,value in region_old.items():
    tmp = [(item[1],item[0]) for item in value if item!=[]]
    if len(tmp)>=3:
        tmp_ = Polygon(tmp)
        for k,v in region_new.items():
            if tmp_.intersects(v):
                 region_map[key]=k
                 break
    else:
        tmp_ = Point(tmp)
        for k,v in region_new.items():
            if tmp_.intersects(v):
                 region_map[key]=k
                 break



# print(region_map)
# print(len(region_old))
# print(len(region_map))
# println()

# region_ = {}
# for key,value in region.items():
#     if [] not in value and len(value)>=3:
#         region_[key] = value
# region_back = {}
# map_region = {}
# for idx, tt in enumerate(region_.items()):
#     # print(tt)
#     map_region[tt[0]] = idx
#     region_back[idx] = tt[1]

left_region = [item for item in region_map.keys()]
house_refine = []
#and np.isnan(float(ie[1])) == False
for ie in house:
	if ie[0] in left_region and np.isnan(float(ie[1])) == False and float(ie[-1])!=0.0 and float(ie[1])!=0.0:
		house_refine.append([region_map[ie[0]],ie[1],ie[2]])


house_sum = []
for item in house_refine:
    tmp = []
    tmp.append(item[0])
    tmp.append(item[1])
    tmp.append(item[2])
    tmp.append(float(item[2]/item[1]))
    house_sum.append(tmp)

    # print("item:", item)



house_array = np.array(house_sum)
house_unit = house_array[:, 1]
price_unit = house_array[:, 3]

unit_max, unit_min  = max(house_unit), min(house_unit)
price_max, price_min  =  max(price_unit), min(price_unit)
print(price_max,price_min)

# re = pd.cut(house_unit, bins=[unit_min,1000,1500, 2000, 2500, unit_max])
# print("re:",re.tolist())
# house_uni_class = pd.cut(house_unit, [unit_min-1,1500, 2000, 2500,3000, 3500,4000,unit_max], labels=False).tolist() # 7 classes 
# price_class = pd.cut(price_unit, [0, 2500,5000,10000, 15000,20000, 30000, 40000,50000, 60000, 70000, 80000, price_max], labels=False).tolist()  # 12 classes
# ,7000,7500,8000,8500,9000,9500,10000, 10500, 11000, 11500,12000],labels=False).values.tolist()

# print(price_class)
house_feature = []
for item,unit,price in zip(house_sum,house_unit,price_unit):
    tmp = []
    tmp.append(item[0])
    tmp.append(int(unit))
    tmp.append(int(price))
    house_feature.append(tmp)

train_house = house_feature[:700]
test_house = house_feature[700:]
# print(train_house)
# print(test_house)
# print(len(test_house))
# print(len(house_feature))
# print 
# println()
file=open(r"../data/train_house.pickle","wb")
pickle.dump(train_house,file) #storing_list
file.close()
file=open(r"../data/test_house.pickle","wb")
pickle.dump(test_house,file) #storing_list
file.close()

# re = pd.cut(house.sqft, bins=[0,500,1000,1500,2000, 2500, 3000,3500, 4000,4500, 5000, 5500, 6000,6500
# ,7000,7500,8000,8500,9000,9500,10000, 10500, 11000, 11500,12000],labels=False).values.tolist()
# # hs = pd.read_csv("../data/house_source_extra.csv",sep = ",").values.tolist()
# house['sq'] = re
# hou = house.dropna(axis=0,how='any') #drop all rows that have any NaN values
# # print(len(hos))
# classi = hou['sq'].values.tolist()
# hos = hou.values.tolist()
# classier = list(set(classi))


# print("house_refine:",house_sum)
# print("before:", len(house))
# print("after:", len(house_sum))

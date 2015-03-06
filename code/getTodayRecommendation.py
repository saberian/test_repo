import numpy as np
import yahoo_finance as yf
import matplotlib.pyplot as plt
import os
import pickle
import stockRecommendationLib as srl
from sklearn.ensemble import GradientBoostingClassifier
import datetime


#today = '2015-03-04'
today = str(datetime.date.today())
max_price = 10
min_price = 1
min_vol = 1  # in millions

MILL = 1000000


raw_data_loc = "../data/historicData";
f_out = open("../recomms/" + today + ".txt", "w")
f_name = "today_sample_" + today
today_data, data_sym_list = pickle.load(
    open("../data/processedData/"+ f_name + ".pkl", "rb"))

clf_name = "gbdt_month_full_" + today
clf = pickle.load(open("../models/" + clf_name +".pkl", "rb"))
goodness_score = clf.predict_proba(today_data)[:,1]

ind = np.argsort(-goodness_score)
#print goodness_score[ind]
f_out.write("=======Best==============="+"\n")
cnt = 0
i = 0
while (cnt < 10):
    t = ind[i]
    sym = data_sym_list[t]
    full_name =  raw_data_loc + '/' + sym +'.pkl'
    dl = pickle.load(open(full_name, "rb"))
    current_price = float(dl[0]['Close'])
    current_volume = float(dl[0]['Volume'])/MILL
    #print "cheking: " + sym + " price is " + str(current_price)
    if current_price < 10 and current_price > 1 and current_volume > 1:
        f_out.write("stock: " + sym + (" score: %.3f" % goodness_score[t]) + \
              " cu price: " + str(current_price) +\
              (" current_volume: %.3f" % current_volume)+"\n")
        cnt = cnt + 1
    i = i + 1

f_out.write("=======Worst==============="+"\n")
nn = len(ind)-1
cnt = 0
i = 0
while (cnt < 10):
    t = ind[nn-i]
    sym = data_sym_list[t]
    full_name =  raw_data_loc + '/' + sym +'.pkl'
    dl = pickle.load(open(full_name, "rb"))
    current_price = float(dl[0]['Close'])
    current_volume = float(dl[0]['Volume'])/MILL
    #print "cheking: " + sym + " price is " + str(current_price)
    if current_price < 10 and current_price > 1 and current_volume > 1:
        f_out.write("stock: " + sym + (" score: %.3f" % goodness_score[t]) + \
              " cu price: " + str(current_price) +\
              (" current_volume: %.3f" % current_volume)+"\n")
        cnt = cnt + 1
    i = i + 1

f_out.close()

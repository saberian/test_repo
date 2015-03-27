import numpy as np
import yahoo_finance as yf
import matplotlib.pyplot as plt
import os
import pickle
import stockRecommendationLib as srl
from sklearn.ensemble import GradientBoostingClassifier
import datetime
import sys
from constants import *


max_price = 10
min_price = 1
min_vol = 1  # in millions
n_recomms = 20

data_date = sys.argv[1]
clf_name =  sys.argv[2]



print "++++++++++++++++++++++++++++++++++++++++++"
print "get recommendations for " + data_date


res_loc = RECOMMS_DIR + "/" +data_date;
if not os.path.isdir(res_loc):
    os.mkdir(res_loc)

f_out = open(res_loc + "/recomms.txt", "w")

f_name = "day_sample_" + data_date
file_loc = PROCESSED_DATA_DIR  + "/"+ f_name + ".pkl"
if not os.path.isfile(file_loc):
    os.system('python getDaySamples.py ' + data_date)

today_data, data_sym_list = pickle.load(open(file_loc, "rb"))

clf = pickle.load(open(MODELS_DIR + "/" + clf_name +"/model.pkl", "rb"))

#goodness_score = clf.predict_proba(today_data)[:,1]
goodness_score = clf.decision_function(today_data)


ind = np.argsort(-goodness_score)
#print goodness_score[ind]
f_out.write("=======Best==============="+"\n")
cnt = 0
i = 0
while (cnt < n_recomms) and (i < len(ind)):
    t = ind[i]
    sym = data_sym_list[t]
    full_name =  HISTORIC_DATA_LOC + '/' + sym +'.pkl'
    dl = pickle.load(open(full_name, "rb"))
    current_price = float(dl[0]['Close'])
    current_volume = float(dl[0]['Volume'])/ MILL
    if current_price < 10 and current_price > 1 and current_volume > 1:
        srl.generateReports(sym, res_loc, goodness_score[t])
        log_str = "stock: " + sym + (" score: %.3f" % goodness_score[t]) + \
             " cu price: " + str(current_price) +\
              (" current_volume: %.3f" % current_volume) #+ \
              #(" target price: %.2f" % srl.getYahoo1Yst(sym))
        print log_str
        f_out.write(log_str+"\n")
        cnt = cnt + 1
    i = i + 1

log_str = "recomms based on " + str(today_data.shape[0]) + " stocks for " + data_date + " is created"
print log_str
srl.writeLogSummary(log_str)


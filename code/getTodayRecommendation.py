import numpy as np
import yahoo_finance as yf
import matplotlib.pyplot as plt
import os
import pickle
import stockRecommendationLib as srl
from sklearn.ensemble import GradientBoostingClassifier
import datetime
import sys


MILL = 1000000


data_date = sys.argv[1]
classifier_date = sys.argv[2]

#today = str(datetime.date.today())
max_price = 10
min_price = 1
min_vol = 1  # in millions
n_recomms = 10

print "++++++++++++++++++++++++++++++++++++++++++"
print "get recommendations for " + data_date

raw_data_loc = "../data/historicData";
res_loc = "../recomms/"+data_date;
if not os.path.isdir(res_loc):
    os.mkdir(res_loc)

f_out = open(res_loc + "/recomms.txt", "w")

f_name = "today_sample_" + data_date
today_data, data_sym_list = pickle.load(open("../data/processedData/"+ f_name + ".pkl", "rb"))

clf_name = "gbdt_month_full_" + classifier_date
clf = pickle.load(open("../models/" + clf_name +"/model.pkl", "rb"))

goodness_score = clf.predict_proba(today_data)[:,1]

ind = np.argsort(-goodness_score)
#print goodness_score[ind]
f_out.write("=======Best==============="+"\n")
cnt = 0
i = 0
while (cnt < n_recomms) and (i < len(ind)):
    t = ind[i]
    sym = data_sym_list[t]
    full_name =  raw_data_loc + '/' + sym +'.pkl'
    dl = pickle.load(open(full_name, "rb"))
    current_price = float(dl[0]['Close'])
    current_volume = float(dl[0]['Volume'])/MILL
    if current_price < 10 and current_price > 1 and current_volume > 1:
        srl.generateReports(sym, res_loc, goodness_score[t])
        f_out.write("stock: " + sym + (" score: %.3f" % goodness_score[t]) + \
             " cu price: " + str(current_price) +\
              (" current_volume: %.3f" % current_volume)+ (" target price: %f" % srl.getYahoo1Yst(sym))+"\n")
        cnt = cnt + 1
    i = i + 1

log_str = "recomms based on " + str(today_data.shape[0]) + " stocks for " + data_date + " is created"
print log_str
srl.writeLogSummary(log_str)


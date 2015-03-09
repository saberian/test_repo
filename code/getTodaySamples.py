import numpy as np
import yahoo_finance as yf
import matplotlib.pyplot as plt
import os
import pickle
import stockRecommendationLib as srl
from sklearn.ensemble import GradientBoostingClassifier
import datetime
import sys

print "++++++++++++++++++++++++++++++++++++++++++"
print "getting data samples for today"
raw_data_loc = "../data/historicData";
today = sys.argv[1]

stock_list = open("../data/stock_list.txt").read().splitlines() 
#stock_list = stock_list[0::400]

data_sym_list =[];
cnt = 0;
train_data = np.array([])
train_label = np.array([])
for sym in stock_list:
    cnt = cnt + 1
    sys.stdout.write('.')
    sys.stdout.flush()
    #print sym + " : " + str(cnt) + " out of " + str(len(stock_list))
    full_name =  raw_data_loc + '/' + sym + ".pkl"
    dl = pickle.load(open(full_name, "rb"))
    if len(dl) > 0:
        if 'Date' in dl[0]:
            last_day = dl[0]['Date']
            if last_day == today:
                sym_data = srl.getStockFeatureForSymbol(dl, False, {})
                if len(sym_data) == 0:
                    #print "len(sym_data) = 0 "
                    continue
                td = sym_data[last_day]
                if td.shape[0] == 0:
                    #print "td.shape[0] = 0"
                    continue
                if sum(np.isnan(td)) > 0:
                    #print "IS NAN"
                    break
                if len(train_data) == 0:
                    train_data = td#np.zeros((1, len(td)))
                    #train_data[0,:] = td
                else:
                    train_data = np.vstack((train_data, td))
                data_sym_list.append(sym)

sys.stdout.write('\n')
f_name = "today_sample_" + today
pickle.dump([train_data, data_sym_list], \
    open("../data/processedData/"+ f_name + ".pkl", "wb"))

log_str = f_name + " is created with " + str(train_data.shape[0]) + " samples"
print log_str
srl.writeLogSummary(log_str)



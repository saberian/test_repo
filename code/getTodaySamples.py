import numpy as np
import yahoo_finance as yf
import matplotlib.pyplot as plt
import os
import pickle
import stockRecommendationLib as srl
from sklearn.ensemble import GradientBoostingClassifier
import datetime



raw_data_loc = "../data/historicData";
stock_list = os.listdir(raw_data_loc)
today = str(datetime.date.today())


data_sym_list =[];
cnt = 0;
train_data = np.array([])
train_label = np.array([])
for f_name in stock_list:
    cnt = cnt + 1
    if '.pkl' not in f_name:
        continue;
    sym = f_name[:-4]
    print sym + " : " + str(cnt) + " out of " + str(len(stock_list))
    full_name =  raw_data_loc + '/' + f_name
    dl = pickle.load(open(full_name, "rb"))
    if len(dl) > 0:
        if 'Date' in dl[0]:
            last_day = dl[0]['Date']
            if last_day == today:
                sym_data = srl.getStockFeatureForSymbol(dl, False, {})
                if len(sym_data) == 0:
                    continue
                td = sym_data[last_day]
                if td.shape[0] == 0:
                    continue
                if sum(np.isnan(td)) > 0:
                    print "IS NAN"
                    break
                if len(train_data) == 0:
                    train_data = td#np.zeros((1, len(td)))
                    #train_data[0,:] = td
                else:
                    train_data = np.vstack((train_data, td))
                print train_data.shape
                data_sym_list.append(sym)
    print " ================"

f_name = "today_sample_" + today
pickle.dump([train_data, data_sym_list], \
    open("../data/processedData/"+ f_name + ".pkl", "wb"))
srl.writeLogSummary(f_name + " is created")



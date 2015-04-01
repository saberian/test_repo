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



def getRecommForDay(day_str, clfr):
    print "processing  " + day_str
    f_name = "day_sample_" + day_str
    file_loc = PROCESSED_DATA_DIR  + "/"+ f_name + ".pkl"
    if not os.path.isfile(file_loc):
        print "get data for " + day_str
        os.system('python getDaySamples.py ' + day_str)
    day_data, day_sym_list = pickle.load(open(file_loc, "rb"))
    goodness_score = clf.decision_function(day_data)
    ind = np.argsort(-goodness_score)
    cnt = 0
    i = 0
    sym_list = []
    while (cnt < n_recomms) and (i < len(ind)):
        t = ind[i]
        sym = day_sym_list[t]
        #print sym
        full_name =  HISTORIC_DATA_LOC + '/' + sym +'.pkl'
        dl = pickle.load(open(full_name, "rb"))
        current_price = float(dl[0]['Close'])
        current_volume = float(dl[0]['Volume'])/ MILL
        if current_price < max_price and current_price > min_price and current_volume > min_vol:
            sym_list.append(sym)
            sys.stdout.write('.')
            sys.stdout.flush()
            cnt = cnt + 1
        i = i + 1
    sys.stdout.write('\n')
    return sym_list



max_price = 10
min_price = 1
min_vol = 1  # in millions
n_recomms = 30
n_days_thr = 0.6
n_days = 6

target_day = sys.argv[1]
clf_name =  sys.argv[2]

clf = pickle.load(open(MODELS_DIR + "/" + clf_name +"/model.pkl", "rb"))


print "++++++++++++++++++++++++++++++++++++++++++"
print "get recommendations for " + target_day

working_day_list = srl.getWorkingDayList()
day_ind = working_day_list.index(target_day)

res_hist = {}
for j in xrange(0, n_days):
    temp_day = working_day_list[day_ind-j]
    temp_list = getRecommForDay(temp_day, clf)
    for sym in temp_list:
        if sym in res_hist:
            res_hist[sym] +=1
        else:
            res_hist[sym] = 1

sorted_res = sorted(res_hist.items(), key=lambda x:-x[1])

res_loc = RECOMMS_DIR + "/" + target_day;
if not os.path.isdir(res_loc):
    os.mkdir(res_loc)
#f_out = open(res_loc + "/recomms.txt", "w")

for sym, h in sorted_res:
    if h > n_days*n_days_thr:
        print sym
        srl.generateReports(sym, res_loc, h)

log_str = "recomms for " + target_day + " is created"
print log_str
srl.writeLogSummary(log_str)

'''

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
    if current_price < 1000 and current_price > 1 and current_volume > 1:
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
srl.writeLogSummary(log_str)'''


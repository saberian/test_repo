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



def getRecommForDayCaffe(day_str):
    f_name = "day_sample_" + day_str
    file_loc = PROCESSED_DATA_DIR  + "/"+ f_name + ".pkl"
    if not os.path.isfile(file_loc):
        print "get data for " + day_str
        os.system('python getDaySamples.py ' + day_str)
    day_data, day_sym_list = pickle.load(open(file_loc, "rb"))
    caffe_res = srl.test_caffe(deploy_file, output_node, cf_model_dir, model_file_name, day_data, 1)

    print caffe_res.shape
    print day_data.shape
    print day_sym_list

    goodness_score = caffe_res[:,1]
    ind = np.argsort(-goodness_score)
    cnt = 0
    i = 0
    sym_list = []
    while (cnt < n_recomms) and (i < len(ind)):
        t = ind[i]
        sym = day_sym_list[t]
        print sym
        full_name =  HISTORIC_DATA_LOC + '/' + sym +'.pkl'
        dl = pickle.load(open(full_name, "rb"))
        current_price = float(dl[0]['Close'])
        current_volume = float(dl[0]['Volume'])/ MILL

        print [current_price, current_volume]

        if current_price < max_price and current_price > min_price and current_volume > min_vol:
            sym_list.append(sym)
            sys.stdout.write('.')
            sys.stdout.flush()
            cnt = cnt + 1
        i = i + 1
    sys.stdout.write('\n')
    return sym_list


prd_name = "caffe_clf_v4_week"
prd_type = "clf"
protobuf_loc = 'caffe_clf_protobuf'
prt_name = protobuf_loc + "/train_val4.prototxt"
deploy_file = protobuf_loc + "/deploy4.prototxt"
output_node = "fc1"
batch_size = 100
n_max_itr = 10000
model_file_name = prd_name + "_iter_"+  str(n_max_itr) + ".caffemodel"
cf_model_dir = MODELS_DIR + "/" + prd_name

max_price = 1000
min_price = 1
min_vol = 0  # in millions
n_recomms = 30
n_days_thr = 0 #0.6
n_days = 1

target_day = "2015-04-01"#sys.argv[1]
print "++++++++++++++++++++++++++++++++++++++++++"
print "get recommendations for " + target_day



working_day_list = srl.getWorkingDayList()
day_ind = working_day_list.index(target_day)

res_hist = {}
for j in xrange(0, n_days):
    temp_day = working_day_list[day_ind-j]
    print temp_day
    print "ess"
    temp_list = getRecommForDayCaffe(temp_day)
    for sym in temp_list:
        if sym in res_hist:
            res_hist[sym] +=1
        else:
            res_hist[sym] = 1

print res_hist

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

import numpy as np
import yahoo_finance as yf
import matplotlib.pyplot as plt
import os
import pickle
import stockRecommendationLib as srl
import datetime
import sys
from datetime import date, timedelta
from constants import *


print "++++++++++++++++++++++++++++++++++++++++++"
print "collecting training example"

current_db_day = sys.argv[1]
n_offset_days = int(sys.argv[2])
ratio = 0.8

stock_list = srl.getStockList()
date_list = srl.getWorkingDayList()
n_labeled_days = len(date_list) - n_offset_days;

sp_ind = int(ratio*n_labeled_days)
train_start_date = srl.convertDay2Date(date_list[0])
train_end_date = srl.convertDay2Date(date_list[sp_ind])

test_start_date = srl.convertDay2Date(date_list[sp_ind+1])
test_end_date = srl.convertDay2Date(date_list[n_labeled_days])

print "started to collect training examples"
train_data, train_label = srl.getDatasetFromList(stock_list,\
                                             train_start_date, train_end_date, current_db_day)

print "started to collect test examples"
test_data, test_label = srl.getDatasetFromList(stock_list, \
                                           test_start_date, test_end_date, current_db_day)

print "dumping the data"
pickle.dump([train_data, train_label, test_data, test_label], \
             open(PROCESSED_DATA_DIR + "/dataset_eval_" + current_db_day + ".pkl", "wb"))

n_total = train_data.shape[0] + test_data.shape[0] 

log_str = "data_eval_" + current_db_day  + " with %d training example and %d test examples is created" % (train_data.shape[0], test_data.shape[0])
print log_str
srl.writeLogSummary(log_str)


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

stock_list = srl.getStockList()
date_list = srl.getWorkingDayList()

start_date = srl.convertDay2Date(date_list[0])
end_date = srl.convertDay2Date(date_list[-1])

print "started to collect all training examples"
train_data, train_label = srl.getDatasetFromList(stock_list,\
                                                 start_date, end_date, current_db_day)

pickle.dump([train_data, train_label], \
             open(PROCESSED_DATA_DIR + "/dataset_full_" + current_db_day + ".pkl", "wb"))

n_total = train_data.shape[0]

log_str = "data_full_" + current_db_day  + " with %d training example is created" % (train_data.shape[0])
print log_str
srl.writeLogSummary(log_str)


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
import h5py

print "++++++++++++++++++++++++++++++++++++++++++"
print "collecting training example"

def getHdf5FromList(fn_name, col_no, pr_type, stock_list, start_date, end_date, current_db_day):
    f = open(fn_name + '.txt', 'w')
    print "getting data from " + str(start_date) + " to " + str(end_date)
    train_data = np.array([])
    train_label = np.array([])
    cnt = 0
    for sym in stock_list:
        #print sym
        sys.stdout.write('.')
        sys.stdout.flush()
        f_name = DATABASE_DIR + "/" + sym + "_" + current_db_day + ".pkl"
        if not os.path.isfile(f_name):
            continue
        datapoints = pickle.load(open(f_name, "r"))

        X, tl, day_lab = srl.getMatrixFromDatabase(datapoints, str(start_date), str(end_date))
        if len(X.shape) == 1:
            continue


        ns = X.shape[0]
        nf = X.shape[1]
        X = np.reshape(X, (ns, 1, nf, 1))


        y = tl[:, col_no] - 1
        if pr_type == "clf":
            y = y + 1
            y = (y>PROFIT_MARGIN)+0.0
            y = y.astype(int)
        if pr_type == "reg":
            y = y * (y < srl.MAX_RESPONSE) + srl.MAX_RESPONSE * (y > srl.MAX_RESPONSE)

        comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
        filename = fn_name + "_" + sym + ".h5"
        with h5py.File(filename, 'w') as g:
            g.create_dataset('data', data=X, **comp_kwargs)
            g.create_dataset('label', data=y.astype(np.float32), **comp_kwargs)
            g.create_dataset('day_lab', data=day_lab, **comp_kwargs)
            g.create_dataset('symbol', data=[sym for e in day_lab])

        f.write(filename + '\n')
        #print td.shape

        cnt = cnt + 1
    sys.stdout.write('\n')



current_db_day = sys.argv[1]
n_offset_days = int(sys.argv[2])
preiod_type = "week"
preiod_index = srl.LABEL_PREIOD.index(preiod_type)
prd_type = "reg" # "clf" #
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
getHdf5FromList(srl.HDF5_LOC + "/train_" +  prd_type, preiod_index, prd_type, stock_list, train_start_date, train_end_date, current_db_day)

getHdf5FromList(srl.HDF5_LOC + "/test_" + prd_type, preiod_index, prd_type, stock_list, test_start_date, test_end_date, current_db_day)


log_str = "hdf5 file for " + current_db_day  + " are created"
print log_str
srl.writeLogSummary(log_str)


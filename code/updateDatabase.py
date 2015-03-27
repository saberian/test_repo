import numpy as np
import yahoo_finance as yf
import matplotlib.pyplot as plt
import os
import pickle
import stockRecommendationLib as srl
from pprint import pprint
import datetime
import sys
from constants import *

last_day = sys.argv[1]#
today = sys.argv[2]
start_ind = int(sys.argv[3])
end_ind = int(sys.argv[4])

print "++++++++++++++++++++++++++++++++++++++++++"
print "updatting dataset from " + last_day + " to " + today

stock_list = srl.getStockList()

cnt = 0
n_added_entry = 0;
for i in xrange(start_ind, end_ind):
    sym = stock_list[i]
    #print sym
    sys.stdout.write('.')
    sys.stdout.flush()
    cnt = cnt + 1
    current_sym_data = {}
    if last_day != "non":
        last_day_file = DATABASE_DIR + '/' + sym + "_" + last_day + ".pkl"
        if os.path.isfile(last_day_file):
            current_sym_data = pickle.load(open(last_day_file, "rb"))
    len_old = len(current_sym_data)

    full_name =  HISTORIC_DATA_LOC + '/' + sym + ".pkl"
    dl = pickle.load(open(full_name, "rb"))

    new_sym_data = srl.getStockFeatureForSymbol(dl, True, current_sym_data, 'all')

    n_added_entry += (len(new_sym_data) - len_old)

    if len(new_sym_data ) > 0:
        if last_day != "non":
            if os.path.isfile(last_day_file):
                os.remove(last_day_file)
        today_file = DATABASE_DIR + '/' + sym + "_" + today + ".pkl"
        pickle.dump(new_sym_data, open(today_file, "w"))


    if cnt % 500 == 0:
        sys.stdout.write('\n')

sys.stdout.write('\n')
log_str = "database updated from " + last_day + " to " + today + \
          " and %d samples are added" % n_added_entry
print log_str
srl.writeLogSummary(log_str)

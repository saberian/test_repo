import numpy as np
import yahoo_finance as yf
import matplotlib.pyplot as plt
import os
import pickle
import stockRecommendationLib as srl
from pprint import pprint
import datetime
import sys

last_day = sys.argv[1]#
today = sys.argv[2]
print "++++++++++++++++++++++++++++++++++++++++++"
print "updatting dataset from " + last_day + " to " + today

raw_data_loc = "../data/historicData"
stock_list = open("../data/stock_list.txt").read().splitlines() 
#stock_list = stock_list[0::400]

database = {}
if last_day != "non":
    database = pickle.load(open("../data/processedData/database_" + last_day + ".pkl", "rb"))

cnt = 0
n_added_entry = 0;
for sym in stock_list:
    #print sym
    sys.stdout.write('.')
    sys.stdout.flush()
    cnt = cnt + 1

    #sym = f_name[:-4]
    #print sym + " : " + str(cnt) + " out of " + str(len(stock_list))

    full_name =  raw_data_loc + '/' + sym + ".pkl"
    dl = pickle.load(open(full_name, "rb"))

    current_sym_data = {}
    if sym in database:
        #print "loaded from database"
        current_sym_data = database[sym]

    len_old = len(current_sym_data)
    new_sym_data = srl.getStockFeatureForSymbol(dl, True, current_sym_data)


    n_added_entry += (len(new_sym_data) - len_old)

    #print [len_old, len(new_sym_data)]

    if len(new_sym_data ) > 0:
        database[sym] = new_sym_data
    if cnt % 500 == 0:
        sys.stdout.write('\n')

sys.stdout.write('\n')
pickle.dump(database, open("../data/processedData/database_" + today + ".pkl", "wb"))

log_str = "database updated from " + last_day + " to " + today + \
          " and %d samples are added" % n_added_entry

print log_str
srl.writeLogSummary(log_str)

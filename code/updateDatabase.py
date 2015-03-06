import numpy as np
import yahoo_finance as yf
import matplotlib.pyplot as plt
import os
import pickle
import stockRecommendationLib as srl
from pprint import pprint
import datetime
import sys


raw_data_loc = "../data/historicData"
stock_list = os.listdir(raw_data_loc)

last_day = sys.argv[1]#
today = str(datetime.date.today())
database = pickle.load(open("../data/processedData/database_" + last_day + ".pkl", "rb"))

cnt = 0
for f_name in stock_list:
    cnt = cnt + 1
    if '.pkl' not in f_name:
        continue;
    sym = f_name[:-4]
    print sym + " : " + str(cnt) + " out of " + str(len(stock_list))

    full_name =  raw_data_loc + '/' + f_name
    dl = pickle.load(open(full_name, "rb"))

    current_sym_data = {}
    if sym in database:
        current_sym_data = database[sym]

    new_sym_data = srl.getStockFeatureForSymbol(dl, True, current_sym_data)

    if len(new_sym_data ) > 0:
        database[sym] = new_sym_data
    #if cnt > 20:
    #    break

pickle.dump(database, open("../data/processedData/database_" + today + ".pkl", "wb"))
srl.writeLogSummary("database_" + today + " is created")

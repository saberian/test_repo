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


print "++++++++++++++++++++++++++++++++++++++++++"
target_day = "2015-03-31"#sys.argv[1]

stock_list = srl.getStockList()

for sym in stock_list:
    #sys.stdout.write('.')
    #sys.stdout.flush()
    full_name =  HISTORIC_DATA_LOC + '/' + sym + ".pkl"
    dl = pickle.load(open(full_name, "rb"))
    if len(dl) == 0:
        continue
    if 'Close' not in dl[0]:
        print sym
        os.remove(full_name)
        print "============"
    
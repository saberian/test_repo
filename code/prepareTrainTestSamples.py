import numpy as np
import yahoo_finance as yf
import matplotlib.pyplot as plt
import os
import pickle
import stockRecommendationLib as srl
import datetime
import sys


MONTH_DAYS = 20

print "++++++++++++++++++++++++++++++++++++++++++"
print "collecting training example"


print "loading database"
today = sys.argv[1]
database = pickle.load(open("../data/processedData/database_" + today + ".pkl", "rb"))

print "started to collect examples"

train_data = np.array([])
train_label = np.array([])

test_data = np.array([])
test_label = np.array([])

extra_train_data = np.array([])
extra_train_label = np.array([])

cnt = 0
for sym in database:
    #print sym
    sys.stdout.write('.')
    sys.stdout.flush()

    td, tl = srl.getMatrixFromDatabse(database[sym])
    if len(td.shape) == 1:
        continue
    if td.shape[0] < (MONTH_DAYS+1):
        continue
    if cnt == 0:
        extra_train_data = td[-MONTH_DAYS:,:]
        extra_train_label = tl[-MONTH_DAYS:,:]
        train_data = td[0:-MONTH_DAYS,:]
        train_label = tl[0:-MONTH_DAYS,:]
        test_data = td[-1,:]
        test_label = tl[-1,:]
    else:
        extra_train_data = np.vstack((extra_train_data, td[-MONTH_DAYS:,:]))
        extra_train_label = np.vstack((extra_train_label, tl[-MONTH_DAYS:,:]))
        train_data = np.vstack((train_data, td[0:-MONTH_DAYS,:]))
        train_label = np.vstack((train_label, tl[0:-MONTH_DAYS,:]))
        test_data = np.vstack((test_data, td[-1,:]))
        test_label = np.vstack((test_label, tl[-1,:]))
    #print str(cnt) + " out of " + str(len(database))
    cnt = cnt + 1
    #if cnt > 10:
        #break

sys.stdout.write('\n')
pickle.dump([train_data, train_label, \
            test_data, test_label, \
            extra_train_data, extra_train_label\
            ], open("../data/processedData/train_data_" + today + ".pkl", "wb"))

n_total = train_data.shape[0] + test_data.shape[0] + extra_train_data.shape[0]

log_str = "train_data_" + today + " with total sample of " + str(n_total) + " is created"
print log_str
srl.writeLogSummary(log_str)


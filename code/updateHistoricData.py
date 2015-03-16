import numpy as np
import yahoo_finance as yf
import os.path
import pickle
import time
from datetime import date, timedelta
import stockRecommendationLib as srl
import datetime
import sys
from constants import *

print "++++++++++++++++++++++++++++++++++++++++++"
print "updating historic data files"

target_day = sys.argv[1]
start_ind = int(sys.argv[2])
end_ind = int(sys.argv[3])

target_date = srl.convertDay2Date(target_day)
sym_list = srl.getStockList()

cnt = 0
n_new_files = 0
n_updated_file = 0
n_up_to_date_file = 0
for i in xrange(start_ind, end_ind):
    sym = sym_list[i]
    #print sym
    sys.stdout.write('.')
    sys.stdout.flush()
    #print sym + " " + str(cnt) + " out of " + str(len(sym_list))
    cnt = cnt + 1
    fname = HISTORIC_DATA_LOC + "/" + sym + '.pkl'
    try:
        r = yf.Share(sym)
    except Exception, e:
        continue
    if not os.path.isfile(fname):
        dl = r.get_historical(START_DATE, str(target_date))
        pickle.dump(dl, open(fname, "wb"))
        n_new_files += 1
    else:
        dl = pickle.load(open(fname, "rb"))
        if len(dl) == 0:
            continue
        if 'Date' not in dl[0]:
            continue
        last_date = srl.convertDay2Date(dl[0]['Date'])
        #print last_date
        if target_date > last_date:
            #print "updating"
            new_dl = r.get_historical(str(last_date), str(target_date))
            new_dl = new_dl[:-1]
            new_dl = new_dl + dl
            pickle.dump(new_dl, open(fname, "wb"))
            n_updated_file +=1
        else:
            #print "file is uptodate"
            n_up_to_date_file +=1
    if cnt % 500 == 0:
        sys.stdout.write('\n')

sys.stdout.write('\n')
log_str = "historic data updated with %d new files, %d updated files and %d file were up to date " %(n_new_files, n_updated_file, n_up_to_date_file)
print log_str
srl.writeLogSummary(log_str)

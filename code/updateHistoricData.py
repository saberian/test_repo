import numpy as np
import yahoo_finance as yf
import os.path
import pickle
import time
from datetime import date, timedelta
import stockRecommendationLib as srl
import datetime
import sys


print "++++++++++++++++++++++++++++++++++++++++++"
print "updating historic data files"


t=time.strptime(sys.argv[1],'%Y-%m-%d')
today =date(t.tm_year,t.tm_mon,t.tm_mday)

sym_list = open("../data/stock_list.txt").read().splitlines() 
#sym_list = sym_list[0::400]

###### get hitoric data ##############
START_DATE = '2010-01-01'
data_path = "../data/historicData"

cnt = 0
n_new_files = 0
n_updated_file = 0
n_up_to_date_file = 0
for sym in sym_list:
    sys.stdout.write('.')
    sys.stdout.flush()
    #print sym + " " + str(cnt) + " out of " + str(len(sym_list))
    cnt = cnt + 1
    fname = data_path + "/" + sym + '.pkl'
    try:
        r = yf.Share(sym)
    except Exception, e:
        continue
    if not os.path.isfile(fname):
        dl = r.get_historical(START_DATE, str(today))
        pickle.dump(dl, open(fname, "wb"))
        n_new_files += 1
    else:
        dl = pickle.load(open(fname, "rb"))
        if len(dl) == 0:
            continue
        if 'Date' not in dl[0]:
            continue
        t=time.strptime(dl[0]['Date'],'%Y-%m-%d')
        last_date =date(t.tm_year,t.tm_mon,t.tm_mday) #+timedelta(1)
        #print last_date
        if today > last_date:
            #print "updating"
            new_dl = r.get_historical(str(last_date), str(today))
            new_dl = new_dl[:-1]
            #print new_dl
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

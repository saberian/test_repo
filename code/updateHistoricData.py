import numpy as np
import yahoo_finance as yf
import os.path
import pickle
import time
from datetime import date, timedelta
import stockRecommendationLib as srl
import datetime



### get list of stock items
file_path = '../data/NASDAQ_list'
names = ['NASDAQ_20150121.csv', 'NASDAQ_20150122.csv', 'NASDAQ_20150123.csv', 
         'NASDAQ_20150126.csv', 'NASDAQ_20150127.csv', 'NASDAQ_20150128.csv', 
         'NASDAQ_20150129.csv', 'NASDAQ_20150202.csv', 'NASDAQ_20150203.csv']
n_files  = len(names)
sym_list = []
for i in xrange(0, n_files):
    f =  open(file_path + '/' + names[i], 'r')
    cnt = 0
    for line in f:
        if cnt == 0:
            cnt = 1
            continue
        ws = line.split(',')
        sym = ws[0]
        if sym not in sym_list:
            sym_list.append(sym)

###### get hitoric data ##############
START_DATE = '2013-01-01'
data_path = "../data/historicData"
today = date.today()
cnt = 0
for sym in sym_list:
    print sym + " " + str(cnt) + " out of " + str(len(sym_list))
    cnt = cnt + 1
    fname = data_path + "/" + sym + '.pkl'
    try:
        r = yf.Share(sym)
    except Exception, e:
        continue
    if not os.path.isfile(fname):
        dl = r.get_historical(START_DATE, str(today))
        pickle.dump(dl, open(fname, "wb"))
    else:
        dl = pickle.load(open(fname, "rb"))
        if len(dl) == 0:
            continue
        if 'Date' not in dl[0]:
            continue
        t=time.strptime(dl[0]['Date'],'%Y-%m-%d')
        next_date =date(t.tm_year,t.tm_mon,t.tm_mday)+timedelta(1)
        if today >= next_date:
            print "updating"
            new_dl = r.get_historical(str(next_date), str(today))
            new_dl = new_dl + dl
            pickle.dump(new_dl, open(fname, "wb"))

srl.writeLogSummary("historic data updated")

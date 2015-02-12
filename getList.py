import numpy as np
import yahoo_finance as yf
import matplotlib.pyplot as plt
import os.path
import pickle


def getLinearCoeff(t_list):
    my_list = [float(x) for x in t_list]
    y = np.array(my_list);
    x = np.arange(0,len(y))
    z = np.polyfit(x,y,1)
    return z[0]

file_path = 'NASDAQ'
names = ['NASDAQ_20150121.csv', 'NASDAQ_20150122.csv', 'NASDAQ_20150123.csv', 
         'NASDAQ_20150126.csv', 'NASDAQ_20150127.csv', 'NASDAQ_20150128.csv', 
         'NASDAQ_20150129.csv', 'NASDAQ_20150202.csv', 'NASDAQ_20150203.csv']
n_files  = len(names)

files = {}
close_price= {}
avg_price = {}
day_5_linear_slope = {}
month_1_linear_slope = {}
month_3_linear_slope = {}
month_6_linear_slope = {}
year_1_linear_slope = {}
price_trends = {}
n_days = 5.0

for i in xrange(0, n_files):
    files[i] =  open(file_path + '/' + names[i], 'r');

for i in xrange(0, n_files):
    cnt = 0;
    for ent in files[i]:
        if cnt == 0 :
            cnt = cnt + 1
            continue;
        ws = ent.split(',')
        if ws[0] not in close_price:
            close_price[ws[0]] = [];  
        close_price[ws[0]].append(float(ws[5]))
        cnt = cnt + 1 
    #print cnt

for sym in close_price:
    avg_price[sym] = sum(close_price[sym][0:int(n_days)]) / n_days
    day_5_linear_slope[sym] = getLinearCoeff(close_price[sym][0:int(n_days)]);

candidate_list = []
up_thr = 1.3
low_thr = 1.1
max_price = 20.0
for sym in close_price:
    ratio = close_price[sym][-1] / avg_price[sym]
    if ((len(close_price[sym]) > n_days) and
        (avg_price[sym] < max_price)  and
        (ratio >=low_thr) and
        (ratio <=up_thr)): 
        candidate_list.append(sym)

#candidate_list = candidate_list
print candidate_list

#candidate_list = ['CBLI']
plt.xticks(rotation=70)
for sym in candidate_list:
    r = yf.Share(sym)
    #print sec + ' price is '+r.get_price()
    fname = 'historicData/' + sym + '.pkl'
    if os.path.isfile(fname):
        dl = pickle.load(open(fname, "rb"))
    else:
        dl = r.get_historical('2013-01-01', '2015-01-29')
        pickle.dump(dl, open(fname, "wb"))
    y_date = []
    y_close_price = []
    for k in dl:
        y_close_price.append(k['Close'])
        y_date.append(k['Date'])
    y_close_price = y_close_price[::-1]
    y_date = y_date[::-1]
    l_y = len(y_close_price)
    day_5_linear_slope[sym] = getLinearCoeff(y_close_price[l_y-5:l_y]);
    month_1_linear_slope[sym] = getLinearCoeff(y_close_price[l_y-30:l_y]);
    month_3_linear_slope[sym] = getLinearCoeff(y_close_price[l_y-90:l_y]);
    month_6_linear_slope[sym] = getLinearCoeff(y_close_price[l_y-180:l_y]);
    year_1_linear_slope[sym] = getLinearCoeff(y_close_price[l_y-370:l_y]);

    pt = [day_5_linear_slope[sym], month_1_linear_slope[sym], month_3_linear_slope[sym], month_6_linear_slope[sym], year_1_linear_slope[sym]]
    print " %s : %1.3f %1.3f %1.3f %1.3f %1.3f" % (sym, pt[0], pt[1], pt[2], pt[3], pt[4]) 




    plt.plot(y_close_price, label=sym)

    #print y_close_price
    #print close_price[sec]
    #plt.xticks(range(len(y_date)), y_date, size='small')
    #print y_date


plt.grid()
plt.legend()
plt.show()


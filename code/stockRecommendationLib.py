import numpy as np
import yahoo_finance as yf
import matplotlib.pyplot as plt
import os.path
import pickle
import yahoo_finance as yf
from time import strftime
import datetime
import time
from datetime import date, timedelta
import os
from matplotlib import finance
from datetime import datetime
from matplotlib.dates import date2num
import sys
from constants import *

def plotCandleStick(ax, sym, date_list, open_price, close_price,\
                    high_price, low_price, n_days, cap):
    price_info = []
    len_list = len(date_list) - 1
    for k in xrange(0, n_days):
        i = len_list - k
        date = date2num(datetime.strptime(date_list[i], "%Y-%m-%d"))
        t = (date, open_price[i], close_price[i],\
                    high_price[i], low_price[i])
        price_info.append(t)

    finance.candlestick(ax, price_info, width=0.2, colorup='r', colordown='k', alpha=1.0)
    ax.grid()
    ax.set_xticklabels([])
    ax.set_ylabel(cap)

def getAllNumbers(share_list):
    date_list = []
    close_price = []
    open_price = []
    high_price = []
    low_price = []
    volume = []
    for share in share_list:
        try:
            date_list.append(share['Date'])
            close_price.append(float(share['Close']))
            open_price.append(float(share['Open']))
            high_price.append(float(share['High']))
            low_price.append(float(share['Low']))
            volume.append(float(share['Volume']))
        except Exception, e:
            print "error"

    date_list = date_list[::-1]
    close_price = np.array(close_price[::-1])
    open_price = np.array(open_price[::-1])
    high_price = np.array(high_price[::-1])
    low_price = np.array(low_price[::-1])
    volume = np.array(volume[::-1])
    return date_list, close_price, open_price,high_price, low_price, volume

def plotDetails(ax, input_y, input_x, n_days, cap):
    data = input_y[-n_days:]
    labs = input_x[-n_days:]
    x = np.arange(0, len(data))
    #plt.xticks(x, labs)
    #locs, labels = plt.xticks()
    #plt.setp(labels, rotation=45)
    ax.plot(data, label=cap)
    ax.set_xticklabels([])
    ax.grid()

def generateReports(sym, loc, score):
    if not os.path.isdir(loc):
        os.mkdir(loc)
    full_name =  HISTORIC_DATA_LOC + '/' + sym +'.pkl'
    dl = pickle.load(open(full_name, "rb"))
    current_price = float(dl[0]['Close'])
    current_volume = float(dl[0]['Volume'])/MILL
    #yahoo_1y_est = getYahoo1Yst(sym)

    date_list, close_price, open_price, high_price, low_price, volume = getAllNumbers(dl)

    plt_title = (" score: %.2f" % score) + \
                (" volume: %.2f" % current_volume) + \
                (" cu price: %.2f" % current_price) #+ \
                #(" target price: %.2f" % yahoo_1y_est)


    plt.figure(figsize=(25,20))
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 2, sharey='row', figsize=(15,10))
    fig.suptitle(plt_title, fontsize=14)

    plotCandleStick(ax1[0], sym, date_list, open_price, close_price,\
                     high_price, low_price, 2*WEEK_DAYS,  "last 2 week")
    plotCandleStick(ax2[0], sym, date_list, open_price, close_price, \
                     high_price, low_price, MONTH_DAYS, "last month")
    plotCandleStick(ax3[0], sym, date_list, open_price, close_price, \
                     high_price, low_price, 6*MONTH_DAYS, "last 6 month")
    plotCandleStick(ax4[0], sym, date_list, open_price, close_price, \
                     high_price, low_price, YEAR_DAYS, "last year")
    
    plotDetails(ax1[1], close_price, date_list, 2*WEEK_DAYS, "last week")
    plotDetails(ax2[1], close_price, date_list, MONTH_DAYS, "last month")
    plotDetails(ax3[1], close_price, date_list, 6*MONTH_DAYS, "last 6 month")
    plotDetails(ax4[1], close_price, date_list, YEAR_DAYS , "last year")

    plt.draw()
    plt.savefig(loc + ("/%.3f" % score) + "_" + sym + "_report.png")
    plt.close('all')

def getYahoo1Yst(sym):
    est = 0
    url = "http://finance.yahoo.com/echarts?s=" + sym
    file_name  = "temp_" + sym
    res = os.system("wget -O " + file_name + " " + url + "  >/dev/null 2>&1 ")
    f= open(file_name)
    phrase = "1Y Target Est"
    lines = f.read().splitlines()

    for i in xrange(0, len(lines)):
        if phrase in lines[i]: 
            #print lines[i]
            #print lines[i+1]
            ws = lines[i+1].split(">")[1]
            est = float(ws[:-3])
            #print est
            break
    os.system("rm " + file_name)
    return est

def mean(in_list):
    if len(in_list) == 0:
        return 0
    return sum(in_list)/len(in_list)

def getLinearCoeff(my_list, d, n):
    my_list = my_list[max(0, d-n):d]
    if len(my_list) == 0:
        return 0
    y = np.array(my_list);
    x = np.arange(0,len(y))
    z = np.polyfit(x,y,1)
    return z[0]

def getAverage(my_list, d, n):
    my_list = my_list[max(0, d-n):d]
    if len(my_list) == 0:
        return 0
    return mean(my_list)

def getMin(my_list, d, n):
    my_list = my_list[max(0, d-n):d]
    if len(my_list) == 0:
        return 0
    return min(my_list)

def getMax(my_list, d, n):
    my_list = my_list[max(0, d-n):d]
    if len(my_list) == 0:
        return 0
    return max(my_list)

def getTrends(price_list, d):
    last_week_trend = getLinearCoeff(price_list, d, WEEK_DAYS)
    last_month_trend = getLinearCoeff(price_list, d, MONTH_DAYS)
    last_3_month_trend = getLinearCoeff(price_list, d, 3*MONTH_DAYS)
    last_6_month_trend = getLinearCoeff(price_list, d, 6*MONTH_DAYS)
    last_year_trend = getLinearCoeff(price_list, d, YEAR_DAYS)
    # assemble the feature
    feature = np.array([last_week_trend, last_month_trend,
                        last_3_month_trend, last_6_month_trend,
                        last_year_trend]);
    return feature

def getMovingAverages(price_list, d):
    last_week_avg = getAverage(price_list, d, WEEK_DAYS)
    last_month_avg = getAverage(price_list, d, MONTH_DAYS)
    last_3_month_avg = getAverage(price_list, d, 3*MONTH_DAYS)
    last_6_month_avg = getAverage(price_list, d, 6*MONTH_DAYS)
    last_year_avg = getAverage(price_list, d, YEAR_DAYS)
    feature = np.array([last_week_avg, last_month_avg, 
                        last_3_month_avg, last_6_month_avg,
                        last_year_avg])
    return feature

def getMovingMins(price_list, d):
    last_week_stat = getMin(price_list, d, WEEK_DAYS)
    last_month_stat = getMin(price_list, d, MONTH_DAYS)
    last_3_month_stat = getMin(price_list, d, 3*MONTH_DAYS)
    last_6_month_stat = getMin(price_list, d, 6*MONTH_DAYS)
    last_year_stat = getMin(price_list, d, YEAR_DAYS)
    feature = np.array([last_week_stat, last_month_stat, 
                        last_3_month_stat, last_6_month_stat,
                        last_year_stat])
    return feature

def getMovingMaxs(price_list, d):
    current_price = price_list[d]
    last_week_stat = getMax(price_list, d, WEEK_DAYS)
    last_month_stat = getMax(price_list, d, MONTH_DAYS)
    last_3_month_stat = getMax(price_list, d, 3*MONTH_DAYS)
    last_6_month_stat = getMax(price_list, d, 6*MONTH_DAYS)
    last_year_stat = getMax(price_list, d, YEAR_DAYS)
    feature = np.array([last_week_stat, last_month_stat, 
                        last_3_month_stat, last_6_month_stat,
                        last_year_stat])
    return feature

def getAllStat(price_list, d):
    current_price = price_list[d]
    f1 = getMovingAverages(price_list, d)
    f2 = getTrends(price_list, d)
    f3 = getMovingMaxs(price_list, d)
    f4 = getMovingMins(price_list, d)
    return np.hstack((f1,f2,f3,f4))

def getRawStat(price_list, d):
    return price_list[(d-YEAR_DAYS):(d+1)]

def getLabel(price_list, d):
    current_price = price_list[d]
    temp_label = np.zeros((3))
    next_day_price_ratio = price_list[d+1] / current_price
    next_week_max_price_ratio = max(price_list[d+1:d+5]) / current_price
    next_month_max_price_ratio = max(price_list[d+1:d+20]) / current_price

    temp_label[0] = next_day_price_ratio
    temp_label[1] = next_week_max_price_ratio
    temp_label[2] = next_month_max_price_ratio

    return temp_label

def getStockFeatureForSymbol(share_list, ret_label, sym_data, target_day):
    if len(share_list) == 0:
        return sym_data
    if 'Close' not in share_list[0]:
        return sym_data
    n_days = len(share_list)
    if  n_days < (YEAR_DAYS+MONTH_DAYS+1):
        return sym_data
    ### get price history
    date_list = []
    close_price = []
    open_price = []
    high_price = []
    low_price = []
    volume = []
    for share in share_list:
        try:
            date_list.append(share['Date'])
            close_price.append(float(share['Close']))
            open_price.append(float(share['Open']))
            high_price.append(float(share['High']))
            low_price.append(float(share['Low']))
            volume.append(float(share['Volume']))
        except Exception, e:
            return sym_data 

    ### reverse the order of lists
    if np.min(close_price) < 0.01:
        return sym_data

    date_list = date_list[::-1]
    close_price = np.array(close_price[::-1])
    open_price = np.array(open_price[::-1])
    high_price = np.array(high_price[::-1])
    low_price = np.array(low_price[::-1])
    volume = np.array(volume[::-1])

    max_daily_change = (high_price - low_price) / close_price
    daily_change = (close_price - open_price) / close_price

    if target_day == "all": #ret_label:
        ### getting range of valid days
        x_min = min(YEAR_DAYS, n_days)
        x_max = max(0, n_days-MONTH_DAYS)
        days = xrange(x_min, x_max);
    else:
        days = []
        if target_day in date_list:
            days = [date_list.index(target_day)]# [n_days-1]

    ### get features per day
    for d in days:
        #print "day is: " + str(d)
        current_date = date_list[d]
        if current_date not in sym_data:
            current_price = close_price[d] + SMALL_NUM;
            f1 = getAllStat(close_price, d)
            f2 = getAllStat(daily_change, d)
            f3 = getAllStat(max_daily_change, d)
            f4 = getAllStat(volume, d)
            '''f1 = getRawStat(close_price, d)
            f2 = getRawStat(daily_change, d)
            f3 = getRawStat(max_daily_change, d)
            f4 = getRawStat(volume, d)'''
            f_all = np.hstack((f1,f2,f3, f4))
            res = f_all
            if ret_label:
                temp_lab = np.array([getLabel(close_price, d)])
                res = [f_all, temp_lab]
            sym_data[current_date] = res
    return sym_data


def getMatrixFromDatabase(data_points, start_date, end_date):
    data = np.array([])
    labels = np.array([])
    sorted_days = sorted(data_points.keys())
    start_ind = 0
    #print start_date
    if start_date in sorted_days:
        start_ind = sorted_days.index(start_date)
    end_ind = len(sorted_days)
    if end_date in sorted_days:
        end_ind = sorted_days.index(end_date)

    for i in xrange(start_ind, end_ind):
        day = sorted_days[i]
        f_all = data_points[day][0]
        lab = data_points[day][1]
        if data.shape[0] == 0:
            data = f_all 
        else:
            data = np.vstack((data, f_all ))
        if labels.shape[0] == 0:
            labels = lab
        else:
            labels = np.vstack((labels, lab))
    return data, labels

def getDatasetFromList(stock_list, start_date, end_date, current_db_day):
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

        td, tl = getMatrixFromDatabase(datapoints, str(start_date), str(end_date))
        #print td.shape
        if len(td.shape) == 1:
            continue

        if cnt == 0:
            train_data = td
            train_label = tl
        else:
            train_data = np.vstack((train_data, td))
            train_label = np.vstack((train_label, tl))
        cnt = cnt + 1
    sys.stdout.write('\n')
    return train_data, train_label

def getStockList():
    stock_list = open("../data/stock_list.txt").read().splitlines()
    #stock_list = stock_list[0::50]
    return stock_list

def convertDay2Date(day_str):
    t=time.strptime(day_str,'%Y-%m-%d')
    return date(t.tm_year,t.tm_mon,t.tm_mday)

def getWorkingDayList():
    today = date.today()
    sym = "AAPL"
    r = yf.Share(sym)
    fname = HISTORIC_DATA_LOC + "/" + sym + '.pkl'
    if not os.path.isfile(fname):
        dl = r.get_historical(START_DATE, str(today))
        pickle.dump(dl, open(fname, "wb"))
    else:
        dl = pickle.load(open(fname, "rb"))
    date_list = []
    for share in dl:
        date_list.append(share['Date'])
    date_list = date_list[::-1]
    return date_list

def writeLogSummary(new_event):
    ct = strftime("%Y-%m-%d %H:%M:%S")
    f = open("../change_log.txt", "append")
    f.write(ct + ": " + new_event + "\n")
    f.close()

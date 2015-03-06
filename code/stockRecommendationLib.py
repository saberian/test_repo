import numpy as np
import yahoo_finance as yf
import matplotlib.pyplot as plt
import os.path
import pickle
import yahoo_finance as yf
from time import strftime
import datetime


WEEK_DAYS = 5
MONTH_DAYS = 20
YEAR_DAYS = 240
SMALL_NUM = 0.0001
PROFIT_MARGIN = 1.05;





def mean(in_list):
    if len(in_list) == 0:
        return 0
    return sum(in_list)/len(in_list)

def getLinearCoeff(my_list, d, n):
    #my_list = [float(x) for x in t_list]
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

def getLabel(price_list, d):
    current_price = price_list[d]
    temp_label = -np.ones((3))
    next_day_price_ratio = price_list[d+1] / current_price
    next_week_max_price_ratio = max(price_list[d+1:d+5]) / current_price
    next_month_max_price_ratio = max(price_list[d+1:d+20]) / current_price
    #print "next_day_price_ratio:" +str(next_day_price_ratio)
    #print "next_week_max_price_ratio:" +str(next_week_max_price_ratio)
    #print "next_month_max_price_ratio:" +str(next_month_max_price_ratio)
    #print current_price
    #print price_list[d+1:d+20]
    if (next_day_price_ratio > PROFIT_MARGIN):
        temp_label[0] = 1;
    if (next_week_max_price_ratio > PROFIT_MARGIN):
        temp_label[1] = 1;
    if (next_month_max_price_ratio > PROFIT_MARGIN):
        temp_label[2] = 1;
    return temp_label

def getStockFeatureForDay(share_list, ret_label, print_opt):
    data = np.array([])
    labels = np.array([])
    if len(share_list) == 0:
        return data, labels
    if 'Close' not in share_list[0]:
        return data, labels
    n_days = len(share_list)
    if  n_days < (YEAR_DAYS+MONTH_DAYS+1):
        return data, labels
    if ret_label:
        ### getting range of valid days
        x_min = min(YEAR_DAYS, n_days)
        x_max = max(0, n_days-MONTH_DAYS)
        days = xrange(x_min, x_max);
    else:
        days = [n_days-1]

    ### get price history
    close_price = []
    open_price = []
    high_price = []
    low_price = []
    volume = []
    for share in share_list:
        try:
            close_price.append(float(share['Close']))
            open_price.append(float(share['Open']))
            high_price.append(float(share['High']))
            low_price.append(float(share['Low']))
            volume.append(float(share['Volume']))
        except Exception, e:
            return data, labels 

    ### reverse the order of lists

    if np.min(close_price) < 0.01:
        return data, labels

    close_price = np.array(close_price[::-1])
    open_price = np.array(open_price[::-1])
    high_price = np.array(high_price[::-1])
    low_price = np.array(low_price[::-1])
    volume = np.array(volume[::-1])

    max_daily_change = (high_price - low_price) / close_price
    daily_change = (close_price - open_price) / close_price

    ### get features per day
    for d in days:
        #print "day is: " + str(d)
        current_price = close_price[d] + SMALL_NUM;
        f1 = getAllStat(close_price, d)
        f2 = getAllStat(daily_change, d)
        f3 = getAllStat(max_daily_change, d)
        f4 = getAllStat(volume, d)
        f_all = np.hstack((f1,f2,f3, f4))
        if data.shape[0] == 0:
            data = np.zeros((1, len(f_all)))
            data[0,:] = f_all 
        else:
            data = np.vstack((data, f_all ))
        ### get label for the instance if needed
        if ret_label:
            temp_lab = np.array([getLabel(close_price, d)])#
            if labels.shape[0] == 0:
                labels = temp_lab
            else:
                labels = np.vstack((labels, temp_lab))
    #####
    if print_opt:
        plt.plot(close_price, label="close_price")
        #plt.plot(daily_change, label="daily_change")
        #plt.plot(max_daily_change, label="max_daily_change")
        plt.grid()
        #plt.legend()
        #plt.draw()
        plt.show()
    return data, labels



'''def downloadStockData(sym, start_day, end_day):
    r = yf.Share(sym)
    fname = '../historicDataNew/' + sym + '.pkl'
    if not os.path.isfile(fname):      
        dl = r.get_historical(start_day, end_day)
        pickle.dump(dl, open(fname, "wb"))
    else:
        dl = pickle.load(open(fname, "rb"))
    return dl
    '''

def getStockFeatureForSymbol(share_list, ret_label, sym_data):
    if len(share_list) == 0:
        return sym_data
    if 'Close' not in share_list[0]:
        return sym_data
    n_days = len(share_list)
    if  n_days < (YEAR_DAYS+MONTH_DAYS+1):
        return sym_data
    if ret_label:
        ### getting range of valid days
        x_min = min(YEAR_DAYS, n_days)
        x_max = max(0, n_days-MONTH_DAYS)
        days = xrange(x_min, x_max);
    else:
        days = [n_days-1]

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
            f_all = np.hstack((f1,f2,f3, f4))
            res = f_all
            if ret_label:
                temp_lab = np.array([getLabel(close_price, d)])
                res = [f_all, temp_lab]
            sym_data[current_date] = res
    return sym_data


def getMatrixFromDatabse(data_points):
    data = np.array([])
    labels = np.array([])

    sorted_days = sorted(data_points.keys())
    for day in sorted_days:
        f_all = data_points[day][0]
        lab = data_points[day][1]
        if data.shape[0] == 0:
            #data = np.zeros((1, len(f_all)))
            data = f_all 
        else:
            data = np.vstack((data, f_all ))
        if labels.shape[0] == 0:
            labels = lab
        else:
            labels = np.vstack((labels, lab))
    return data, labels

def writeLogSummary(new_event):
    ct = strftime("%Y-%m-%d %H:%M:%S")
    f = open("../change_log.txt", "append")
    f.write(ct + ": " + new_event + "\n")
    f.close()











from constants import *
import stockRecommendationLib as srl
import pickle
import numpy as np



def getArrays(share_list):
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
    arrays = {"Date": date_list, "Close": close_price, "Open": open_price, \
              "High": high_price, "Low": low_price, "Volume": volume}
    return arrays

historic_date_index = {}
historic_data = {}
historic_array_data = {}
sym_list = srl.getStockList()
for sym in sym_list:
    print sym
    fname = HISTORIC_DATA_LOC + "/" + sym + '.pkl'
    dl = pickle.load(open(fname, "rb"))
    historic_data[sym] = dl
    historic_array_data[sym] = getArrays(dl)
    date_dict = {}
    for i in xrange(0, len(dl)):
        shr = dl[i]
        if type(shr) == dict:
            if 'Date' in shr and len(shr) > 1:
                date_dict[shr['Date']] = i
    historic_date_index[sym] = date_dict

print len(historic_data)


pickle.dump([historic_data, historic_array_data, historic_date_index], \
            open(HISTORIC_DATA_LOC + "/date_indexed_historic.py", "w"))
import numpy as np
import sys
import h5py
import matplotlib.pyplot as plt
import stockRecommendationLib as srl
caffe_root = '../../caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
from constants import *
import yahoo_finance as yf
import pickle
from strategy import *
from portfolio import *
from predictionModel import *

def simulateModelResults():
    start_day = "2014-03-31"
    end_day = "2015-03-13"
    index_sym = "UHAL"
    tc = 1
    my_init_cash = 10000
    my_strategy = Strategy(trade_period=70, 
                           n_max_trade_per_day = 1,
                           max_cash_per_trade = 500,
                           trade_cost = tc, 
                           below_thr = 0.05,
                           above_thr = 0.10,
                           min_target_profit = 0.10,
                           min_volume = 1000000)
    my_portfolio = Portfolio(cash=my_init_cash, trade_cost=tc)
    my_model = PredictionModel(model_type="s2")

    print "loading the data"
    historic_data, historic_array_data, historic_date_index = pickle.load(open(HISTORIC_DATA_LOC + \
                                                          "/date_indexed_historic.pkl"))
    
    print "data is loaded"

    day_list = srl.getWorkingDayList()
    sym_list = srl.getStockList()
    start_ind = day_list.index(start_day)
    end_ind = day_list.index(end_day)

    day_cnt = 0
    wealth_array = []
    index_wealth_array = []
    n_index_stk = 0 
    index_remain = 0

    for i in xrange(start_ind, end_ind):
        print "======================================================="
        day = day_list[i]
        print "started day " + day
        
        today_stock_info = {} 
        for sym in sym_list:
            if day in historic_date_index[sym]:
                ind = historic_date_index[sym][day]
                today_stock_info[sym] =  historic_data[sym][ind]
        

        model_res = my_model.getStockScores(today_stock_info, historic_array_data, day)

        #print "morning update"
        my_portfolio.updateMorning(today_stock_info, day, my_strategy)
        my_portfolio.printStat()
        #my_portfolio.printOwnedStocks()

        # comapre with index strategy ##
        index_prc = float(today_stock_info[index_sym]["Open"])
        if i == start_ind:
            n_index_stk = int(my_init_cash/ index_prc)
            index_remain = my_init_cash - n_index_stk * index_prc
        index_wealth = (n_index_stk * index_prc) + index_remain
        print "index_wealth %.2f" % index_wealth
        index_wealth_array.append(index_wealth)
        wealth_array.append(my_portfolio.getTotalValue())

        #print "sell stocks"
        sell_ind_list = my_strategy.getSellRecommendations(my_portfolio, day)
        my_portfolio.sellFromList(sell_ind_list, day)
        #my_portfolio.printStat()

        #print "buy stocks"
        buy_stk_list = my_strategy.getBuyRecommendations(model_res, today_stock_info, my_portfolio._cash)
        my_portfolio.buyFromList(buy_stk_list)
        #my_portfolio.printStat()

        #print "afternoon update"
        my_portfolio.updateAfternoon()
        my_strategy.printPerformance(my_portfolio)
        #my_portfolio.printStat()

        day_cnt += 1
        #if day_cnt > 3: 
        #    break

    print len(wealth_array)
    n = len(wealth_array)
    plt.plot(xrange(0, n), wealth_array, label="wealth")
    plt.plot(xrange(0, n), index_wealth_array, label="index wealth")

    plt.grid()
    plt.draw()
    plt.legend(loc='upper left')
    plt.savefig("sim_wealth.png")  

#=================================================================
#=================================================================
#=================================================================

simulateModelResults()




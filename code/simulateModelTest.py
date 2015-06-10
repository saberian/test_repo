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

def getPredictionFromModel(test_data, model_prp):
    if model_prp["type"] == "caffe":
        caffe.set_mode_gpu()
        net = caffe.Classifier(model_prp["deploy_file"], model_prp["cf_model_dir"] + "/" + model_prp["model_file_name"])
        res = net.forward_all(**{"data": test_data})[model_prp["output_node"]].squeeze()
    return res

def getHdf5Data(test_file_list, test_file_loc):
    test_data = np.array([])
    test_label = np.array([])
    day_labs = []
    sym_list = []
    test_files_names = open(test_file_loc + "/" + test_file_list).read().splitlines()
    ns = 0
    nf = 0

    for fn in test_files_names:
        f = h5py.File(fn, 'r')
        td = np.array(f["data"])
        tl = np.array(f["label"])
        ns += td.shape[0]
        nf = td.shape[2]

    test_data = np.zeros((ns, 1, nf, 1))
    test_label = np.zeros((ns))
    cnt = 0
    for fn in test_files_names:
        f = h5py.File(fn, 'r')
        td = np.array(f["data"])
        tl = np.array(f["label"])
        day_labs += f['day_lab']
        sym_list += f['symbol'] #[ for e in xrange(0, td.shape[0])]
        ns = td.shape[0]
        test_data[cnt:(ns+cnt), :, : , :] = td
        test_label[cnt:(cnt+ns)] = tl
        cnt += ns
    return test_data, test_label, day_labs, sym_list


def simulateModelResults(test_res, test_label, day_labs, data_sym_list):
    day_list = srl.getWorkingDayList()
    day_ind_list = {}
    for i in xrange(0, len(day_labs)):
        day = day_labs[i]
        if day not in day_ind_list:
            day_ind_list[day] = [i]
        else:
            day_ind_list[day].append(i)

    ########### index historic data based on date ###########
    historic_date_index = {}
    historic_data = {}
    sym_list = srl.getStockList()
    for sym in sym_list:
        fname = HISTORIC_DATA_LOC + "/" + sym + '.pkl'
        dl = pickle.load(open(fname, "rb"))
        historic_data[sym] = dl
        date_dict = {}
        for i in xrange(0, len(dl)):
            shr = dl[i]
            if type(shr) == dict:
                if 'Date' in shr and len(shr) > 1:
                    date_dict[shr['Date']] = i
        historic_date_index[sym] = date_dict

    my_strategy = strategy(trade_period=10, 
                           n_max_trade_per_day = 1,
                           max_cash_per_trade = 500,
                           trade_cost = 10, 
                           below_thr = 0.02,
                           above_thr = 0.05,
                           min_volume = 1000000)
    my_portfolio = portfolio(cash=10000, trade_cost=10)


    day_cnt = 0
    wealth_array = []
    prtfolio_change = []
    for day in day_list:
        if day in day_ind_list:
            print "======================================================="
            print "started day " + day
            
            today_stock_info = {} 
            for sym in sym_list:
                if day in historic_date_index[sym]:
                    ind = historic_date_index[sym][day]
                    today_stock_info[sym] =  historic_data[sym][ind]
            
            model_res = []
            ind_list = day_ind_list[day]
            for i in ind_list:  #test_res[i] #np.random.rand()
                if data_sym_list[i] == "QQQ":
                    val = 1
                else:
                    val = 0
                print data_sym_list[i]
                model_res.append((data_sym_list[i], val))


            print "morning update"
            my_portfolio.updateMorning(today_stock_info, day)
            my_portfolio.printStat()
            wealth_array.append(my_portfolio.getTotalValue())

            print "sell stocks"
            sell_ind_list = my_strategy.getSellRecommendations(my_portfolio, day)
            my_portfolio.sellFromList(sell_ind_list, day)
            my_portfolio.printStat()

            print "buy stocks"
            buy_stk_list = my_strategy.getBuyRecommendations(model_res, today_stock_info, my_portfolio._cash)
            my_portfolio.buyFromList(buy_stk_list)
            my_portfolio.printStat()

            print "afternoon update"
            my_portfolio.updateAfternoon()
            my_portfolio.printStat()

            day_cnt += 1
            if day_cnt > 3: 
                break
    print len(wealth_array)
    n = len(wealth_array)
    plt.plot(xrange(0, n), wealth_array, label="wealth")
    plt.grid()
    plt.draw()
    plt.legend(loc='upper left')
    plt.savefig("sim_wealth.png")  

#=================================================================
#=================================================================
#=================================================================

prp = {}
prp["type"] = "caffe"
data_name = 'dataset_eval_2015-05-04'

prd_name = "caffe_reg_v4_week"
prd_type = "reg"
protobuf_loc = 'caffe_reg_protobuf'

prp["deploy_file"] = protobuf_loc + "/deploy4.prototxt"
prp["output_node"] = "fc1"
batch_size = 1000
n_max_itr = 100000
n_batch_test = 100 # not very important
prp["model_file_name"] = prd_name + "_iter_"+  str(n_max_itr) + ".caffemodel"
prp["cf_model_dir"] = MODELS_DIR + "/" + prd_name

test_data, test_label, day_labs, sym_list = getHdf5Data("test_" + prd_type + ".txt", srl.HDF5_LOC)
#print sym_list
#print day_labs

#test_res = getPredictionFromModel(test_data, prp)
#pickle.dump(test_res, open("res_temp.pkl", "w"))
test_res = pickle.load(open("res_temp.pkl"))

#print sym_list

res = simulateModelResults(test_res, test_label, day_labs, sym_list)




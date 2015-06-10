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

    trade_period = 10
    n_max_trade_per_day = 3
    cash_per_trade = 500
    trade_cost = 10 
    below_thr = 0.02
    above_thr = 0.05
    MIN_VOLUME = 1000000

    total_budget = 10000
    current_cash = total_budget
    protfolio = []
    protfolio_value = 0
    funds_in_clearing = []


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


    day_cnt = 0
    wealth_array = []
    prtfolio_change = []
    for day in day_list:
        if day in day_ind_list:
            print "======================================================="
            print "started day " + day 
            curr_day_ind = day_list.index(day)
            ############ get cleared funds ####################
            in_clearning_cash = 0
            temp_fund_in_cl = []
            cleared_cash = 0
            for i in xrange(0, len(funds_in_clearing)):
                d, v = funds_in_clearing[i]
                if curr_day_ind - day_list.index(d) > 2:
                    cleared_cash += v
                else:
                    in_clearning_cash += v
                    temp_fund_in_cl.append((d,v))
            current_cash += cleared_cash
            funds_in_clearing = temp_fund_in_cl
            print "cleared %.2f cash: " % cleared_cash
            #print "current cash is %.2f  and %.2f is in clearning" % (current_cash, in_clearning_cash)
            ######   evaluate the current portfolio   ##########
            #prf_change = 0
            new_prf_val = 0
            for (n_st, stk) in protfolio:
                sym = stk["Symbol"]
                ind = historic_date_index[sym][day]
                new_stk =  historic_data[sym][ind]
                stk_ch = n_st * (float(new_stk['Open']) - float(stk['Open']))
                print sym + " changed from %.2f to %.2f" % (float(stk['Open']), float(new_stk['Open']))
                #prf_change += stk_ch
                new_prf_val += n_st * float(new_stk['Open'])

            prf_change = new_prf_val  - protfolio_value
            protfolio_value = new_prf_val 
            print "profolio change is %.2f , prtf val is %.2f " % (prf_change, protfolio_value)
            total_wealth = protfolio_value + current_cash + in_clearning_cash
            print "cash: %.2f, in cleanring: %.2f, prtf val: %.2f, total w: %.2f" \
                  %(current_cash, in_clearning_cash, protfolio_value, total_wealth)
            wealth_array.append(total_wealth)
            prtfolio_change.append(prf_change)
            ##### sell stocks ##################################
            rec_cash = 0
            new_portfolio = []
            for (n_st, stk) in protfolio:
                sym = stk["Symbol"]
                ind = historic_date_index[sym][day]
                new_stk =  historic_data[sym][ind]
                purchase_day_ind = day_list.index(stk['Date'])
                pur_flag = False
                #print sym
                #print "trade prd = " + str((curr_day_ind - purchase_day_ind))
                if  (curr_day_ind - purchase_day_ind)  > trade_period:
                    print "sell %s stock because it is more than trade period" % sym
                    pur_flag = True
                elif float(new_stk['Open']) > float(stk['Open']) * (1 + above_thr):
                    print "sell %s stock because it met the target" % sym
                    pur_flag = True
                elif float(new_stk['Open']) < float(stk['Open']) * (1 - below_thr):
                    print "sell %s stock because it went below thr" % sym
                    pur_flag = True
                if pur_flag == True:
                    stk_val = n_st * float(new_stk['Open'])
                    rec_cash +=  stk_val - trade_cost
                    protfolio_value -= stk_val
                    in_clearning_cash += stk_val - trade_cost
                else:
                    new_portfolio.append((n_st, stk))

            protfolio = new_portfolio


            print "sold %.2f worth of stock" % rec_cash
            total_wealth = protfolio_value + current_cash + in_clearning_cash
            print "cash: %.2f, in cleanring: %.2f, prtf val: %.2f, total w: %.2f" \
                  %(current_cash, in_clearning_cash, protfolio_value, total_wealth)
            funds_in_clearing.append((day, rec_cash))
            #### buy stocks ################################
            ind_list = day_ind_list[day]
            day_res = test_res[ind_list]
            top_res = sorted(range(len(day_res)), key=lambda i: -day_res[i])[0:n_max_trade_per_day]
            selected_ind = [ind_list[e] for e in top_res]
            selected_sym = [data_sym_list[e] for e in selected_ind]
            for sym in selected_sym:
                dl = historic_data[sym] 
                if day not in historic_date_index[sym]:
                    print historic_date_index[sym]
                    print sym
                    print day
                    print "ddddddddddddd" 
                    continue

                current_stk = dl[historic_date_index[sym][day]]
                if current_stk["Volume"] > MIN_VOLUME and \
                   current_cash > (cash_per_trade + 2 * trade_cost):
                    n_stk = int(cash_per_trade / float(current_stk['Open']))
                    if n_stk > 0 :
                        protfolio.append((n_stk, current_stk))
                        stock_cost = float(current_stk['Open']) * n_stk
                        current_cash -=  (stock_cost + trade_cost)
                        protfolio_value += stock_cost 
                        print " buy %d of %s stock valued at %.2f, remaining fund is %.2f" \
                         %(n_stk, sym, float(current_stk['Open']) * n_stk, current_cash)
            total_wealth = protfolio_value + current_cash + in_clearning_cash
            #print "total wealth is %.2f : " %(total_wealth)
            print "cash: %.2f, in cleanring: %.2f, prtf val: %.2f, total w: %.2f" \
                  %(current_cash, in_clearning_cash, protfolio_value, total_wealth)

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

    plt.clf()
    plt.grid()
    plt.draw()
    plt.plot(xrange(0, n), prtfolio_change, label="prtfolio change")
    plt.legend(loc='upper left')
    plt.savefig("sim_prf.png")


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




import stockRecommendationLib as srl

class OwnedStock:
    def __init__(self, active=False, n_stk=0, stk=None, target_price=0, thr_price=0, age=0):
        self._active = active
        self._n_stk = n_stk
        self._sym = stk["Symbol"]
        self._purchase_date = stk["Date"]
        self._purchase_price = float(stk["Open"])
        self._current_price = float(stk["Open"])
        self._pur_stk_1 = stk
        self._cur_stk_1 = stk
        self._target_price = target_price
        self._thr_price = thr_price
        self._age = age
    def printStat(self):
        st = str(self._active) + " " + self._sym + " " + str(self._n_stk) + " " + \
             "age: %d tg: %.2f th: %.2f" %(self._age, self._target_price, self._thr_price)
        print st
    def updateStk(self, stk):
        self._current_price = float(stk["Open"])
        self._cur_stk_1 = stk


class Portfolio:
    def __init__(self, cash=0, trade_cost=10):
        self._cash = cash
        self._owned_stock_list = []
        self._stock_value = 0
        self._funds_in_clearing = []
        self._in_clearning_cash = 0
        self._trade_cost = trade_cost
        self._work_day_list = srl.getWorkingDayList()

    def getTotalValue(self):
        return self._stock_value + self._cash + self._in_clearning_cash

    def printStat(self):
        print "cash: %.2f, in cleanring: %.2f, portfolio value: %.2f, total wealth: %.2f" \
            %(self._cash, self._in_clearning_cash, self._stock_value, self.getTotalValue())

    def printOwnedStocks(self):
        print "Owned Stocks: "
        for ostk in self._owned_stock_list:
            ostk.printStat()
        print "Owned Stocks ++++++++++++++++++++ "

    def buyFromList(self, stock_list):
        for o_stk in stock_list:
            self.buyStock(o_stk)

    def buyStock(self,ostk):
        stk_cost = float(ostk._purchase_price) * ostk._n_stk
        self._cash -= (stk_cost + self._trade_cost)
        ostk._active = True
        self._owned_stock_list.append(ostk)
        self._stock_value += stk_cost
        print " buy %d of %s stock valued at %.2f with total %.2f" % \
                (ostk._n_stk, ostk._sym, ostk._purchase_price, stk_cost)

    '''def buyStock(self, stk, n_stk):
        stk_cost = float(stk['Open']) * n_stk
        self._cash -= (stk_cost + self._trade_cost)
        self._owned_stock_list.append(owned_stock(True, n_stk, stk))
        self._stock_value += stk_cost
        print " buy %d of %s stock valued at %.2f with total %.2f" % \
                (n_stk, stk['Symbol'], float(stk['Open']), stk_cost)'''

    def sellFromList(self, ind_list, date):
        for stk_id in ind_list:
            self.sellStock(stk_id, date)

    def sellStock(self, stk_id, date):
        self._owned_stock_list[stk_id]._active = False
        stk = self._owned_stock_list[stk_id]
        stk_val = stk._n_stk * stk._current_price
        self._stock_value -= stk_val
        rec_cash =  (stk_val - self._trade_cost)
        self._funds_in_clearing.append((date, rec_cash))
        self._in_clearning_cash += rec_cash

    def updateAfternoon(self):
        ### consistency checks
        temp_cl = 0
        for d, v in self._funds_in_clearing:
            temp_cl += v
        if temp_cl != self._in_clearning_cash:
            print "in clearing cash is not consitent"
        ###########  ###############
        new_ow_stk = []
        cur_val = 0
        for st in self._owned_stock_list:
            if st._active:
                new_ow_stk.append(st)
                cur_val += st._n_stk * st._current_price
        if cur_val != self._stock_value:
            print "stock value is not consitent"
        self._owned_stock_list = new_ow_stk

    def updateMorning(self, new_stock_list, today, strategy):
        ###### update portfolio with new stock info
        new_stk_val = 0
        for i in xrange(0, len(self._owned_stock_list)): 
            sym = self._owned_stock_list[i]._sym
            self._owned_stock_list[i]._age += 1
            self._owned_stock_list[i].updateStk(new_stock_list[sym])#_cur_stk = new_stock_list[sym]
            tg, th = strategy.adjustTargets(self._owned_stock_list[i])
            self._owned_stock_list[i]._target_price = tg
            self._owned_stock_list[i]._thr_price = th
            new_stk_val += self._owned_stock_list[i]._n_stk * float(new_stock_list[sym]["Open"])
        print "yesterday stock val: %.2f and today val: %.2f" % (self._stock_value, new_stk_val)
        self._stock_value = new_stk_val
        ####### clear cash values ###############
        self._in_clearning_cash = 0
        temp_in_cleaning = [];
        cleared_cash = 0
        for d, v in self._funds_in_clearing:
            today_ind = self._work_day_list.index(today)
            if today_ind - self._work_day_list.index(d) > 2:
                cleared_cash += v
            else:
                self._in_clearning_cash += v
                temp_in_cleaning.append((d,v))
        self._funds_in_clearing = temp_in_cleaning
        self._cash += cleared_cash
        print "cleared %.2f cash" % cleared_cash

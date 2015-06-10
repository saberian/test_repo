
import stockRecommendationLib as srl
from portfolio import *


class Strategy:
    def __init__(self, trade_period=10, 
                       n_max_trade_per_day = 3,
                       max_cash_per_trade = 500,
                       trade_cost = 10, 
                       below_thr = 0.02, 
                       above_thr = 0.05,
                       min_target_profit = 0.10,
                       min_volume = 1000000):
        self._trade_period = trade_period
        self._n_max_trade_per_day = n_max_trade_per_day 
        self._max_cash_per_trade = max_cash_per_trade
        self._trade_cost = trade_cost
        self._below_thr = below_thr
        self._above_thr = above_thr
        self._min_target_profit = min_target_profit
        self._min_volume = min_volume
        self._today_prediction = []
        self._day_list = srl.getWorkingDayList()
        self.n_good_recom = 0
        self.n_total_recoms = 0
        self.n_timeout_recom = 0
        self.n_bad_recom = 0

    def printPerformance(self, portfolio):
        n = float(self.n_total_recoms - len(portfolio._owned_stock_list))
        if n == 0 :
            print "strategy performance: no sold stock"
        else:
            print "strategy performance:  n_sold: %d, good_recom: %.2f, timeout_recom %.2f, bad_recom %.2f" \
                  % (int(n), self.n_good_recom/n, self.n_timeout_recom/n, self.n_bad_recom/n)

    def adjustTargets(self, ostk):
        pur_price = ostk._purchase_price
        cur_price = ostk._current_price
        old_tg = ostk._target_price
        old_th = ostk._thr_price
        age = ostk._age

        tg = old_tg
        th = old_th

        if cur_price > pur_price :
            #diff = (cur_price - pur_price) / 2
            #th = cur_price - diff
            #tg = cur_price + diff
            th = cur_price * (1 - self._below_thr)
            tg = cur_price * (1 + self._above_thr)
            #print "updated targets"
            
        return tg, th

    def getSellRecommendations(self, portfolio, today):
        rec_list = []
        curr_day_ind = self._day_list.index(today)
        for i in range(len(portfolio._owned_stock_list)):
            ostk = portfolio._owned_stock_list[i]
            sym = ostk._sym
            cur_price = ostk._current_price
            pur_price = ostk._purchase_price
            pur_day_ind = self._day_list.index(ostk._purchase_date)
            sell_flag = False
            if  ostk._age  > self._trade_period:
                print "sell %s stock because it is more than trade period" % sym
                sell_flag = True
            elif cur_price >= ostk._target_price:
                print "sell %s stock because it met the target" % sym
                sell_flag = True
            elif cur_price < ostk._thr_price:
                print "sell %s stock because it went below thr" % sym
                sell_flag = True
            if sell_flag:
                prt = (ostk._current_price - ostk._purchase_price) / ostk._purchase_price
                if prt > self._min_target_profit:
                    self.n_good_recom += 1
                elif prt < -self._below_thr:
                    self.n_bad_recom += 1
                else:
                    self.n_timeout_recom += 1
                rec_list.append(i)
        return rec_list

    def getBuyRecommendations(self, model_res, today_stock_info, current_cash):
        rec_list = []
        top_res = sorted(range(len(model_res)), key=lambda i: -model_res[i][1])
        n_trade = 0
        for i in top_res:
            sym = model_res[i][0]
            #print "sym %s and score: %.2f" % (sym, model_res[i][1])
            stk = today_stock_info[sym]
            if stk["Volume"] > self._min_volume and \
                current_cash > (self._max_cash_per_trade + 2 * self._trade_cost):
                stk_price = float(stk['Open'])
                n_stk = int(self._max_cash_per_trade / stk_price)
                if n_stk > 0  and n_trade < self._n_max_trade_per_day:
                    current_cash -= n_stk * stk_price
                    n_trade += 1
                    tg = stk_price * (1 + self._above_thr)
                    th = stk_price * (1 - self._below_thr)
                    rec_list.append(OwnedStock(False, n_stk, stk, tg, th, 0))
                    self.n_total_recoms += 1
            if n_trade >= self._n_max_trade_per_day:
                break
        return rec_list


import stockRecommendationLib as srl


class PredictionModel:
    def __init__(self, model_type):
        self._model_type = model_type

    def getStockScores(self, today_stock, historic_array_data, date):
        model_res = []
        if self._model_type == "s1":
            for sym in today_stock:  #test_res[i] #np.random.rand()
                if sym == "UHAL": #"QQQ":
                    val = 1
                else:
                    val = 0
                model_res.append((sym, val))
        if self._model_type == "s2":
            n = 50
            for sym in today_stock:
                cur_ind = historic_array_data[sym]["Date"].index(date)
                avg = srl.getAverage(historic_array_data[sym]["Open"], cur_ind, n)
                if avg == 0:
                    val = -10
                else:
                    val = (avg -  historic_array_data[sym]["Open"][cur_ind]) #/ avg
                model_res.append((sym, val))

        return model_res
import os
import sys
import stockRecommendationLib as srl


target =  sys.argv[1] #"history" or "database"
today = sys.argv[2]
n_jobs = int(sys.argv[3])
if target == "database":
    last_day = sys.argv[4]#

stock_list = srl.getStockList()

n_items = len(stock_list)
n_chunk =  n_items/ n_jobs

print [n_items, n_jobs, n_chunk]

start_ind  = 0
end_ind = n_chunk

while start_ind < end_ind:
    if target == "database":
        command = "python updateDatabase.py %s %s %d %d" % (last_day, today, start_ind, end_ind)
    if target == "history":
        command = "python updateHistoricData.py %s %d %d" % (today, start_ind, end_ind)

    os.system(command + " &")
    print command 
    start_ind += n_chunk
    end_ind += n_chunk
    if end_ind > n_items:
        end_ind = n_items
    #print [start_ind, end_ind]

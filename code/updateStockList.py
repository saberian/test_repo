import os
import sys 
import stockRecommendationLib as srl
from constants import *


url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
os.system("wget ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt   >/dev/null 2>&1 ")
f = open("nasdaqlisted.txt", "r")
f_out = open(DATA_DIR + "/stock_list.txt", "w")
# get rid of firt line
l = f.readline()
cnt = 0
for l in f:
    l = l.strip()
    ws = l.split("|")
    if len(ws[0]) > 10:
        continue
    if ws[3] == "N" and ws[4]=="N":
       f_out.write(ws[0]+"\n")
       cnt +=1

f_out.close()

os.system("rm nasdaqlisted.txt ")
log_str = "new stock list with %d enteries is created" % cnt 
print log_str
srl.writeLogSummary(log_str)


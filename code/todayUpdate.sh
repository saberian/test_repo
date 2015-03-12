
last_day="non"
today="2015-03-11"
classifier_date="2015-03-09"
echo $last_day
LOG_FILE=../change_log.txt
#echo "============= started a full update cycle========" >>  $LOG_FILE
#python updateStockList.py
#python updateHistoricData.py $today
#python updateDatabase.py  $last_day  $today
#python prepareTrainTestSamples.py  $today
#python trainClassifier.py  $today
#python getTodaySamples.py  $today
python getTodayRecommendation.py  $today $classifier_date
echo "============= finished a full update cycle========" >> $LOG_FILE 

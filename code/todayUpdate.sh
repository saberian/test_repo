
today="2015-03-18"
last_day="2015-03-13"
LOG_FILE=../change_log.txt
echo "============= started a full update cycle========" >>  $LOG_FILE
python parallelExecuter.py "history" $today 10
python parallelExecuter.py "database" $today 10 $last_day
python prepareFullDataset.py  $today
python trainClassifier.py "dataset_full_"$today 0
python getDaySamples.py $today
python getRecommendation.py $today "linear_reg_month_dataset_full_"$today
echo "============= finished a full update cycle========" >> $LOG_FILE 

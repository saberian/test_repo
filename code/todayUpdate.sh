
last_day="2015-03-04"
echo $last_day
python updateHistoricData.py
python updateDatabase.py  $last_day
python prepareTrainTestSamples.py
python trainClassifier.py
python getTodaySamples.py
python getTodayRecommendation.py
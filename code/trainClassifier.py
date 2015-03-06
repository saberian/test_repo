import numpy as np
import os
import pickle
import stockRecommendationLib as srl
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import datetime


def visualizeIterError(clf, clf_name, train_data, train_label, \
                        test_data, test_label):
    prm = clf.get_params()
    n_estimators = prm['n_estimators']

    err_train = np.zeros((n_estimators,))
    for i, y_pred in enumerate(clf.staged_predict(train_data)):
        err_train[i] = 1-accuracy_score(train_label, y_pred)
    plt.plot(np.arange(n_estimators) + 1, err_train,
        label='Train Error',
        color='blue')
    err_test = np.zeros((n_estimators,))
    for i, y_pred in enumerate(clf.staged_predict(test_data)):
        err_test[i] = 1-accuracy_score(test_label, y_pred)
    plt.plot(np.arange(n_estimators) + 1, err_test,
            label='Test Error',
            color='red')
    plt.grid()
    plt.legend(loc='upper right')
    plt.draw()
    plt.ylabel('error rate')
    plt.xlabel('num of weak learners')
    plt.savefig(clf_name + "_iter_evolution.png")


def testClassifier(clf, test_data, test_label):
    n_test_samples = test_data.shape[0]
    rg_test =  sum(test_label<0) / float(n_test_samples)
    print "num test sample: " + str(n_test_samples)
    print "random guessing acc: " + str(max(rg_test, 1-rg_test))
    clf_test = clf.score(test_data, test_label)
    print "clf test accuracy: " + str(clf_test)

def trainClassifier(train_data, train_label, name):
    print "============================"
    print "training classifier " + name 
    # get performance of random guessing
    n_training_samples = train_data.shape[0]
    rg_train = sum(train_label<0) / float(n_training_samples)
    print "num training sample: " + str(n_training_samples)
    print "random guessing acc: " + str(max(rg_train, 1-rg_train)) 

    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, 
        max_depth=3, random_state=0).fit(train_data, train_label)

    clf_train =  clf.score(train_data, train_label)
    print "clf training accuracy: " + str(clf_train)
    pickle.dump(clf, open("../models/"+name+".pkl", "wb"))
    srl.writeLogSummary("classifier " + name + " is trained")
    return clf


today = str(datetime.date.today())

train_data, train_label, \
            test_data, test_label, \
            extra_train_data, extra_train_label =\
            pickle.load(open("../data/processedData/train_data_" + today + ".pkl", "rb"))

clf_name = "gbdt_month_test_" + today
clf_test_month = trainClassifier(train_data, train_label[:,2], clf_name)
testClassifier(clf_test_month, test_data, test_label[:,2])
visualizeIterError(clf_test_month, clf_name, train_data, train_label[:,2], \
                        test_data, test_label[:,2])

full_train_data = np.vstack((train_data, extra_train_data))
full_train_label = np.vstack((train_label, extra_train_label))
clf_name = "gbdt_month_full_" + today
clf_full_month = trainClassifier(full_train_data, full_train_label[:,2], clf_name)



import numpy as np
import os
import pickle
import stockRecommendationLib as srl
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import datetime
import sys
from constants import *


def visualizeGbdtIterError(clf, clf_name, train_data, train_label, \
                        test_data, test_label):
    prm = clf.get_params()
    n_estimators = prm['n_estimators']
    plt.clf()

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
    plt.savefig(MODELS_DIR + "/"+ model_dir_name + "/" + clf_name + "_iter_evolution.png")


def testPredictor(clf, test_data, test_label, name):
    print "testing " + name
    n_test_samples = test_data.shape[0]
    rg_test =  sum(test_label==0) / float(n_test_samples)
    model_dir =  MODELS_DIR + "/" + name
    clf_prd = clf.decision_function(test_data)

    clf_prd = clf_prd.squeeze()

    print clf_prd.shape
    print [max(clf_prd), min(clf_prd)]
    inds = np.argsort(-clf_prd)
    n_top = 10
    cnt = 0
    for i in xrange(0, n_top):
        print [clf_prd[inds[i]], test_label[inds[i]]]
        if test_label[inds[i]] == 1:
            cnt += 1

    top_acc = float(cnt)/n_top 

    print "====== down acc"
    cnt = 0
    for j in xrange(0, n_top):
        i = n_test_samples - j - 1
        print [clf_prd[inds[i]], test_label[inds[i]]]
        if test_label[inds[i]] == 0:
            cnt += 1

    down_acc = float(cnt)/n_top 



    fpr, tpr, thrs = roc_curve(test_label, clf_prd)
    clf_auc = roc_auc_score(test_label, clf_prd)
    plt.clf()
    plt.plot(fpr, tpr, label=(name + " %.2f - %.2f - %.2f" % (clf_auc, top_acc, down_acc)))
    plt.grid()
    plt.draw()
    plt.legend(loc='lower right')
    plt.savefig(model_dir + "/roc.png")


def trainGbdtClassifier(train_data, train_label, name):
    print "============================"
    print "training classifier " + name
    model_dir =  MODELS_DIR + "/" + name
    if not os.path.exists(model_dir):
        os.system("mkdir " + model_dir)

    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, 
        max_depth=3, random_state=0).fit(train_data, train_label)

    pickle.dump(clf, open(model_dir + "/model.pkl", "wb"))
    srl.writeLogSummary("classifier " + name + " is trained")
    return clf

def trainLogisticRegClassifier(train_data, train_label, name):
    print "============================"
    print "training classifier " + name
    model_dir =  MODELS_DIR + "/" + name
    if not os.path.exists(model_dir):
        os.system("mkdir " + model_dir)

    clf = LogisticRegression().fit(train_data, train_label)

    pickle.dump(clf, open(model_dir + "/model.pkl", "wb"))
    srl.writeLogSummary("classifier " + name + " is trained")
    return clf


def trainGbdtRegressor(train_data, train_label, name):
    print "============================"
    print "training gbdt regressor " + name
    model_dir =  MODELS_DIR + "/" + name
    if not os.path.exists(model_dir):
        os.system("mkdir " + model_dir)
    clf = GradientBoostingClassifier().fit(train_data, train_label)
    pickle.dump(clf, open(model_dir + "/model.pkl", "wb"))
    srl.writeLogSummary("regressor " + name + " is trained")
    return clf

def trainLinearRegressor(train_data, train_label, name):
    print "============================"
    print "training linear regressor " + name
    model_dir =  MODELS_DIR + "/" + name
    if not os.path.exists(model_dir):
        os.system("mkdir " + model_dir)
    clf = LinearRegression().fit(train_data, train_label)
    pickle.dump(clf, open(model_dir + "/model.pkl", "wb"))
    srl.writeLogSummary("regressor " + name + " is trained")
    return clf


print "++++++++++++++++++++++++++++++++++++++++++"
data_name = sys.argv[1]
test_flag = int(sys.argv[2])

if test_flag == 1:
    train_data, train_label, test_data, test_label =\
            pickle.load(open(PROCESSED_DATA_DIR + "/" + data_name + ".pkl", "rb"))
if test_flag == 0:
    train_data, train_label =\
            pickle.load(open(PROCESSED_DATA_DIR + "/" + data_name + ".pkl", "rb"))

### convert ratio's into labels
train_label_cls = (train_label>PROFIT_MARGIN) + 0.0

#model_dir_name = "gbdt_month_" + data_name
#clf_test_month = trainGbdtClassifier(train_data, train_label_cls[:,2], model_dir_name)
#clf_test_month = pickle.load(open(MODELS_DIR + "/" + model_dir_name +"/model.pkl", "rb"))

#model_dir_name_reg = "gbdt_reg_month_" + data_name
#clf_reg_test_month = trainGbdtRegressor(train_data, train_label[:,2], model_dir_name_reg)

model_dir_month_lin_reg = "linear_reg_month_" + data_name
clf_lin_reg_test_month = trainLinearRegressor(train_data, train_label[:,2], model_dir_month_lin_reg)

model_dir_week_lin_reg = "linear_reg_week_" + data_name
clf_lin_reg_test_week = trainLinearRegressor(train_data, train_label[:,1], model_dir_week_lin_reg)

model_dir_day_lin_reg = "linear_reg_day_" + data_name
clf_lin_reg_test_day = trainLinearRegressor(train_data, train_label[:,0], model_dir_day_lin_reg)


#model_dir_name_log_reg = "logistic_reg_month_" + data_name
#clf_log_reg_test_month = trainLogisticRegClassifier(train_data, train_label_cls[:,2], model_dir_name_log_reg)


if test_flag == 1:
    test_label_cls = (test_label>PROFIT_MARGIN) + 0.0
    #testPredictor(clf_test_month, test_data, test_label_cls[:,2], model_dir_name)
    #visualizeGbdtIterError(clf_test_month, model_dir_name, train_data,\
    #                       train_label_cls[:,2], \
    #                       test_data, test_label_cls[:,2])
    ########################
    testPredictor(clf_lin_reg_test_month, test_data, test_label_cls[:,2], model_dir_month_lin_reg)
    testPredictor(clf_lin_reg_test_month, test_data, test_label_cls[:,1], model_dir_week_lin_reg)
    testPredictor(clf_lin_reg_test_month, test_data, test_label_cls[:,0], model_dir_day_lin_reg)


    #testPredictor(clf_log_reg_test_month, test_data, test_label_cls[:,2], model_dir_name_log_reg)
    #testGbdtClassifier(clf_reg_test_month, test_data, test_label_cls[:,2], model_dir_name_reg)
    #visualizeGbdtIterError(clf_reg_test_month, model_dir_name_reg, train_data, \
    #                       train_label_cls[:,2], \
    #                       test_data, test_label_cls[:,2])


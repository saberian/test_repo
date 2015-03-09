import numpy as np
import os
import pickle
import stockRecommendationLib as srl
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import datetime
import sys

def visualizeGbdtIterError(clf, clf_name, train_data, train_label, \
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
    plt.savefig("../models/"+ model_dir_name + "/iter_evolution.png")


def testGbdtClassifier(clf, test_data, test_label, name):
    n_test_samples = test_data.shape[0]
    rg_test =  sum(test_label==0) / float(n_test_samples)
    model_dir =  "../models/" + name
    f_out=open(model_dir + "/stats.txt", "a")
    log_str = "num test sample: " + str(n_test_samples)
    print log_str
    f_out.write(log_str+"\n")
    log_str = "random guessing acc: " + str(max(rg_test, 1-rg_test))
    print log_str
    f_out.write(log_str+"\n")
    clf_test = clf.score(test_data, test_label)
    log_str = "clf test accuracy: " + str(clf_test)
    print log_str
    f_out.write(log_str+"\n")
    f_out.close()

def trainGbdtClassifier(train_data, train_label, name):
    print "============================"
    print "training classifier " + name
    model_dir =  "../models/" + name
    if not os.path.exists(model_dir):
        os.system("mkdir " + model_dir)
    f_out=open(model_dir + "/stats.txt", "a")
    # get performance of random guessing
    n_training_samples = train_data.shape[0]
    rg_train = sum(train_label==0) / float(n_training_samples)
    log_str = "num training sample: " + str(n_training_samples)
    print log_str
    f_out.write(log_str+"\n")
    log_str = "random guessing acc: " + str(max(rg_train, 1-rg_train))
    print log_str
    f_out.write(log_str+"\n")

    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, 
        max_depth=3, random_state=0).fit(train_data, train_label)

    clf_train =  clf.score(train_data, train_label)
    log_str = "clf training accuracy: " + str(clf_train)
    print log_str
    f_out.write(log_str+"\n")
    f_out.close()
    pickle.dump(clf, open(model_dir + "/model.pkl", "wb"))
    srl.writeLogSummary("classifier " + name + " is trained")
    return clf

print "++++++++++++++++++++++++++++++++++++++++++"
print "training classifiers"
today = sys.argv[1]
train_data, train_label, \
            test_data, test_label, \
            extra_train_data, extra_train_label =\
            pickle.load(open("../data/processedData/train_data_" + today + ".pkl", "rb"))


### convert ratio's into labels
PROFIT_MARGIN = 1.05;


train_label = (train_label>PROFIT_MARGIN) + 0.0
test_label = (test_label>PROFIT_MARGIN) + 0.0
extra_train_label = (extra_train_label>PROFIT_MARGIN) + 0.0



model_dir_name = "gbdt_month_test_" + today
clf_test_month = trainGbdtClassifier(train_data, train_label[:,2], model_dir_name)
testGbdtClassifier(clf_test_month, test_data, test_label[:,2], model_dir_name)
visualizeGbdtIterError(clf_test_month, model_dir_name, train_data, train_label[:,2], \
                        test_data, test_label[:,2])





full_train_data = np.vstack((train_data, extra_train_data))
full_train_label = np.vstack((train_label, extra_train_label))
model_dir_name = "gbdt_month_full_" + today
clf_full_month = trainGbdtClassifier(full_train_data, full_train_label[:,2], model_dir_name)



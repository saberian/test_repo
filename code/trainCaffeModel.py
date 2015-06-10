import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import shutil
import tempfile
import sklearn
import sklearn.datasets
import sklearn.linear_model
import pickle
from constants import *
import stockRecommendationLib as srl

caffe_root = '../../caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

def getCaffeInfoLog(file_name, phr, ind):
    lines = open(file_name).read().splitlines()
    res = []
    for l in lines:
        if phr in l:
            ws=l.split(" ")
            res.append(float(ws[-ind]))
    return res

def writeCaffeSolver(prd_name, prt_name, cf_model_dir, n_test_itr, n_max_itr, base_lr):
    n_itr_snap = int(n_max_itr/4)
    f = open(protobuf_loc  + '/solver.prototxt', 'w')
    f.write("net: \"%s\"\n" % prt_name)
    f.write("test_iter: %d \n" % n_test_itr)
    f.write("test_interval: %d \n" % n_itr_snap)
    f.write("base_lr: %f \n" % base_lr)
    f.write("lr_policy: \"step\""+"\n")
    f.write("gamma: 0.1"+"\n")
    f.write("stepsize: % d \n" % n_itr_snap)
    f.write("display:  %d \n" %  100) #int(n_itr_snap/20)
    f.write("max_iter: %d \n" % n_max_itr)
    f.write("momentum: 0.9"+"\n")
    f.write("weight_decay: 0.0005"+"\n")
    f.write("snapshot: %d \n" % n_itr_snap)
    f.write("snapshot_prefix: \"" + cf_model_dir + "/" + prd_name +"\"\n")
    f.write("solver_mode: GPU"+"\n")


############### input parameters #################
caffe_loc = caffe_root + '/.build_release/tools/'
data_name = 'dataset_eval_2015-05-04'
dim_no = 1

'''prd_name = "caffe_clf_v3_week"
prd_type = "clf"
protobuf_loc = 'caffe_clf_protobuf'
base_lr = 0.01'''


prd_name = "caffe_reg_v4_week"
prd_type = "reg"
protobuf_loc = 'caffe_reg_protobuf'
base_lr = 0.01

prt_name = protobuf_loc + "/train_val4.prototxt"
deploy_file = protobuf_loc + "/deploy4.prototxt"
output_node = "fc1"
batch_size = 1000
n_max_itr = 100000
n_batch_test = 100 # not very important



cf_model_dir = MODELS_DIR + "/" + prd_name
if not os.path.isdir(MODELS_DIR + "/" + prd_name):
    os.mkdir(MODELS_DIR + "/" + prd_name)

print "writting solver file"
writeCaffeSolver(prd_name, prt_name, cf_model_dir, n_batch_test, n_max_itr, base_lr)

print "training deep network"
cmd = caffe_loc + "/caffe.bin train --solver=" + protobuf_loc + "/solver.prototxt"
os.system(cmd  + " 2>&1 | tee " + cf_model_dir + "/caffe.log")

print "testing the model"
model_file_name = prd_name + "_iter_"+  str(n_max_itr) + ".caffemodel"

caffe_res_test, test_label = srl.testHdf5Caffe(deploy_file, output_node, cf_model_dir, \
    model_file_name, "test_" + prd_type + ".txt", srl.HDF5_LOC)

caffe_res_train, train_label = srl.testHdf5Caffe(deploy_file, output_node, cf_model_dir, \
    model_file_name, "train_" + prd_type + ".txt", srl.HDF5_LOC)


if prd_type == "clf":
    caffe_res_test = caffe_res_test[:,1]
    caffe_res_train = caffe_res_train[:,1]
if prd_type == "reg":
    test_label = (test_label > (PROFIT_MARGIN-1)) + 0
    train_label = (train_label > (PROFIT_MARGIN-1)) + 0


print "============="
print caffe_res_test.shape
print caffe_res_train.shape

print test_label.shape
print train_label.shape

plt.clf()

fpr, tpr, thrs = sklearn.metrics.roc_curve(test_label, caffe_res_test)
prd_auc = sklearn.metrics.roc_auc_score(test_label, caffe_res_test)
plt.plot(fpr, tpr, label=("caffe test %.2f" % (prd_auc)))

fpr, tpr, thrs = sklearn.metrics.roc_curve(train_label, caffe_res_train)
prd_auc = sklearn.metrics.roc_auc_score(train_label, caffe_res_train)
plt.plot(fpr, tpr, label=("caffe train %.2f" % (prd_auc)))

plt.grid()
plt.draw()
plt.legend(loc='lower right')
plt.savefig(cf_model_dir + "/roc.png")

plt.clf()
plt.plot(getCaffeInfoLog(cf_model_dir + "/caffe.log", ", loss = ", 1))
plt.grid()
plt.draw()
plt.savefig(cf_model_dir + "/train_loss.png") 


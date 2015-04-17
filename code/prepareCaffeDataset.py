import numpy as np
import matplotlib.pyplot as plt

caffe_root = '/mnt/ehsan_files/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe

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


def writeCaffeSolver(clf_name, prt_name, cf_model_dir, n_test_itr, n_max_itr):
    n_itr_snap = int(n_max_itr/5)
    f = open('solver.prototxt', 'w')
    f.write("net: \"%s\"\n" % prt_name)  
    f.write("test_iter: %d \n" % n_test_itr)
    f.write("test_interval: %d \n" % n_itr_snap)
    f.write("base_lr: 0.01"+"\n")
    f.write("lr_policy: \"step\""+"\n")
    f.write("gamma: 0.1"+"\n")
    f.write("stepsize: % d \n" % n_itr_snap)
    f.write("display:  %d \n" % int(n_itr_snap/20))
    f.write("max_iter: %d \n" % n_max_itr)
    f.write("momentum: 0.9"+"\n")
    f.write("weight_decay: 0.0005"+"\n")
    f.write("snapshot: %d \n" % n_itr_snap)
    f.write("snapshot_prefix: \"" + cf_model_dir + "/" + clf_name +"\"\n")
    f.write("solver_mode: GPU"+"\n")

def writeHDF5data(X, y, Xt, yt):

    ns = X.shape[0]
    nf = X.shape[1]
    X = np.reshape(X, (ns, 1, nf, 1))
    nt = Xt.shape[0]
    Xt = np.reshape(Xt, (nt, 1, nf, 1))

    dirname = os.path.abspath('../data/hdf5/')
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    train_filename = os.path.join(dirname, 'train.h5')
    test_filename = os.path.join(dirname, 'test.h5') 
    # HDF5DataLayer source should be a file containing a list of HDF5 filenames.
    # To show this off, we'll list the same data file twice.
    with h5py.File(train_filename, 'w') as f:
        f['data'] = X
        f['label'] = y.astype(np.float32)
    with open(os.path.join(dirname, 'train.txt'), 'w') as f:
        f.write(train_filename + '\n')
        #f.write(train_filename + '\n')    
    # HDF5 is pretty efficient, but can be further compressed.
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
    with h5py.File(test_filename, 'w') as f:
        f.create_dataset('data', data=Xt, **comp_kwargs)
        f.create_dataset('label', data=yt.astype(np.float32), **comp_kwargs)
    with open(os.path.join(dirname, 'test.txt'), 'w') as f:
        f.write(test_filename + '\n')

def prepareData(data_name, dim_no, batch_size, cf_model_dir):
    train_data, train_label, test_data, test_label =\
            pickle.load(open(PROCESSED_DATA_DIR + "/" + data_name + ".pkl", "rb"))
    X = train_data
    y = train_label[:, dim_no] - 1
    y = y * (y < 4) + 4 * (y > 4)
    y = y
    Xt = test_data
    yt = test_label[:, dim_no] - 1
    yt = yt * (yt < 4) + 4 * (yt > 4)
    yt = yt

    n_batch_train = int(len(y) / batch_size)
    n_used_samples_train = n_batch_train * batch_size
    X = X[0:n_used_samples_train,:]
    y = y[0:n_used_samples_train]

    n_batch_test = int(len(yt) / batch_size)
    n_used_samples_test = n_batch_test * batch_size
    Xt = Xt[0:n_used_samples_test,:]
    yt = yt[0:n_used_samples_test]
    x_mean = np.mean(X, axis=0)
    pickle.dump(x_mean, open(cf_model_dir + "/x_mean.pkl", "w"))
    X -= x_mean[np.newaxis,:]
    Xt -= x_mean[np.newaxis,:]
    return X, y, Xt, yt

def learn_and_test_LR(X, y, Xt, yt):
    clf = sklearn.linear_model.LinearRegression()
    clf.fit(X, y)
    yt_pred = clf.predict(Xt)
    return yt_pred


caffe_loc = '/mnt/ehsan_files/caffe/build/tools'
data_name = 'dataset_eval_2015-04-01'
dim_no = 2
clf_name = "caffe_v1_month"
prt_name = "train_val3.prototxt"
deploy_file = "deploy3.prototxt"
output_node = "fc2"
batch_size = 100
n_max_itr = 1000
cf_model_dir = MODELS_DIR + "/" + clf_name

if not os.path.isdir(MODELS_DIR + "/" + clf_name):
    os.mkdir(MODELS_DIR + "/" + clf_name)

print "loading and normalizaing data"
X, y, Xt, yt = prepareData(data_name, dim_no, batch_size, cf_model_dir)
print "writing hdf5 files"
writeHDF5data(X, y, Xt, yt)
n_batch_test = len(yt) / batch_size
print "writting solver file"
writeCaffeSolver(clf_name, prt_name, cf_model_dir, n_batch_test, n_max_itr)
print "training deep network"
cmd = caffe_loc + "/caffe train --solver=solver.prototxt"
os.system(cmd  + " 2>&1 | tee " + cf_model_dir + "/caffe.log")

model_file_name = clf_name + "_iter_"+  str(n_max_itr) + ".caffemodel"
caffe_res = srl.test_caffe(deploy_file, output_node, cf_model_dir, model_file_name, Xt, 0)
lr_res = learn_and_test_LR(X, y, Xt, yt)

plt.clf()
test_label = (yt > (PROFIT_MARGIN-1)) + 0

fpr, tpr, thrs = sklearn.metrics.roc_curve(test_label, caffe_res)
clf_auc = sklearn.metrics.roc_auc_score(test_label, caffe_res)
clf_mse = sklearn.metrics.mean_squared_error(yt, caffe_res) / 2.0
plt.plot(fpr, tpr, label=("caffe %.2f, %.8f" % (clf_auc, clf_mse)))

fpr, tpr, thrs = sklearn.metrics.roc_curve(test_label, lr_res)
clf_auc = sklearn.metrics.roc_auc_score(test_label, lr_res)
clf_mse = sklearn.metrics.mean_squared_error(yt, lr_res) / 2.0
plt.plot(fpr, tpr, label=("linear reg %.2f, %.8f" % (clf_auc, clf_mse)))
plt.grid()
plt.draw()
plt.legend(loc='lower right')
plt.savefig(cf_model_dir + "/roc.png")


'''
print('Accuracy Caffe res: {:.8f}'.format(sklearn.metrics.mean_squared_error(yt, caffe_res) / 2.0))
print('Accuracy LR  : {:.8f}'.format(sklearn.metrics.mean_squared_error(yt, yt_pred) / 2.0))
print('Accuracy Def : {:.8f}'.format(sklearn.metrics.mean_squared_error(yt, np.mean(y) + (0 * yt_pred)) / 2.0 ))
print('Accuracy Zero: {:.8f}'.format(sklearn.metrics.mean_squared_error(yt, (0 * yt_pred)) / 2.0 ))
print('Accuracy Rand: {:.8f}'.format(sklearn.metrics.mean_squared_error(yt, np.random.rand(yt.shape[0], )/ 100.0) / 2.0 ))
print('Accuracy NRnd: {:.8f}'.format(sklearn.metrics.mean_squared_error(yt, np.random.normal(0, 1, yt.shape[0])) / 2.0 ))
'''

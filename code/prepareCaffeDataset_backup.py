import numpy as np
import matplotlib.pyplot as plt

caffe_root = '/mnt/ehsan_files/caffe/'  # this file is expected to be in {caffe_root}/examples
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
from constants import *
import pickle

def writeCaffeSolver(clf_name, prt_name, n_test_itr, n_max_itr):
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
    f.write("snapshot_prefix: \"" + MODELS_DIR + "/" + clf_name + "/" + clf_name +"\"\n")
    f.write("solver_mode: GPU"+"\n")

def writeHDF5data(X, y, Xt, yt):


    print X.shape
    ns = X.shape[0]
    nf = X.shape[1]

    X = np.reshape(X, (ns, 1, nf, 1))
    print X.shape
    print Xt.shape
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

def prepareData(data_name, dim_no, batch_size):
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
    X -= x_mean[np.newaxis,:]
    Xt -= x_mean[np.newaxis,:]
    return X, y, Xt, yt


def learn_and_test_CF(solver_file, output_node, X, y, Xt, yt):
    caffe.set_mode_gpu()
    solver = caffe.get_solver(solver_file)
    solver.solve()

    accuracy = 0
    test_iters = int(len(Xt) / solver.test_nets[0].blobs['data'].num)
    test_res = np.array([])
    for i in range(test_iters):
        solver.test_nets[0].forward()
        accuracy += solver.test_nets[0].blobs['accuracy'].data
        if test_res.shape[0] == 0:
            test_res = solver.test_nets[0].blobs[output_node].data.copy()
        else:
            test_res = np.vstack((test_res, solver.test_nets[0].blobs[output_node].data.copy()))

    test_res = test_res.squeeze()
    return test_res
    #return accuracy

def test_caffe(deploy_file, output_node, model_file_name, test_data):

    caffe.set_mode_gpu()
    net = caffe.Classifier(deploy_file, model_file_name)

    n_samples = test_data.shape[0]
    n_features = test_data.shape[1]
    res = np.array([])
    
    caffe_in = np.zeros((n_samples, 1, n_features, 1))
    for i in xrange(0, n_samples):
        caffe_in[i] = test_data[i,:].reshape((1, n_features, 1))
    res = net.forward_all(**{"data": caffe_in})[output_node].squeeze()

    '''batch_size = 5
    for k in xrange(0, n_samples/batch_size):
        temp_ind = k * batch_size
        caffe_in = np.zeros((batch_size, 1, n_features, 1))
        for i in xrange(0, batch_size):
            temp_i = temp_ind + i
            caffe_in[i] = test_data[temp_i,:].reshape((1, n_features, 1))
        print caffe_in.shape
        out = net.forward_all(**{"data": caffe_in})
        print "forwareded the net"
        if res.shape[0] == 0:
            res = out["fc1"]
        else:
            res = np.vstack((res, out["fc1"]))'''

    #print [(k, v.data.shape) for k, v in net.blobs.items()]

    return res

def learn_and_test_LR(X, y, Xt, yt):
    clf = sklearn.linear_model.LinearRegression()
    clf.fit(X, y)
    yt_pred = clf.predict(Xt)
    print('Accuracy LR  : {:.8f}'.format(sklearn.metrics.mean_squared_error(yt, yt_pred) / 2.0))
    print('Accuracy Def : {:.8f}'.format(sklearn.metrics.mean_squared_error(yt, np.mean(y) + (0 * yt_pred)) / 2.0 ))
    print('Accuracy Zero: {:.8f}'.format(sklearn.metrics.mean_squared_error(yt, (0 * yt_pred)) / 2.0 ))
    print('Accuracy Rand: {:.8f}'.format(sklearn.metrics.mean_squared_error(yt, np.random.rand(yt.shape[0], )/ 100.0) / 2.0 ))
    print('Accuracy NRnd: {:.8f}'.format(sklearn.metrics.mean_squared_error(yt, np.random.normal(0, 1, yt.shape[0])) / 2.0 ))






data_name = 'dataset_eval_2015-04-01'
clf_name = "caffe_v1_month"
prt_name = "train_val3.prototxt"
deploy_file = "deploy3.prototxt"
output_node = "fc2"
dim_no = 2
batch_size = 100
n_max_itr = 1000

 
if not os.path.isdir(MODELS_DIR + "/" + clf_name):
    os.mkdir(MODELS_DIR + "/" + clf_name)

print "loading and normalizaing data"
X, y, Xt, yt = prepareData(data_name, dim_no, batch_size)
print "writing hdf5 files"
writeHDF5data(X, y, Xt, yt)
n_batch_test = len(yt) / batch_size
print "writtting solver file"
writeCaffeSolver(clf_name, prt_name, n_batch_test, n_max_itr)
print "training deep network"
test_res_1 = learn_and_test_CF('solver.prototxt', output_node, X, y, Xt, yt)
print "testing the network"
model_file_name = MODELS_DIR + "/" + clf_name + "/" + clf_name + "_"+  str(n_max_itr) + ".caffemodel"
test_res_2 = test_caffe(deploy_file, output_node, model_file_name, Xt)
print('Accuracy Caffe_final res2: {:.8f}'.format(sklearn.metrics.mean_squared_error(yt, test_res_2) / 2.0))
print('Accuracy Caffe_final res1: {:.8f}'.format(sklearn.metrics.mean_squared_error(yt, test_res_1) / 2.0))
print "testing linear regression"
learn_and_test_LR(X, y, Xt, yt)

print np.linalg.norm(test_res_1 - test_res_2)


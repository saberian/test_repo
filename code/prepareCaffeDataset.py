import numpy as np
import matplotlib.pyplot as plt

# Make sure that caffe is on the python path:
caffe_root = '/mnt/ehsan_files/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

import os
import h5py
import shutil
import tempfile

# You may need to 'pip install scikit-learn'
import sklearn
import sklearn.datasets
import sklearn.linear_model

from constants import *
import pickle

def writeCaffeSolver(clf_name):
    f = open('solver.prototxt', 'w')
    f.write("net: \"train_val2.prototxt\""+"\n")
    f.write("test_iter: 120"+"\n")
    f.write("test_interval: 1000"+"\n")
    f.write("base_lr: 0.001"+"\n")
    f.write("lr_policy: \"step\""+"\n")
    f.write("gamma: 0.1"+"\n")
    f.write("stepsize: 1000"+"\n")
    f.write("display: 100"+"\n")
    f.write("max_iter: 10000"+"\n")
    f.write("momentum: 0.9"+"\n")
    f.write("weight_decay: 0.0005"+"\n")
    f.write("snapshot: 1000"+"\n")
    f.write("snapshot_prefix: \"../models/" + clf_name + "/\""+"\n")
    f.write("solver_mode: GPU"+"\n")

def prepareHdf5Data(data_name, dim_no):
    train_data, train_label, test_data, test_label =\
            pickle.load(open(PROCESSED_DATA_DIR + "/" + data_name + ".pkl", "rb"))
    X = train_data
    y = train_label[:, dim_no] - 1
    y = y * (y < 4) + 4 * (y > 4)
    y = y / 4.0
    Xt = test_data
    yt = test_label[:, dim_no] - 1
    yt = yt * (yt < 4) + 4 * (yt > 4)
    yt = yt / 4.0

    x_mean = np.mean(X, axis=0)
    X -= x_mean[np.newaxis,:]
    Xt -= x_mean[np.newaxis,:]

    '''dirname = os.path.abspath('../data/hdf5/')
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
        f.write(test_filename + '\n')'''

    return X, y, Xt, yt


def learn_and_test_CF(solver_file, X, y, Xt, yt):
    caffe.set_mode_cpu()
    solver = caffe.get_solver(solver_file)
    solver.solve()

    accuracy = 0
    test_iters = int(len(Xt) / solver.test_nets[0].blobs['data'].num)
    test_res = []
    for i in range(test_iters):
        solver.test_nets[0].forward()
        accuracy += solver.test_nets[0].blobs['accuracy'].data
        test_res.append(solver.test_nets[0].blobs['fc2'].data)

    print len(test_res)
    print Xt.shape
    accuracy /= test_iters
    print("Accuracy CAFFE: {:.8f}".format(accuracy))
    #return accuracy

def learn_and_test_LR(X, y, Xt, yt):
    clf = sklearn.linear_model.LinearRegression()
    clf.fit(X, y)
    yt_pred = clf.predict(Xt)
    print('Accuracy LR: {:.8f}'.format(sklearn.metrics.mean_squared_error(yt, yt_pred)))
    print('Accuracy Def: {:.8f}'.format(sklearn.metrics.mean_squared_error(yt, np.mean(y) + (0 * yt_pred))))
    print('Accuracy Zero: {:.8f}'.format(sklearn.metrics.mean_squared_error(yt, (0 * yt_pred))))

data_name = 'dataset_eval_2015-04-01'
clf_name = "caffe_v1_month"
 
if not os.path.isdir("../models/" + clf_name):
    os.mkdir("../models/" + clf_name)

writeCaffeSolver(clf_name)

X, y, Xt, yt = prepareHdf5Data(data_name, 2)

learn_and_test_CF('solver.prototxt', X, y, Xt, yt)
learn_and_test_LR(X, y, Xt, yt)

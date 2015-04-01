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


def learn_and_test(solver_file):
    caffe.set_mode_cpu()
    solver = caffe.get_solver(solver_file)
    solver.solve()

    accuracy = 0
    test_iters = int(len(Xt) / solver.test_nets[0].blobs['data'].num)
    for i in range(test_iters):
        solver.test_nets[0].forward()
        accuracy += solver.test_nets[0].blobs['accuracy'].data
    accuracy /= test_iters
    return accuracy


data_name = 'dataset_eval_2015-03-26'
train_data, train_label, test_data, test_label =\
        pickle.load(open(PROCESSED_DATA_DIR + "/" + data_name + ".pkl", "rb"))
X = train_data
y = train_label[:,2] - 1
y = y * (y < 4) + 4 * (y > 4)
y = y / 4.0
Xt = test_data
yt = test_label[:,2] - 1
yt = yt * (yt < 4) + 4 * (yt > 4)
yt = yt / 4.0

'''X, y = sklearn.datasets.make_classification(
    n_samples=10000, n_features=4, n_redundant=0, n_informative=2, 
    n_clusters_per_class=2, hypercube=False, random_state=0
)

y = y + 0.5

# Split into train and test
X, Xt, y, yt = sklearn.cross_validation.train_test_split(X, y)
'''


print X.shape
print y.shape
print [max(y), min(y)]
print np.max(X)
print Xt.shape
print yt.shape
print [max(yt), min(yt)]

y_mean = np.mean(y)
x_mean = np.mean(X, axis=0)
print x_mean.shape

X -= x_mean[np.newaxis,:]
print X.shape
Xt -= x_mean[np.newaxis,:]



clf = sklearn.linear_model.LinearRegression()
clf.fit(X, y)
yt_pred = clf.predict(Xt)
print('Accuracy: {:.8f}'.format(sklearn.metrics.mean_squared_error(yt, yt_pred)))
print('Accuracy D: {:.8f}'.format(sklearn.metrics.mean_squared_error(yt, y_mean + (0 * yt_pred))))
print('Accuracy 0: {:.8f}'.format(sklearn.metrics.mean_squared_error(yt, (0 * yt_pred))))


# Write out the data to HDF5 files in a temp directory.
# This file is assumed to be caffe_root/examples/hdf5_classification.ipynb
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
    f.write(train_filename + '\n')
    
# HDF5 is pretty efficient, but can be further compressed.
comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
with h5py.File(test_filename, 'w') as f:
    f.create_dataset('data', data=Xt, **comp_kwargs)
    f.create_dataset('label', data=yt.astype(np.float32), **comp_kwargs)
with open(os.path.join(dirname, 'test.txt'), 'w') as f:
    f.write(test_filename + '\n')


#acc = learn_and_test('solver.prototxt')
#print("Accuracy: {:.3f}".format(acc))

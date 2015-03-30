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




X, y = sklearn.datasets.make_classification(
    n_samples=10000, n_features=4, n_redundant=0, n_informative=2, 
    n_clusters_per_class=2, hypercube=False, random_state=0
)

y = y + 0.5

# Split into train and test
X, Xt, y, yt = sklearn.cross_validation.train_test_split(X, y)

# Train and test the scikit-learn SGD logistic regression.
#clf = sklearn.linear_model.SGDClassifier(
#    loss='log', n_iter=1000, penalty='l2', alpha=1e-3, class_weight='auto')
clf = sklearn.linear_model.LinearRegression()

clf.fit(X, y)
yt_pred = clf.predict(Xt)
#print('Accuracy: {:.3f}'.format(sklearn.metrics.accuracy_score(yt, yt_pred)))

print('Accuracy: {:.3f}'.format(sklearn.metrics.mean_squared_error(yt, yt_pred)))



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


acc = learn_and_test('hdf5_classification/solver.prototxt')
print("Accuracy: {:.3f}".format(acc))

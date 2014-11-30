import numpy as np
import scipy as sp
from sklearn import svm
from sklearn.decomposition import PCA

# Read data sets
trainData = np.genfromtxt('data/train.csv', delimiter = ',')
trainLables = np.genfromtxt('data/trainLabels.csv', delimiter = ',')
testData = np.genfromtxt('data/test.csv', delimiter = ',')

# PCA: dimensionality reduction
pca = PCA(n_components = 30)
pcaTrainData = pca.fit_transform(trainData)
pcaTestData = pca.transform(testData)

# SVC RBF kernel
clf = svm.SVC()
clf.fit(pcaTrainData, trainLables)
prediction = clf.predict(pcaTestData)

indexes = range(1, len(prediction) + 1, 1)
result = np.column_stack((indexes, prediction))

# write to file
np.savetxt('predictions/pred_2_pca_svmRBF.csv', result.astype(int), fmt = '%s', delimiter = ',', header = 'Id,Solution')
import numpy as np
import scipy as sp
from sklearn import svm

# Read data sets
trainData = np.genfromtxt('data/train.csv', delimiter = ',')
trainLables = np.genfromtxt('data/trainLabels.csv', delimiter = ',')
testData = np.genfromtxt('data/test.csv', delimiter = ',')

# SVC RBF kernel
clf = svm.SVC()
clf.fit(trainData, trainLables)
prediction = clf.predict(testData)

indexes = range(1, len(prediction) + 1, 1)
result = np.column_stack((indexes, prediction))

# write to file
np.savetxt('predictions/pred_1.csv', result.astype(int), fmt = '%s', delimiter = ',', header = 'Id,Solution')
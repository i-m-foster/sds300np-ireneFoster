from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.datasets import load_unit_test
import pandas as pd
import numpy as np

dfWide = pd.read_csv("activityData3.csv", sep=",")
dfWide = pd.read_csv("wideActivityData2.csv", sep=",")
results6 = pd.read_csv("hyperaktiv/patient_info.csv", sep=";")
smallResults = results6.merge(dfWide, on='ID', how='right')
smallResults = smallResults.sort_values('ID')
smallResults = np.ravel(smallResults[["ADHD"]])


sampleData = pd.read_csv("sampleData.csv", sep=",")

sampleData['time'] =  pd.to_datetime(sampleData['time'], format='%m-%d-%Y %H:%M')
sampleData['time'] = pd.to_datetime(sampleData['time']).dt.strftime("%H"+":"+"%M")
sampleData['time'] =  pd.to_datetime(sampleData['time'], format='%H:%M')
sampleData = sampleData.drop('Unnamed: 0', axis=1)
sampleData = sampleData.sort_values('ID')

# Turn data in 3D array
sample2 = np.zeros((85,50,1))

for y in range(85):
    for i in range(50):
        sample2[y,i,0] = sampleData.loc[y*50+i,'activity']

#print(sktime.datatypes.check_raise(sample2, "numpy.ndarray"))

from sklearn.model_selection import train_test_split
from sktime.forecasting.model_selection import temporal_train_test_split
X_train, X_test,y_train, y_test = train_test_split(sample2, smallResults, test_size=0.2, train_size=0.8, shuffle = False)


# # Individual TDE algorithm
# from sktime.classification.dictionary_based import IndividualTDE
# from sktime.datasets import load_unit_test
# #X_train, y_train = load_unit_test(split="train", return_X_y=True)
# #X_test, y_test = load_unit_test(split="test", return_X_y=True) 
# clf = IndividualTDE() 
# clf.fit(X_train, y_train) 
# IndividualTDE(...)
# print(clf.score(X_test, y_test))
# y_pred = clf.predict(X_test) 
# print(y_pred)
# print(y_test)
# # about 64% acuracy on sample data
# print("IndividualTDE")
# print(clf.score(X_test, y_test))


# # KNeighborsTimeSeriesClassifier
# from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
# from sktime.datasets import load_unit_test
# #X_train, y_train = load_unit_test(return_X_y=True, split="train")
# #X_test, y_test = load_unit_test(return_X_y=True, split="test")
# classifier = KNeighborsTimeSeriesClassifier(distance="euclidean")
# classifier.fit(X_train, y_train)
# KNeighborsTimeSeriesClassifier(...)
# y_pred = classifier.predict(X_test)
# print(y_pred)
# print(classifier.score(X_test, y_test))

# Dummy classifier
# from sktime.classification.dummy import DummyClassifier
# from sktime.datasets import load_unit_test
# #X_train, y_train = load_unit_test(split="train", return_X_y=True)
# #X_test, y_test = load_unit_test(split="test", return_X_y=True) 
# clf = DummyClassifier() 
# clf.fit(X_train, y_train) 
# DummyClassifier(...)
# y_pred = clf.predict(X_test) 
# print(y_pred)
# print(clf.score(X_test, y_test))

# # time series svc
# from sktime.classification.kernel_based import TimeSeriesSVC
# from sklearn.gaussian_process.kernels import RBF
# from sktime.dists_kernels import AggrDist
# from sktime.datasets import load_unit_test
# #X_train, y_train = load_unit_test(return_X_y=True, split="train")
# #X_test, y_test = load_unit_test(return_X_y=True, split="test")
# mean_gaussian_tskernel = AggrDist(RBF())
# classifier = TimeSeriesSVC(kernel=mean_gaussian_tskernel)
# classifier.fit(X_train, y_train)
# TimeSeriesSVC(...)
# y_pred = classifier.predict(X_test)
# print(y_pred)
# print(classifier.score(X_test, y_test))

# # arsenal
# from sktime.classification.kernel_based import Arsenal
# from sktime.datasets import load_unit_test
# #X_train, y_train = load_unit_test(split="train", return_X_y=True)
# #X_test, y_test =load_unit_test(split="test", return_X_y=True) 
# clf = Arsenal(num_kernels=100, n_estimators=5) 
# clf.fit(X_train, y_train) 
# Arsenal(...)
# y_pred = clf.predict(X_test)
# print(y_pred)
# print(clf.score(X_test, y_test))

# # CNN classifier
# from sktime.classification.deep_learning.cnn import CNNClassifier
# from sktime.datasets import load_unit_test
# X_train, y_train = load_unit_test(split="train")
# X_test, y_test = load_unit_test(split="test")
# cnn = CNNClassifier(n_epochs=20,batch_size=4)  
# cnn.fit(X_train, y_train)  
# CNNClassifier(...)
# y_pred = cnn.predict(X_test)
# print(y_pred)
# print(cnn.score(X_test, y_test))

# # fcn
# from sktime.classification.deep_learning.fcn import FCNClassifier
# from sktime.datasets import load_unit_test
# X_train, y_train = load_unit_test(split="train", return_X_y=True)
# X_test, y_test = load_unit_test(split="test", return_X_y=True)
# fcn = FCNClassifier(n_epochs=20,batch_size=4)  
# fcn.fit(X_train, y_train)  
# FCNClassifier(...)
# y_pred = fcn.predict(X_test)
# print(y_pred)
# print(fcn.score(X_test, y_test))

# # lstmfcn
# from sktime.classification.deep_learning import LSTMFCNClassifier
# from sktime.datasets import load_unit_test
# X_train, y_train = load_unit_test(split="train", return_X_y=True)
# X_test, y_test = load_unit_test(split="test", return_X_y=True)
# clf = LSTMFCNClassifier(n_epochs=20,batch_size=4)  
# clf.fit(X_train, y_train)  
# LSTMFCNClassifier(...)
# y_pred = clf.predict(X_test)
# print(y_pred)
# print(clf.score(X_test, y_test))

# tapnet
from sktime.classification.deep_learning.tapnet import TapNetClassifier
from sktime.datasets import load_unit_test
X_train, y_train = load_unit_test(split="train")
X_test, y_test = load_unit_test(split="test")
tapnet = TapNetClassifier(n_epochs=20,batch_size=4)  
tapnet.fit(X_train, y_train) 
TapNetClassifier(...)
y_pred = tapnet.predict(X_test)
print(y_pred)
print(tapnet.score(X_test, y_test))








# from pyts.datasets import load_basic_motions
# from pyts.multivariate.transformation import WEASELMUSE
# X_train, X_test, y_train, y_test = load_basic_motions(return_X_y=True)
# transformer = WEASELMUSE()
# X_new = transformer.fit_transform(X_train, y_train)
# X_new.shape

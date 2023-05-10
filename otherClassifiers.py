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


from tslearn.utils import to_time_series_dataset
X = to_time_series_dataset([[1, 2, 3, 4], [1, 2, 3], [2, 5, 6, 7, 8, 9]])
y = [0, 0, 1]

# from tslearn.neighbors import KNeighborsTimeSeriesClassifier
# knn = KNeighborsTimeSeriesClassifier(n_neighbors=2)
# knn.fit(X_train, y_train)
# print(knn)
# #print(y_pred)
# print(knn.score(X_test, y_test))

# from tslearn.svm import TimeSeriesSVC
# clf = TimeSeriesSVC(C=1.0, kernel="gak")
# clf.fit(X_train, y_train)
# print(clf.score(X_test, y_test))

# from tslearn.shapelets import LearningShapelets
# clf = LearningShapelets(n_shapelets_per_size={3: 1})
# clf.fit(X_train, y_train)
# print(clf.score(X_test, y_test))




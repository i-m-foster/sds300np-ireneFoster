import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline

from sktime.datasets import load_arrow_head  # univariate dataset
from sktime.datasets import load_basic_motions  # multivariate dataset
from sktime.transformations.panel.rocket import Rocket



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
# X_train, X_test = temporal_train_test_split(sample2 )
# y_train, y_test = temporal_train_test_split(smallResults )
#X_train, y_train = load_basic_motions(split="train", return_X_y=True)
#print(y_train)
#X_train.to_csv("ytrain.csv")

rocket = Rocket()  # by default, ROCKET uses 10,000 kernels
rocket.fit(X_train)
X_train_transform = rocket.transform(X_train)

classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
classifier.fit(X_train_transform, y_train)

#X_test, y_test = load_basic_motions(split="test", return_X_y=True)
X_test_transform = rocket.transform(X_test)

classifier.score(X_test_transform, y_test)

# 58% accuracy on samples
print(classifier.score(X_test_transform, y_test))
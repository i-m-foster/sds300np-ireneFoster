# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 22:02:08 2023

@author: ifoster25
"""

import os
import pandas as pd
import re
import numpy as np
directory = 'hyperaktiv/activity_data'


# Set sample data of 50 data points from every subject
# Empty df with correct row names
sampleData = pd.DataFrame(columns = ['time', 'activity', 'ID'])


# iterate over files in directory, 
for filename in os.listdir(directory):
    print(filename)
    f = os.path.join(directory, filename)
    # read in each csv
    df = pd.read_csv(f, sep=";")
    timeList = []
    activityList = []

    # get only number of id
    idNum = re.findall(r'\d+', filename)

    # if id is one digit get rid of zero
    if idNum[0][0] == "0":
        idNum[0] = idNum[0][1]
        print("done")
        print(idNum[0])

    # add each time/activity row to df
    f# or i in range(len(df)):
    for i in range(50):    
        # timeList.append(df.loc[i,'TIMESTAMP'])
        # activityList.append(df.loc[i,'ACTIVITY'])
        # add row to new df with id, and time/activity 
        sampleData.loc[len(sampleData.index)] = [df.loc[i,'TIMESTAMP'], df.loc[i,'ACTIVITY'], idNum[0]]
        print(sampleData.loc[i])
    #print(timeList)
    
sampleData.to_csv("sampleData.csv") 



# dfLong = pd.read_csv("longActivityData2.csv", sep=",")
# dfLong = dfLong.drop('Unnamed: 0', axis=1)
# dfLong['time'] =  pd.to_datetime(dfLong['time'], format='%m-%d-%Y %H:%M')
# print("Created at {:d}:{:02d}".format(dfLong.loc[1,'time'].hour, dfLong.loc[1,'time'].minute))
# dfLong['time'] = pd.to_datetime(dfLong['time']).dt.strftime("%H"+":"+"%M") 


# making multi-index df
# id = dfLong.loc[:,'ID']
# time = dfLong.loc[:,'time']
# #activity = dfLong.loc[:,'activity'].tolist()
# activity = dfLong.loc[:,'activity']

# result5 = []

# for i in range(len(id)):
# #for i in range(50):
#     print(i)
#     tuple = (time[i], activity[i], id[i])
#     # tuple = (id[i], activity[i])
#     # tuple2 = (time[i], tuple)
#     result5.append(tuple)

# index, activity, ID = zip(*result5)

# frame = pd.DataFrame({
#     'activity': activity,
#     'ID' : ID
# }, index=pd.DatetimeIndex(index))

# frame['activity'] = frame['activity'].astype(int)

# frame.sort_index()


dfWide = pd.read_csv("activityData3.csv", sep=",")
dfWide = pd.read_csv("wideActivityData2.csv", sep=",")
results6 = pd.read_csv("hyperaktiv/patient_info.csv", sep=";")
smallResults = results6.merge(dfWide, on='ID', how='right')
smallResults = smallResults[["ID", "ADHD"]]
smallResults = smallResults.sort_values('ID')


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
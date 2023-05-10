# import required module
import os
import pandas as pd
import re
import numpy as np
# assign directory
directory = 'hyperaktiv/activity_data'
 
df = pd.read_csv('hyperaktiv/activity_data/patient_activity_01.csv', sep=";")
#print('hyperaktiv/activity_data/patient_activity_01')
#print(df.to_string()) 

# Empty df with correct row names
df2 = pd.DataFrame(columns = ['ID', 'time', 'activity'])


# iterate over files in directory, 
for filename in os.listdir(directory):
    print(filename)
    f = os.path.join(directory, filename)
    # read in each csv
    df = pd.read_csv(f, sep=";")
    timeList = []
    activityList = []
    # add each time/activity row to list
    for i in range(len(df)):
        timeList.append(df.loc[i,'TIMESTAMP'])
        activityList.append(df.loc[i,'ACTIVITY'])
    #print(timeList)
    
    # get only number of id
    idNum = re.findall(r'\d+', filename)

    # if id is one digit get rid of zero
    if idNum[0][0] == "0":
        idNum[0] = idNum[0][1]
        print("done")
    #print(idNum[0])
    # add row to new df with id, and time/activity lists
    df2.loc[len(df2.index)] = [idNum[0], timeList, activityList]
    #df2.to_clipboard()
#print(df2.to_string())
print("length: " , len(df2['time']))

timeLength = []

lengthTime = 0
# find avg length of list
for i in range(len(df2)):
    timeLength.append(len(df2.loc[i,'time']))
    lengthTime = lengthTime + len(df2.loc[i,'time'])

import matplotlib.pyplot as plt
avgTime = lengthTime/len(df2)
print("total length:", lengthTime)
print("Avg length: ", avgTime)
print(len(timeLength))
print(np.std(timeLength))
print(np.max(timeLength))
print(np.min(timeLength))
print(np.mean(timeLength))
plt.plot(timeLength, 'ro')
#plt.axis([0, 6, 0, 20])
plt.show()


np.savetxt("timeLength.csv", timeLength, delimiter=",")
# df2.to_csv("wideActivityData2.csv")



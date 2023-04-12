# import required module
import os
import pandas as pd
import sktime
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
    for i in range(len(df)):
        timeList.append(df.loc[i,'TIMESTAMP'])
        activityList.append(df.loc[i,'ACTIVITY'])
    #print(timeList)
    df2.loc[len(df2.index)] = [filename[:len(filename)-4], timeList, activityList]
    df2.to_clipboard()
print(df2.to_string())
print("length: " , len(df2['time']))

lengthTime = 0
# find avg length of list
for i in range(len(df2)):
    lengthTime = lengthTime + len(df2.loc[i,'time'])

avgTime = lengthTime/len(df2)
print("total length:", lengthTime)
print("Avg length: ", avgTime)
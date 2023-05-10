# import required module
import os
import pandas as pd
import re
import numpy as np
# assign directory
directory = 'hyperaktiv/activity_data'
 
# df = pd.read_csv('hyperaktiv/activity_data/patient_activity_01.csv', sep=";")
# #print('hyperaktiv/activity_data/patient_activity_01')
# #print(df.to_string()) 

# # Empty df with correct row names
# df2 = pd.DataFrame(columns = ['ID', 'time', 'activity'])


# # iterate over files in directory, 
# for filename in os.listdir(directory):
#     print(filename)
#     f = os.path.join(directory, filename)
#     # read in each csv
#     df = pd.read_csv(f, sep=";")
#     timeList = []
#     activityList = []
#     # add each time/activity row to list
#     for i in range(len(df)):
#         timeList.append(df.loc[i,'TIMESTAMP'])
#         activityList.append(df.loc[i,'ACTIVITY'])
#     #print(timeList)
    
#     # get only number of id
#     idNum = re.findall(r'\d+', filename)

#     # if id is one digit get rid of zero
#     if idNum[0][0] == "0":
#         idNum[0] = idNum[0][1]
#         print("done")
#     #print(idNum[0])
#     # add row to new df with id, and time/activity lists
#     df2.loc[len(df2.index)] = [idNum[0], timeList, activityList]
#     #df2.to_clipboard()
# #print(df2.to_string())
# print("length: " , len(df2['time']))

# lengthTime = 0
# # find avg length of list
# for i in range(len(df2)):
#     lengthTime = lengthTime + len(df2.loc[i,'time'])

# avgTime = lengthTime/len(df2)
# print("total length:", lengthTime)
# print("Avg length: ", avgTime)


# df2.to_csv("wideActivityData2.csv")

# Empty df with correct row names
dfLong = pd.DataFrame(columns = ['time', 'activity', 'ID'])


# iterate over files in directory, 
for filename in os.listdir(directory):
    print(filename)
    f = os.path.join(directory, filename)
    # read in each csv
    df = pd.read_csv(f, sep=";")
    
    if len(df) >= 9978:
        
        # get only number of id
        idNum = re.findall(r'\d+', filename)
        
        # if id is one digit get rid of zero
        if idNum[0][0] == "0":
            idNum[0] = idNum[0][1]
            print("done")
            print(idNum[0])
        
        # add each time/activity row to df
        for i in range(9978):
            # add row to new df with id, and time/activity 
            dfLong.loc[len(dfLong)] = [df.loc[i,'TIMESTAMP'], df.loc[i,'ACTIVITY'], idNum[0]]
            print(dfLong.loc[i])
    else:  
        # get only number of id
        idNum = re.findall(r'\d+', filename)
        
        # if id is one digit get rid of zero
        if idNum[0][0] == "0":
            idNum[0] = idNum[0][1]
            print("done")
            print(idNum[0])
        
        # add each time/activity row to df
        for k in range(len(df)):
            # add row to new df with id, and time/activity 
            dfLong.loc[len(dfLong)] = [df.loc[k,'TIMESTAMP'], df.loc[k,'ACTIVITY'], idNum[0]]
            print(dfLong.loc[k])
            
        for j in range(9978 - len(df)):
            # add row to new df with id, and time/activity 
            dfLong.loc[len(dfLong)] = ['02-23-2009 23:59', 0, idNum[0]]
            print(dfLong.loc[j])    
    #print(timeList)
    
dfLong.to_csv("longActivityData2.csv")  

#dfWide2 = pd.read_csv("wideActivityData2.csv", sep=",")
# dfLong = pd.read_csv("longActivityData2.csv", sep=",")

# dfLong = dfLong.drop('Unnamed: 0', axis=1)

#dfWide3 = dfWide2.drop('Unnamed: 0', axis=1)

#dfWide4 = pd.MultiIndex.from_frame(dfWide3)
#pd.MultiIndex.from_frame(dfWide2)
#dfWide2.to_csv("activityData4.csv")
#print(dfWide2.to_string())

# id = ['a','b','c']
# prd = [0,1,2,]
# days = [2,5,4]

# result3 = []

# for i in range(len(id)):
#     print(i)
#     tuple = (id[i], prd[i], days[i])
#     result3.append(tuple)
    
# result4 = pd.MultiIndex.from_tuples(result3)    

# result = [(idx, i, p) for d, idx in zip(days, id) for i in range(1, d+1) for p in prd]

# result2 = pd.MultiIndex.from_tuples(result)

# print (pd.MultiIndex.from_tuples(result))

# id = dfLong.loc[:,'ID']
# time = dfLong.loc[:,'time']
# activity = dfLong.loc[:,'activity']

# result = [(idx, i, p) for d, idx in zip(activity, id) for i in range(1, d+1) for p in time]

# print (pd.MultiIndex.from_tuples(result))

# dfLong['time'] =  pd.to_datetime(dfLong['time'], format='%m-%d-%Y %H:%M')
# dfLong['date'] =  pd.to_datetime(dfLong['time'], format='%H:%M')

# id = dfLong.loc[:,'ID']
# time = dfLong.loc[:,'time']
# #activity = dfLong.loc[:,'activity'].tolist()
# activity = dfLong.loc[:,'activity']

# result3 = []

# for i in range(len(id)):
# #for i in range(50):
#     print(i)
#     tuple = (id[i], time[i], activity[i])
#     #tuple = (id[i], time[i])
#     result3.append(tuple)
    
# #result4.to_csv("activityData5.csv")    
    
# index = pd.MultiIndex.from_tuples(result3, names=["ID", "time", "activity"])

# s = pd.Series(activity, index=index)

# result4 = pd.MultiIndex.from_tuples(result3)

# print(result4[0][0])

# what = result4[0][1]


# dfLong['time'] =  pd.to_datetime(dfLong['time'], format='%m-%d-%Y %H:%M')
# dfLong.set_index('time')

# id = dfLong.loc[:,'ID']
# time = dfLong.loc[:,'time']
# #activity = dfLong.loc[:,'activity'].tolist()
# activity = dfLong.loc[:,'activity']

# result3 = []

# for i in range(len(id)):
# #for i in range(50):
#     print(i)
#     #tuple = (id[i], time[i], activity[i])
#     tuple = (id[i], time[i])
#     result3.append(tuple)
    
# #result4.to_csv("activityData5.csv")    
# multi_index = pd.MultiIndex.from_tuples(result3, names=["ID", "time"])    
# index = pd.MultiIndex.from_tuples(result3, names=["ID", "time"])

# cols = pd.MultiIndex.from_tuples(["A"])


# result4 = pd.DataFrame(dfLong.loc[:,'activity'].tolist(), columns=cols,index=pd.DatetimeIndex(multi_index))
# print(result4)

# result4['A'] = result4['A'].astype(int)
# #result4.index[-1] = pd.to_datetime(result4.index[-1])


# dfLong['time'] =  pd.to_datetime(dfLong['time'], format='%m-%d-%Y %H:%M')
# dfLong.set_index('time')
# print("Created at %s:%s" % (dfLong[1,'time'].hour, dfLong[1,'time'].minute))

# id = dfLong.loc[:,'ID']
# time = dfLong.loc[:,'time']
# #activity = dfLong.loc[:,'activity'].tolist()
# activity = dfLong.loc[:,'activity']

# result5 = []

# for i in range(len(id)):
# #for i in range(50):
#     print(i)
#     #tuple = (id[i], time[i], activity[i])
#     tuple = (id[i], time[i])
#     tuple2 = (tuple, activity[i])
#     result5.append(tuple2)

# index, values = zip(*result5)

# frame = pd.DataFrame({
#     'values': values
# }, index=pd.DatetimeIndex(index))



# multi_index = pd.MultiIndex.from_tuples(result3, names=["ID", "time"])    
# index = pd.MultiIndex.from_tuples(result3, names=["ID", "time"])

# cols = pd.MultiIndex.from_tuples(["A"])


# result4 = pd.DataFrame(dfLong.loc[:,'activity'].tolist(), columns=cols,index=pd.DatetimeIndex(multi_index))
# print(result4)

# result4['A'] = result4['A'].astype(int)
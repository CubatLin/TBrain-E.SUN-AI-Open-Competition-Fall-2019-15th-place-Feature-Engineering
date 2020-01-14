#TBrain E.SUN AI Open Competition Fall 2019 15th place(top1%) Feature Engineering
#--------------------------------------------------------------------------#
#Team:菜雞互啄
#Members:Alexi,Pat,Michael,Ming-Xiang,Ethan
#F1 score prediction competition: 15th of 1366 teams
#Creativity presentation competition: 2nd
#Competition Link: https://tbrain.trendmicro.com.tw/Competitions/Details/10
#--------------------------------------------------------------------------#
import pandas as pd
import numpy as np
import copy
from collections import deque
from pandas.core.frame import DataFrame

#Sample Data
df = pd.DataFrame(np.random.randint(0,1000,size=(10000, 1)),columns = ["time"])
df['GroupKey'] = np.random.choice(a=[10, 20, 30,40,50,60,70,80,90,100],size=10000,p=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
df['Cat1'] = np.random.choice(a=[100, 200, 300,400,99,33,45,6,7,8,3,3333,32],size=10000,p=[1/13,1/13,1/13,1/13,1/13,1/13,1/13,1/13,1/13,1/13,1/13,1/13,1/13])
df['ID'] = df.index

#sort data
#Notice:You need to sort data by  yourself for the purpose of Rolling_Mode_Function can get correct mode feature(s)
df = df.sort_values(["GroupKey","time"])

#Part1:Get Mode Function
def Get_Mode(mode_dic):
       MaxCount = 0;tmp_mode_deque = deque();
       for i in mode_dic.keys():
           if mode_dic[i]>=MaxCount:
               MaxCount=mode_dic[i]
               tmp_mode_deque.append([i,mode_dic[i]])
       mode_list=[]
       for i in range(len(tmp_mode_deque)):
           if tmp_mode_deque[i][1]>= MaxCount:
               mode_list.append(tmp_mode_deque[i][0])
       return mode_list

#Part2:Main Fucntion with Leakage option   
def Rolling_Mode_Function(df,GroupKey,JoinKey,Var,Leakage=False):
    tmp = copy.deepcopy(df)
    tmp_list = [list(tmp[JoinKey]),
                list(tmp[GroupKey]),
                list(tmp[Var].astype(str)),
                [0]*tmp.shape[0]
                ]
    data=[]
    for j in range(len(tmp_list[0])):
        data.append([tmp_list[i][j] for i in range(len(tmp_list))])

    GroupKey_bool = None;start_point=0
    for i in range(len(data)):
        if data[i][1]!=GroupKey_bool:
            if (Leakage==True and i>0):
                save_point=i
                mode_dic_list = {value: key for key, value in mode_dic.items()}
                mode_dic_list = [(str(mode_dic_list[d])) for d in sorted(mode_dic_list.keys(),reverse = True)]
                for j in range(start_point,save_point,1):
                    data[j].append([])
                    for l in range(len(mode_dic_list)):
                        if mode_dic_list[l] in set(data[j][2]):
                           data[j][4].append(mode_dic_list[l])
                           break;              
                start_point=save_point
            GroupKey_bool=data[i][1]
            mode_dic={}
            mode_dic[data[i][2]]=mode_dic.get(data[i][2],0)+1
            data[i][2] = [data[i][2]]
            data[i][3] = len([data[i][2]])
        else:
            mode_dic[data[i][2]]=mode_dic.get(data[i][2],0)+1
            mode_value = Get_Mode(mode_dic)
            data[i][2] = mode_value
            data[i][3] = len(mode_value)

    save_point=i+1
    mode_dic_list = {value: key for key, value in mode_dic.items()}
    mode_dic_list = [(str(mode_dic_list[d])) for d in sorted(mode_dic_list.keys(),reverse = True)]
    for j in range(start_point,save_point,1):
        data[j].append([])
        for l in range(len(mode_dic_list)):
            if mode_dic_list[l] in set(data[j][2]):
                data[j][4].append(mode_dic_list[l])
                break;

    data=DataFrame(data)
    if Leakage==False:
        data.columns = [str(JoinKey),str(GroupKey),Var+"_mode",Var+"_mode_count"]
    else:
        data.columns = [str(JoinKey),str(GroupKey),Var+"_mode",Var+"_mode_count",Var+"_mode_Leckage"]
    globals()[Var + "_ModeFrame" ] = data
    print("Done: "+Var+"_ModeFrame")

#Input four parameters:
#Rolling_Mode_Function(DataFrame,Grouping Key,JoinKey,Feature)
import time
tStart = time.time()

Rolling_Mode_Function(df,'GroupKey','ID','Cat1',Leakage=True)

tEnd = time.time()
print ("It cost %f sec" % (tEnd - tStart))
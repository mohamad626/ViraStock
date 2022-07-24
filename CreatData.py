# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 18:59:24 2021

@author: ASUS V502
"""

#Import Module
import pandas as pd
import numpy as np
from numpy import *
import math


kx=1
def Creat_data(Data,mt,td):

    Data2=Data
    tx1=math.ceil((td)*kx); tx2=math.ceil((td)*2*kx); tx3=math.ceil((td)*3*kx); tx4=math.ceil((td)*4*kx) 
    tx5=math.ceil((td)*5*kx)
    tx=max(tx1,tx2,tx3,tx4,tx5)
    ts=max(tx,(tx2+tx4))
    Data=Data[1:(len(Data)-mt+1)]
    ap2=np.array(Data2['Close'])
    ap=np.array(Data['Close'])
    
    # Creat Feauters 
    
# x1 Calc Ratio Price to MovingAverage
    ma=pd.DataFrame()
    ma['AM1']=Data['Close'].rolling(window=tx1).mean()
    ma['AM2']=Data['Close'].rolling(window=tx2).mean()
    ma['AM3']=Data['Close'].rolling(window=tx3).mean()
    ma['AM4']=Data['Close'].rolling(window=tx4).mean() 
    ma['AM5']=Data['Close'].rolling(window=tx5).mean()
    xt1=pd.DataFrame()
    xt1['val1']=Data['Close']/ma['AM1']
    xt1['val2']=Data['Close']/ma['AM2']
    xt1['val3']=Data['Close']/ma['AM3']
    xt1['val4']=Data['Close']/ma['AM4']
    xt1['val5']=Data['Close']/ma['AM5']
    xt1=np.array(xt1)
    x1=xt1[1:(len(xt1)-td+1)]
    xp1=xt1[(len(xt1)-td):len(xt1)]
    
# x2 Calc Ratio Volume to MovingAverageVolume
    vt=pd.DataFrame()
    vt['vt1']=Data['Volume'].rolling(window=tx1).mean()+ 0.001
    vt['vt2']=Data['Volume'].rolling(window=tx2).mean()+ 0.001
    vt['vt3']=Data['Volume'].rolling(window=tx3).mean()+ 0.001
    vt['vt4']=Data['Volume'].rolling(window=tx4).mean()+ 0.001
    vt['vt5']=Data['Volume'].rolling(window=tx5).mean()+ 0.001
    xt2=pd.DataFrame()
    xt2['val1']=Data['Volume']/vt['vt1']
    xt2['val2']=Data['Volume']/vt['vt2']
    xt2['val3']=Data['Volume']/vt['vt3']
    xt2['val4']=Data['Volume']/vt['vt4']
    xt2['val5']=Data['Volume']/vt['vt5']
    xt2=np.array(xt2)
    x2=xt2[1:(len(xt2)-td+1)]
    xp2=xt2[(len(xt2)-td):len(xt2)]
    
# x3 Calc Ratio Macd Indicator
    ma12=pd.DataFrame()
    ma12['ma12']=Data['Close'].rolling(window=tx1).mean()
    ma26=pd.DataFrame()
    ma26['ma26']=Data['Close'].rolling(window=tx4).mean()
    MACDLine = ma12['ma12']-ma26['ma26']
    SignalLine = MACDLine.rolling(window=tx2).mean()
    xt3=pd.DataFrame()
    xt3['val']=MACDLine/SignalLine
    xt3=np.array(xt3)
    x3=xt3[1:(len(xt3)-td+1)]
    xp3=xt3[(len(xt3)-td):len(xt3)]
      
# x4 Calc Ratio Bollinger Band

    period=[tx1,tx2,tx3,tx4,tx5]
    period_1=[14]
    xt4=pd.DataFrame()
    for j in period:
        Middle=pd.DataFrame()
        Middle['Mid']=Data['Close'].rolling(window=j).mean()
        mstd=pd.DataFrame()
        mstd['mstd']=Data['Close'].rolling(window=j).std()
        upper=Middle['Mid']+2*mstd['mstd'] 
        lower=Middle['Mid']-2*mstd['mstd']
        xtj4=pd.DataFrame()
        xtj4['val1']=upper/Data['Close']
        xtj4['val2']=lower/Data['Close']
        xtj4=np.array(xtj4)
        col_x4_1='x'+str(j)
        col_x4_2='x'+str(j+1)
        xt4[col_x4_1]=xtj4[:,0]
        xt4[col_x4_2]=xtj4[:,1]
     
    xt4=np.array(xt4)
    x4=xt4[1:(len(xt4)-td+1)]
    xp4=xt4[(len(xt4)-td):len(xt4)]
      
# x5 Calc Ratio Riward to Risk
    xt5=pd.DataFrame()
    period_2=[tx5]
    for j in period:
        Highest=pd.DataFrame()
        Highest['High']=Data['High'].rolling(window=j).max()
        Lowest=pd.DataFrame()
        Lowest['Low']=Data['Low'].rolling(window=j).min()
        Riward=(Highest['High']/Data['Close'])
        Risk=Data['Close']/Lowest['Low']
        xtj5=pd.DataFrame()
        xtj5['val1']=Riward/Risk
        xtj5=np.array(xtj5)
        col_x5='x'+str(j)
        xt5[col_x5]=xtj5[:,0]
    xt5=np.array(xt5)
    x5=xt5[1:(len(xt5)-td+1)]
    xp5=xt5[(len(xt5)-td):len(xt5)]
    
# x6 Calc Data Riward  
    r1=Data['Close'].shift(td)
    r2=Data['Close']
    r3=r2/r1
    r3=r3.dropna()
    v1=list(zeros((td,1),dtype=float)); v2=list(zeros((td,1),dtype=float)); v3=list(zeros((td,1),dtype=float)); v4=list(zeros((td,1),dtype=float))
    k=len(r3)
    for i in range(0,k):
      ri=r3[0:i+1]
      ri=ri.dropna()
      ri=np.array(ri)
      s=len(ri)
      xmax=ri.max()
      xmin=ri.min()
      xmean=ri.mean()
      xlast=ri[s-1]
      v1.append(xmax)
      v2.append(xmin)
      v3.append(xmean)
      v4.append(xlast)
    xt6=pd.DataFrame()
    xt6['val1']=v1
    xt6['val2']=v2
    xt6['val3']=v3
    xt6['val4']=v4
    xt6=np.array(xt6)
    x6=xt6[1:(len(xt6)-td+1)]
    xp6=xt6[(len(xt6)-td):len(xt6)]
    
    
# x7 Calc MFI
   
    df=Data
    typical_price=(df['Close']+df['High']+df['Low'])/3
    xt7=pd.DataFrame()
    for j in period:
        money_flow=typical_price*df['Volume']
        money_flow
        positive_flow=[]
        negative_flow=[]
        for i in range(1,len(typical_price)):
         if typical_price[i]>typical_price[i-1]:
            positive_flow.append(money_flow[i-1])
            negative_flow.append(0)
         elif typical_price[i]<typical_price[i-1]:
            positive_flow.append(0)
            negative_flow.append(money_flow[i-1])
    
         else :
            positive_flow.append(0)
            negative_flow.append(0)
    
        positive_mf=[]
        negative_mf=[]
    
        for i in range(j-1,len(positive_flow)):
            positive_mf.append(sum(positive_flow[i+1-j:i+1]))
                               
        for i in range(j-1,len(negative_flow)):
            negative_mf.append(sum(negative_flow[i+1-j:i+1]))
            
        mfi=100*(np.array(positive_mf) / (np.array(positive_mf) + np.array(negative_mf)))
        mtx=np.zeros((j))
        mfi_j = np.concatenate((mtx, mfi))
        mfi_j=pd.DataFrame(mfi_j)
        col_x7='x'+str(j)
        xt7[col_x7]=mfi_j[0]
    xt7=np.array(xt7)
    x7=xt7[1:(len(xt7)-td+1)]
    xp7=xt7[(len(xt7)-td):len(xt7)]
    
    
# x8 Calc RSI
    def EMA(S:np.ndarray, L:int, r:float=1):
        a = (1 + r) / (L + r)
        nD0 = S.size
        nD = nD0 - L + 1
        M = np.zeros(nD)
        M[0] = np.mean(S[:L])
        for i in range(1, nD):
            M[i] = a * S[i + L - 1] + (1 - a) * M[i - 1]
        return M
    
    
    xt8=pd.DataFrame()
    for j in period:
        L=j
        C=Data['Close']
        nD0 = C.size
        nD1 = nD0 - 1
        U = np.zeros(nD1)
        D = np.zeros(nD1)
        for i in range(nD1):
            d = C[i + 1] - C[i]
            if d > 0:
                U[i] = d
            else:
                D[i] = -d
        emaU = EMA(U, L, r=0)
        emaD = EMA(D, L, r=0)
        RS = emaU / emaD
        rsi = 100 - 100 / (1 + RS)
        mtx=np.zeros((j))
        rsi_j = np.concatenate((mtx, rsi))
        rsi_j=pd.DataFrame(rsi_j)
        col_x8='x'+str(j)
        xt8[col_x8]=rsi_j[0]
    
    xt8=np.array(xt8)
    x8=xt8[1:(len(xt8)-td+1)]
    xp8=xt8[(len(xt8)-td):len(xt8)]
    
    # x9 Calc Ratio Price to other price
    rap=pd.DataFrame()
    rap['v1']=Data['High']
    rap['v2']=Data['Low']
    rap['v3']=Data['Open']
   
    xt9=pd.DataFrame()
    xt9['val1']=Data['Close']/rap['v1']
    xt9['val2']=Data['Close']/rap['v2']
    xt9['val3']=Data['Close']/rap['v3']

    xt9=np.array(xt9)
    x9=xt9[1:(len(xt9)-td+1)]
    xp9=xt9[(len(xt9)-td):len(xt9)]
    
    # Assembly Input Data
    xtotal=np.concatenate((x1,x2,x3,x4,x5,x6,x7,x8,x9),1)
    xptotal=np.concatenate((xp1,xp2,xp3,xp4,xp5,xp6,xp7,xp8,xp9),1)
    P=xtotal[ts:len(xtotal)]
    Pe=xptotal
    
    # Assembly Target Data
    t1=Data['Close'].shift(td)
    t2=Data['Close']
    t3=t2/t1
    T=t3[(td+ts):len(t3)]
    T=np.array(T)
    return Data,Data2,P,T,Pe,ap,ap2
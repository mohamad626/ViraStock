# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 23:46:29 2021

@author: ASUS V502
"""

import streamlit as st
import yfinance as yf
import pandas as pd
#from datetime import datetime 
from sklearn.model_selection import train_test_split
#import os;
#import matplotlib.pyplot as plt


st.title('پیش بینی قیمت ارزهای دیجیتال')
st.title('ساخته شده توسط محمد بیابانی')
symbol_list={'بیت کوین':'BTC','اتریوم':'ETH','باینسس کوین':'BNB','سولانا':'SOL','ریپل':'XRP'}
symbol_name= st.selectbox('نام ارز مورد نظر',symbol_list)
symbol=st.write('نمادی که انتخاب کردید' +' :' +  symbol_list[symbol_name])
symbol_final=symbol_list[symbol_name]+'-USD'
interval_list={'1d':'max','1m':'5d','2m':'1mo','5m':'1mo','15m':'1mo','30m':'1mo','1h':'2y'}
interval_name= st.selectbox('تایم فریم مورد نظر',interval_list)
td=st.number_input('تعداد کندلهای مورد نظر را وارد کنید',value=50)
period_sel=interval_list[interval_name]
#symbol_final=symbol_name +'-USD'

data_read =yf.download(tickers = symbol_final, period =period_sel ,interval =interval_name)
Data=data_read.dropna()
Data2=Data
mt=0


from CreatData import Creat_data
Data,Data2,P,T,Pe,ap,ap2=Creat_data(Data,mt,td)

from sklearn.tree import DecisionTreeRegressor
X_train, X_test, y_train, y_test = train_test_split(P,T)
tree = DecisionTreeRegressor().fit(X_train,y_train)
print("Training set accuracy: {:.3f}".format(tree.score(X_train, y_train)))
print("Test set accuracy: {:.3f}".format(tree.score(X_test, y_test)))
dtree = tree.predict(Pe)
Newprice=dtree*ap[len(Data)-td:len(Data)]


p0=ap[len(ap)-1]
pi=[p0]
priceedit=[p0]
Se=[]
for i in range(len(Newprice)):
    pf=Newprice[i] 
    pi.append(pf)
    si=pi[i+1]/pi[i]
    if si>1.05:
       sei=1.05
    elif si<.95:  
        sei=.95
    else:
        sei=si
    Se.append(sei)
    pe=priceedit[i]*sei
    priceedit.append(pe)



Pf=pd.DataFrame()
Pf['قیمت پیش بینی شده']=priceedit
# Pf['نمودار قیمت در روزهای گذشته']=ap2[len(Data2)-mt:len(Data2)]
st.write(' نمودار قیمت پیش بینی شده')
st.line_chart(Pf)

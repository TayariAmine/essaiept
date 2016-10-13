# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 19:29:35 2016

@author: oualid
"""
from download_ohlc import *
from linear_model import *
from ta_lib import *
from datetime import datetime
from pandas.io.data import DataReader
import pytz
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import math 

# defining the start and end time
START = datetime(1990, 1, 1, 0, 0, 0, 0, pytz.utc)
END = datetime.today().utcnow()

# loading the list of the 500 symbols constituents of the SPY500 index
spy500 = pd.read_csv("constituents.csv")
# downloading historical daily data for all 500 symbols
data_spy = download_ohlc(spy500.Symbol, START, END)
#Selecting Apple data 
df = data_spy['AAPL']
# Computing the "independent features" 
for n in (1, 2, 3, 4, 5, 10, 20, 60):
    df = LogRet(df, n)
    
for  n in ( 3, 4, 5, 10, 20, 60 ):
    df = SMAChange(df,n)


for  n in (3, 4, 5, 10, 20, 60):
    df = ExpMA(df,n)

for  n in (3, 4, 5, 10, 20, 60):
    df = VWExpMA(df,n)
    
# defining the target variable
Target = pd.Series(df.LogRet_2.shift(-2), name = 'Target', index = df.index)
df = df.join(Target)
# Putting independent  and dependent variables in a separate data frame
Data  = df[['LogRet_1', 'LogRet_2', 'LogRet_3', 'LogRet_4', 'LogRet_5', 'LogRet_10', 'LogRet_20', 
               'SMAChange_3', 'SMAChange_4', 'SMAChange_5', 'SMAChange_10',
       'SMAChange_20', 'EMA_3', 'EMA_4', 'EMA_5', 'EMA_10', 'EMA_20',
       'VWEMA_3', 'VWEMA_4', 'VWEMA_5', 'VWEMA_10', 'VWEMA_20', 'Target']]
# Dropping the missing values
Data.dropna(inplace=True)
# Fitting a linear regression 
lr = LinearRegression()
lr = lr.fit(Data.ix[:,:-1], Data.Target)





#### DATA ANALYSIS ####


#### Statistics description    
Data.describe()


#### Histogram plots
i=1
for variable in ['LogRet_1', 'LogRet_2', 'LogRet_3', 'LogRet_4', 'LogRet_5', 'LogRet_10', 'LogRet_20', 
               'SMAChange_3', 'SMAChange_4', 'SMAChange_5', 'SMAChange_10',
       'SMAChange_20', 'EMA_3', 'EMA_4', 'EMA_5', 'EMA_10', 'EMA_20',
       'VWEMA_3', 'VWEMA_4', 'VWEMA_5', 'VWEMA_10', 'VWEMA_20','Target']:
           plt.figure(str(i))
           plt.hist(Data[variable])
           plt.title(variable+"Histogram")
           i=i+1
            

#### Scatter plots
i=1
for variable in ['LogRet_1', 'LogRet_2', 'LogRet_3', 'LogRet_4', 'LogRet_5', 'LogRet_10', 'LogRet_20', 
               'SMAChange_3', 'SMAChange_4', 'SMAChange_5', 'SMAChange_10',
       'SMAChange_20', 'EMA_3', 'EMA_4', 'EMA_5', 'EMA_10', 'EMA_20',
       'VWEMA_3', 'VWEMA_4', 'VWEMA_5', 'VWEMA_10', 'VWEMA_20']:
           plt.figure(str(i))
           plt.xlabel(variable)
           plt.ylabel('Target')
           plt.scatter(Data[variable],Data['Target'])
           plt.title(variable+"/ Target Scatter Plot")
           i=i+1

"""
Some variables show visual patterns--> For example: When the scatter plot is horizontal, there is no correlation between the target and the considered variable

"""

#### k-Binned scatter plot
from bscatter import *
i=1
for variable in ['LogRet_1', 'LogRet_2', 'LogRet_3', 'LogRet_4', 'LogRet_5', 'LogRet_10', 'LogRet_20', 
               'SMAChange_3', 'SMAChange_4', 'SMAChange_5', 'SMAChange_10',
       'SMAChange_20', 'EMA_3', 'EMA_4', 'EMA_5', 'EMA_10', 'EMA_20',
       'VWEMA_3', 'VWEMA_4', 'VWEMA_5', 'VWEMA_10', 'VWEMA_20']:
           plt.figure(str(i))
           plt.xlabel(variable)
           plt.ylabel('Target')
           (Xq,XSe,Yq,YSe)=bscatter(Data[variable],Data['Target'],10)
           plt.scatter(Xq,Yq)
           plt.title(variable+"/Target K_Binned Scatter Plot")
           i=i+1

i=1
for variable in ['LogRet_1', 'LogRet_2', 'LogRet_3', 'LogRet_4', 'LogRet_5', 'LogRet_10', 'LogRet_20', 
               'SMAChange_3', 'SMAChange_4', 'SMAChange_5', 'SMAChange_10',
       'SMAChange_20', 'EMA_3', 'EMA_4', 'EMA_5', 'EMA_10', 'EMA_20',
       'VWEMA_3', 'VWEMA_4', 'VWEMA_5', 'VWEMA_10', 'VWEMA_20']:
           plt.figure(str(i))
           plt.xlabel(variable)
           plt.ylabel('Target')
           (Xq,XSe,Yq,YSe)=bscatter(Data[variable],Data['Target'],10)
           plt.scatter(XSe,YSe)
           plt.title(variable+"/Target K_Binned Scatter Plot")
           i=i+1

"""
These scatter plots show  independent cases(Example: VWEMA_20),a positive distribution(Example LogRET1) and a negative distribution(Example EMA)

"""

#### Linear Correlation
from scipy.stats.stats import pearsonr
corr=[]
for variable in ['LogRet_1', 'LogRet_2', 'LogRet_3', 'LogRet_4', 'LogRet_5', 'LogRet_10', 'LogRet_20', 
               'SMAChange_3', 'SMAChange_4', 'SMAChange_5', 'SMAChange_10',
       'SMAChange_20', 'EMA_3', 'EMA_4', 'EMA_5', 'EMA_10', 'EMA_20',
       'VWEMA_3', 'VWEMA_4', 'VWEMA_5', 'VWEMA_10', 'VWEMA_20']:
           matrix =numpy.corrcoef(Data['Target'],Data[variable])
           linear_correlation=matrix[0,1]
           print ('Correlation between Target and '+variable +' is '+str(linear_correlation))
           corr.append(linear_correlation)

"""           
This coeffecient tells us about the strength of the relationship betwenn x and y
If the test concludes that the correlation coefficient is significantly different from 0, the correlation coefficient is "significant"
If the coefficient is positive we have a positive correlation else we have a negative correlation
If the test concludes that the correlation coefficient is not significantly diffrent from 0 (it is close to 0), the correlation coefficient is "not significant"

"""

#### Mutual Information
mut=[]
for n in list(Data)[0:len(list(Data))-1]:
    tar=pd.qcut(Data['Target'], 20, labels=False)
    variable=pd.qcut(Data[n], 20, labels=False)
    prob=pd.crosstab(variable, tar, margins=True)
    i=0
    Info=0
    for i in range(0,19):
        for j in range(0,19):
            Info=Info+prob.iloc[i,j]*(log((prob.iloc[i,j]*prob.iloc[20,20]))-log((prob.iloc[i,20]*prob.iloc[20,j])))/prob.iloc[20,20]
    mut.append(Info)


#### Plot of mutual information against the linear correlation
plt.scatter(x= corr,y= mut)
plt.title("mutuel information=f(correlation)")

""" 
The mutual information and the linear correlation are describing different aspects of the association between two random variables. 
Mutual Information doesn't mean whether the association is linear or not, while the linear correlation may be zero and the variables may still be stochastically dependent. 
Moreover, the linear correlation can be calculated directly from a data sample without the need to actually know the probability distributions involved , while Mutual Information requires knowledge of the distributions, whose estimation, if unknown, is a much more delicate and uncertain work compared to the estimation of the linear correlation.

""" 



#### LINEAR MODEL ####


#### Fitting a linear regression 
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr = lr.fit(Data.ix[:,:-1], Data.Target)


### Assumptions
"""
*Weak exogeneity: This essentially means that the predictor variables x can be treated as fixed values, rather than random variables
*Linearity: This means that the mean of the response variable is a linear combination of the parameters (regression coefficients) and the predictor variables
*Constant variance (homoscedasticity): This means that different response variables have the same variance in their errors
*Independence of the errors
*Lack of multicollinearity in the predictors: For standard least squares estimation methods, the design matrix X must have full column rank p 

--> Mathematically speaking 

Epsilon stands for the residual: 

*var(Epsiloni)=constant
*cov(Epsiloni,Epsilonj)=0 
*cov(xi,xj)=0
*cov(xi,Epsilonj))=0

"""

#### Linear model using the training data and test data
from sklearn.cross_validation import train_test_split

#Data split
train, test = train_test_split(Data,test_size=0.2)

#Linear model for training 
trainFit = LinearRegression()
trainFit = trainFit.fit(train.ix[:,:-1], train.Target)

#Prediction
pred = trainFit.predict(test.ix[:,:-1])


#### 20-binned testing scatter plot
k=20
Xq,XSe,Yq,YSe=bscatter(pred, test.iloc[:,-1],k)
plt.scatter(Xq,Yq)
plt.title("20-Binned testing scatter plot Y function of y^")

#### 20-binned training scatter plot
predTrain = trainFit.predict(train.ix[:,:-1])
k=20
Xq,XSe,Yq,YSe=bscatter(predTrain, train.iloc[:,-1],k)
plt.scatter(Xq,Yq)
plt.title("20-Binned training scatter plot Y function of y^")

#### R-squared
RsquaredTest = trainFit.score(test.ix[:,:-1], test.iloc[:,-1], sample_weight=None)
RsquaredTrain = trainFit.score(train.ix[:,:-1], train.iloc[:,-1], sample_weight=None)


#### Trading strategy implementation
a=df.ix[-2:,5:-1]
a=a[['LogRet_1', 'LogRet_2', 'LogRet_3', 'LogRet_4', 'LogRet_5', 'LogRet_10', 'LogRet_20', 
               'SMAChange_3', 'SMAChange_4', 'SMAChange_5', 'SMAChange_10',
       'SMAChange_20', 'EMA_3', 'EMA_4', 'EMA_5', 'EMA_10', 'EMA_20',
       'VWEMA_3', 'VWEMA_4', 'VWEMA_5', 'VWEMA_10', 'VWEMA_20']]
predTrading = trainFit.predict(a)
print("The profit (/loss) would be " + str(Data.ix[:,-1][-1]))
































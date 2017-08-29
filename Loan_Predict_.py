# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 15:37:58 2017

@author: Nishant
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder



train = pd.read_csv('train_data.csv', index_col = 0)
test = pd.read_csv('test_data.csv', index_col = 0)
col_names = train.columns.tolist()
print(col_names)
#print(train.info())
#print(train.head())

features = ['Gender','Married','Education','Self_Employed','Dependents','Loan_Status']

fill_withCommon = ['Dependents','Gender','Credit_History','Married','Self_Employed']
fill_withMean = ['LoanAmount','Loan_Amount_Term']

def featureEng(data):
    #Removing Loans_ID
    df = data #.drop('Loan_ID',axis=1) 
    
    #Removing or filling NaN Values
    for feature in fill_withMean:
        if feature in data.columns.values:
            df[feature] = df[feature].fillna(df[feature].mean())
            
    for feature in fill_withCommon:
        if feature in data.columns.values:
            df[feature] = df[feature].fillna(df[feature].value_counts().index[0])
    
    #For encoding features/converting them from categorical to float using label encoder
    
    for feature in features:
        if feature in data.columns.values:
            le = LabelEncoder()
            df[feature] = le.fit_transform(df[feature].fillna('0'))
    
    #Adding some features
    #House_Income = Application Income + Coapplicant Income
    df['House_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df = df.drop(['ApplicantIncome',  'CoapplicantIncome'], axis = 1)
    
    #Now Featuring some remaning values
    
    dummies = pd.get_dummies(df.Property_Area)
    df = pd.concat([df,dummies],axis=1)
    df = df.drop('Property_Area',axis=1)
    
    return df


train = featureEng(train)
test = featureEng(test)

#train['Dependents'] = test['Dependents'].fillna(0)
#train['Dependents'].notnull().value_counts()
#test['Dependents'] = test['Dependents'].fillna(0)
#test['Dependents'].notnull().value_counts()

#print(train.head())

train.insert(len(train.columns)-1,'Loan_Status',train.pop('Loan_Status'))
scale_features = ['LoanAmount','Loan_Amount_Term','House_Income']

train[scale_features] = train[scale_features].apply(lambda x:(x.astype(int) - min(x))/(max(x)-min(x)), axis = 0)
test[scale_features] = test[scale_features].apply(lambda x:(x.astype(int) - min(x))/(max(x)-min(x)), axis = 0)

train_X = train.iloc[:, :-1]
train_Y = train.iloc[:,-1]
test_X = test

#print(train_Y.head())
#print(train.info())


lr = LogisticRegression()
lr.fit(train_X, train_Y)

pred_Y = lr.predict(test_X)

pred_Y = ['Y' if x==1 else 'N' for x in pred_Y]

test_X['Loan_Status']=pred_Y
test_X = test_X['Loan_Status']
print(test_X)

test_X.to_csv('loan_submission_Fit.csv',sep=',',header=True)
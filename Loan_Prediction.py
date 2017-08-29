# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 14:52:37 2017

@author: Nishant
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression


train = pd.read_csv('train_data.csv', header = 0)
test = pd.read_csv('test_data.csv', header = 0)
full_data = [train, test]
#print(train.head())
#print(train.info())

#Cleaning the train dataset
train['Gender'] = train['Gender'].fillna('Male')
train['Gender'].notnull().value_counts()
train['Married'] = train['Married'].fillna('Yes')
train['Married'].notnull().value_counts()
train['Self_Employed'] = train['Self_Employed'].fillna('No')
train['Self_Employed'].notnull().value_counts()
train['Credit_History'] = train['Credit_History'].fillna(1)
train['Credit_History'].notnull().value_counts()
train['Dependents'] = train['Dependents'].fillna(0)
train['Dependents'].notnull().value_counts()
train['LoanAmount']=train['LoanAmount'].fillna(np.mean(train.LoanAmount))
train['LoanAmount_Log']=np.log(train['LoanAmount'])
train['Loan_Amount_Term']=train['Loan_Amount_Term'].fillna(360)
train['Loan_Amount_Term_Log']=np.log(train['Loan_Amount_Term'])
train['TotalIncome']=train['LoanAmount_Log']+train['Loan_Amount_Term_Log']
#print(train.Credit_History.head())

#cleaning the test data
test['Gender'] = train['Gender'].fillna('Male')
test['Gender'].notnull().value_counts()
test['Married'] = train['Married'].fillna('Yes')
test['Married'].notnull().value_counts()
test['Self_Employed'] = train['Self_Employed'].fillna('No')
test['Self_Employed'].notnull().value_counts()
test['Credit_History'] = train['Credit_History'].fillna(1)
test['Credit_History'].notnull().value_counts()
test['Dependents'] = train['Dependents'].fillna(0)
test['Dependents'].notnull().value_counts()
test['LoanAmount']=train['LoanAmount'].fillna(np.mean(train.LoanAmount))
test['LoanAmount_Log']=np.log(train['LoanAmount'])
test['Loan_Amount_Term']=train['Loan_Amount_Term'].fillna(360)
test['Loan_Amount_Term_Log']=np.log(train['Loan_Amount_Term'])
test['TotalIncome']=train['LoanAmount_Log']+train['Loan_Amount_Term_Log']

#Converting categorical data to float/numerical

var_mod = ['Gender','Married','Education','Self_Employed','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    train[i] = le.fit_transform(train[i])
#print(train.dtypes)

var_mod = ['Gender','Married','Education','Self_Employed']
le = LabelEncoder()
for i in var_mod:
    test[i] = le.fit_transform(test[i])
#print(test.dtypes)

train_X = train[['Gender', 'Married', 'Education', 'Self_Employed', 'Credit_History', 'TotalIncome']].values
train_Y = train['Loan_Status'].values
#print(train_X)
test_X = test[['Gender', 'Married', 'Education', 'Self_Employed', 'Credit_History', 'TotalIncome']].values

lm = LinearRegression()
lm.fit(train_X, train_Y)
test_Y = lm.predict(test_X)
print(test_Y)


test_Y = ['Y' if x==1 else 'N' for x in test_Y]



report=pd.DataFrame({' Loan_Status ' : np.array(test_Y)}, index=test.Loan_ID)
print(report)


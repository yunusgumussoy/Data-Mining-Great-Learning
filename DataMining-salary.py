# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 23:46:18 2023

@author: Yunus

Data Mining Course @ Great Learning
"""

# importing all the necessary packages 
import pandas as pd
import numpy as np

# Loading the data set 
data=pd.read_csv("Salary.csv")

# checking first 5 rows
data.head()

data.columns

data.describe()

data.shape

# checking missing values
data.isnull()

data.isnull().any()

data.isnull().sum()

import matplotlib.pyplot as plt
import seaborn as sns

plt.plot(data['Salary'],data['YearsExperience'])

data

from sklearn.model_selection import train_test_split
x = data.drop('Salary',axis = 1)

x

y=data['Salary']

y.head()

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 42) 

from sklearn.linear_model import LinearRegression 

L=LinearRegression()

L.fit(xtrain,ytrain)

y_pred=L.predict(xtest)

print(L.score(xtest, ytest)) 

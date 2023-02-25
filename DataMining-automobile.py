# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 21:04:10 2023

@author: Yunus

Data Mining Course @ Great Learning
"""
# Exploratory Data Analysis on Automobile data set

# Dataset Link - https://www.kaggle.com/toramky/automobile-dataset

# Importing all the necessary packages 
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

# Loading the dataset 
data=pd.read_csv('Automobile_data.csv')

data.head()

data.columns

# checking the dimension of the data set 
data.shape

# printing the first 5 records of the data set 
data.head()

data.isnull().any()

data.describe()

# Missing Value Detection and Treatment

# checking the count of records in each column of the data set. 
# If the count of records is lesser than the total number of records, we can conclude that there are blank records. 

data.info()

# Treating missing values in the "normalized losses" column
from numpy import nan
data['normalized-losses']=data['normalized-losses'].replace('?',np.nan)

# checking if the '?' have been replaced with nan 
data.head()

# checking missing values
print(data.isnull().sum())

data.median()

# Imputing missing values with their respective columns median
data.fillna(data.median(), inplace=True)

print(data.isnull().sum())

# checking data again to make sure missing values are replaced
data.head()

data.info()

# data type of NormalizedLosses is object, we have to change its data type to numeric (float)
data['normalized-losses']=pd.to_numeric(data['normalized-losses'], downcast="float")

# checking data again to make sure NormalizedLosses type became float
data.info()

# Outlier Detection and treatment
# Outlier detection using boxplots 
plt.figure(figsize= (20,15))
plt.subplot(3,4,1)
sns.boxplot(data['normalized-losses'])

plt.subplot(3,4,2)
sns.boxplot(data['symboling'])

plt.subplot(3,4,3)
sns.boxplot(data['wheel-base'])

plt.subplot(3,4,4)
sns.boxplot(data['length'])

plt.subplot(3,4,5)
sns.boxplot(data['width'])

plt.subplot(3,4,6)
sns.boxplot(data['curb-weight'])

plt.subplot(3,4,7)
sns.boxplot(data['engine-size'])

plt.subplot(3,4,8)
sns.boxplot(data['city-mpg'])

plt.subplot(3,4,9)
sns.boxplot(data['highway-mpg'])

plt.subplot(3,4,10)
sns.boxplot(data['height'])

plt.subplot(3,4,11)
sns.boxplot(data['compression-ratio'])

# imputing outliers with the 5th and 95th percentiles.
data['normalized-losses']=data['normalized-losses'].clip(lower=data['normalized-losses'].quantile(0.05), upper=data['normalized-losses'].quantile(0.95))
data['wheel-base']=data['wheel-base'].clip(lower=data['wheel-base'].quantile(0.05), upper=data['wheel-base'].quantile(0.95))
data['length']=data['length'].clip(lower=data['length'].quantile(0.05), upper=data['length'].quantile(0.95))
data['width']=data['width'].clip(lower=data['width'].quantile(0.05), upper=data['width'].quantile(0.95))
data['engine-size']=data['engine-size'].clip(lower=data['engine-size'].quantile(0.05), upper=data['engine-size'].quantile(0.95))
data['city-mpg']=data['city-mpg'].clip(lower=data['city-mpg'].quantile(0.05), upper=data['city-mpg'].quantile(0.95))
data['highway-mpg']=data['highway-mpg'].clip(lower=data['highway-mpg'].quantile(0.05), upper=data['highway-mpg'].quantile(0.95))
data['compression-ratio']=data['compression-ratio'].clip(lower=data['compression-ratio'].quantile(0.05), upper=data['compression-ratio'].quantile(0.89))

# detecting outliers again using boxplots to make sure quantile imputing worked
plt.figure(figsize= (20,15))
plt.subplot(3,4,1)
sns.boxplot(data['normalized-losses'])

plt.subplot(3,4,2)
sns.boxplot(data['symboling'])

plt.subplot(3,4,3)
sns.boxplot(data['wheel-base'])

plt.subplot(3,4,4)
sns.boxplot(data['length'])

plt.subplot(3,4,5)
sns.boxplot(data['width'])

plt.subplot(3,4,6)
sns.boxplot(data['curb-weight'])

plt.subplot(3,4,7)
sns.boxplot(data['engine-size'])

plt.subplot(3,4,8)
sns.boxplot(data['city-mpg'])

plt.subplot(3,4,9)
sns.boxplot(data['highway-mpg'])

plt.subplot(3,4,10)
sns.boxplot(data['height'])

plt.subplot(3,4,11)
sns.boxplot(data['compression-ratio'])

# Data Vizualisation
sns.pairplot(data,diag_kind='kde')

# Correlation Plot & heat map
f, ax = plt.subplots(figsize=(20, 10))
corr = data.corr("pearson")
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool_), cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax,annot=True)

# there is multi colinearity in the data set

# risk rating histogram
data.symboling.hist(bins=6,color='red');
plt.title("Insurance risk ratings of vehicles")
plt.ylabel('Number of vehicles')
plt.xlabel('Risk rating');

# fuel type plot
data['fuel-type'].value_counts().plot(kind='bar',color='orange')
plt.title("Frequency chart (fuel type)")
plt.ylabel('Number of vehicles')
plt.xlabel('Fuel type');

# Data Preprocessing

"""
This dataset has 15 categorical variables and most of them have more than 2 categories. 
We can not run a regression model on text data. 
So, in order to deal with this challenge we will use label encoding.
Label coding is the process of converting categorical (text) data into numerical data.
"""

# data before lable encoding 
data['body-style'].head(20)

# Label encoding 
from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
data['body-style'] = labelencoder.fit_transform(data['body-style'])

# data after label encoding 
data['body-style'].head(20)

"""
After running the label encoding code, we can see that the variable body-style has numerical values ranging from 0-4.

The problem with lable encoding is that it introduces an order between the categories, i.e. 0>1>2>3>4. 
This might confuse the model into thinking that convertible is greater than hatchback.

So to deal with this problem, we will use one hot encoder.

In one hot encoding, categorical columns that have been label encoded are split into multiple colums and the values are replaced with 0's and 1's. 
1's mark the preasence of a value and 0 its absence.
"""

# data before one hot encoding 
data['body-style'].head(10)

# One hot encoding 
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(enc.fit_transform(data[['body-style']]).toarray())

# data after one hot encoding 
enc_df


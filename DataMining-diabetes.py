# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 20:16:11 2023

@author: Yunus

Data Mining Course @ Great Learning
"""
# Exploratory Data Analytics on Pima Indians Diabetes Database

# Data Set Link - https://www.kaggle.com/uciml/pima-indians-diabetes-database

"""
Data Desription

Predictor Variables

Preganancies - Number of times the patient got pregnant
Glucose - Plasma glucose concentration
Blood Preassure - Diastolic Blood Preassure (mmHg)
Skin Thickness - Triceps skin fold thickness (mm)
Insulin - 2-Hour serum insulin (mu U/ml)
BMI: Body mass index (weight in kg/(height in m)^2)
DiabetesPedigreeFunction: Diabetes pedigree function
Age: Age (years)
Outcome: Class variable (0 or 1)
"""

# importing all the necessary packages 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the data set 
data=pd.read_csv('diabetes.csv')

# checking first 5 rows
data.head()

# checking first 10 rows
data.head(10)

# checking last 5 rows
data.tail()

data.columns

# checking the dimension of the data set 
data.shape

# Missing Value Detection and Treatment
"""
The following values in a data set are considered to be missing values -
-Blank Values
-NaN
-null
-Some countinuous columns might have 0's to indicate missing data.
"""

# checking the count of records in each column of the data set
data.info()

# If the count of records is lesser than the total number of records, i.e. 768, 
# we can conclude that there are blank records.

data.isna()
data.isna().any()

data.describe()

# 0's in the columns should be replaced with the median, since the median is least affected by outliers.
# For that aim, replacing the 0's with NaN.
# The records that have 0's in columns Glucose, Blood Preassure, Skin Thickness, Insulin and BMI will be replaced with NaN 

from numpy import nan

data['Glucose']=data['Glucose'].replace(0,np.nan)

data['BloodPressure']=data['BloodPressure'].replace(0,np.nan)

data['SkinThickness']=data['SkinThickness'].replace(0,np.nan)

data['Insulin']=data['Insulin'].replace(0,np.nan)

data['BMI']=data['BMI'].replace(0,np.nan)

# Count of NaN values in the dataset 
print(data.isnull().sum())

data.median()

# Imputing missing values with their respective columns median
data.fillna(data.median(), inplace=True)

# Checking if the missing values have been imputed 
print(data.isnull().sum())

# Outlier detection using boxplots 
plt.figure(figsize= (20,15))
plt.subplot(4,4,1)
sns.boxplot(data['Pregnancies'])

plt.subplot(4,4,2)
sns.boxplot(data['Glucose'])

plt.subplot(4,4,3)
sns.boxplot(data['BloodPressure'])

plt.subplot(4,4,4)
sns.boxplot(data['SkinThickness'])

plt.subplot(4,4,5)
sns.boxplot(data['Insulin'])

plt.subplot(4,4,6)
sns.boxplot(data['BMI'])

plt.subplot(4,4,7)
sns.boxplot(data['DiabetesPedigreeFunction'])

plt.subplot(4,4,8)
sns.boxplot(data['Age'])

# Apart from 'Glucose' all the other attributes show preasence of outliers. 
# These lower level and upper level outliers will be replaced by the 5th and 95th percentile respectively.

from IPython.display import Image
from IPython.core.display import HTML 
Image(url= "https://i.ibb.co/Y8dtdDL/iqr.png")

data['Pregnancies']=data['Pregnancies'].clip(lower=data['Pregnancies'].quantile(0.05), upper=data['Pregnancies'].quantile(0.95))

data['BloodPressure']=data['BloodPressure'].clip(lower=data['BloodPressure'].quantile(0.05), upper=data['BloodPressure'].quantile(0.95))

data['SkinThickness']=data['SkinThickness'].clip(lower=data['SkinThickness'].quantile(0.05), upper=data['SkinThickness'].quantile(0.95))

data['Insulin']=data['Insulin'].clip(lower=data['Insulin'].quantile(0.05), upper=data['Insulin'].quantile(0.95))

data['BMI']=data['BMI'].clip(lower=data['BMI'].quantile(0.05), upper=data['BMI'].quantile(0.95))

data['DiabetesPedigreeFunction']=data['DiabetesPedigreeFunction'].clip(lower=data['DiabetesPedigreeFunction'].quantile(0.05), upper=data['DiabetesPedigreeFunction'].quantile(0.95))

data['Age']=data['Age'].clip(lower=data['Age'].quantile(0.05), upper=data['Age'].quantile(0.95))

# visualising the boxplots after imputing the outliers 
plt.figure(figsize= (20,15))
plt.subplot(4,4,1)
sns.boxplot(data['Pregnancies'])

plt.subplot(4,4,2)
sns.boxplot(data['Glucose'])

plt.subplot(4,4,3)
sns.boxplot(data['BloodPressure'])

plt.subplot(4,4,4)
sns.boxplot(data['SkinThickness'])

plt.subplot(4,4,5)
sns.boxplot(data['Insulin'])

plt.subplot(4,4,6)
sns.boxplot(data['BMI'])

plt.subplot(4,4,7)
sns.boxplot(data['DiabetesPedigreeFunction'])

plt.subplot(4,4,8)
sns.boxplot(data['Age'])

# As there are still outliers in columns Skin Thickness and Insulin, manipulating the percentile values.
data['SkinThickness']=data['SkinThickness'].clip(lower=data['SkinThickness'].quantile(0.07), upper=data['SkinThickness'].quantile(0.93))

plt.subplot(4,4,4)
sns.boxplot(data['SkinThickness'])

data['Insulin']=data['Insulin'].clip(lower=data['Insulin'].quantile(0.21), upper=data['Insulin'].quantile(0.80))

plt.subplot(4,4,5)
sns.boxplot(data['Insulin'])

# As there are still outliers in Insulin, manipulating the percentile values.
data['Insulin']=data['Insulin'].clip(lower=data['Insulin'].quantile(0.25), upper=data['Insulin'].quantile(0.75))

# checking outliers of Insulin again
plt.subplot(4,4,5)
sns.boxplot(data['Insulin'])

"""
The outliers of Skin Thickness were treated by minor changes in the percentiles but the outliers of insulin require a major changes in the percentiles. 
This might result in too much data manipulation, which migh jepordise the models.
Attribute Insulin might have to be removed from the data set.
"""

# Data Visualization
# understanding the distribution diabitic Vs Non Diabitic patients in the data set.

sns.countplot(x=data['Outcome'])

# Correlation Plot & heat map
f, ax = plt.subplots(figsize=(20, 10))
corr = data.corr("pearson")
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax,annot=True)

"""
The correlation plot shows the relation between the parameters.

Glucose,Age,BMI and Pregnancies are the most correlated parameters with the Outcome.
Insulin and DiabetesPedigreeFunction have little correlation with the outcome.
BloodPressure and SkinThickness have tiny correlation with the outcome.
There is a little correlation between Age and Pregnancies,Insulin and Skin Thickness, BMI and Skin Thickness,Insulin and Glucose
"""
# Pair plot analysis 
sns.pairplot(data,hue='Outcome',diag_kind='kde')

"""
From pairplot,

we can infer that most of the predictor variables are weak predictors of Outcome.

The kernal density plots (diagonal) suggests that the distribution for diabetic and non diabetic are very similar and are overlapping each other significantly, hence they wont be able to differentiate between a diabetic patient and a non diabetic patient.

The scatterplots also suggest very poorly corelated data (data with not hidden patterns or relationships). Hence models built on this data might not be able to identify any hidden patterns or might identify nonsense patterns i.e. patterns that do not make sense.

The plot shows that there is some relationship between parameters. Outcome is added as hue (Variable in “data“ to map plot aspects to different colors.). We see that blue and orange dots are overlap

Pregnancies and age have some kind of a linear line.

BloodPressure and age have little relation. Most of the aged people have BloodPressure.

Insulin and Glucose have some relation
"""

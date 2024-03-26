# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 22:32:25 2024

@author: Priyanka
"""

"""
Problem statment:
Officeworks is a leading retail store in Australia, with numerous outlets around the country. 
The manager would like to improve the customer experience by providing them online predictive 
prices for their laptops if they want to sell them. 
To improve this experience the manager would like us to build a model which is 
sustainable and accurate enough. 
Apply Lasso and Ridge Regression model on the dataset and predict the price,given other attributes. 
Tabulate R squared, RMSE, and correlation values.


# Data Dictionary:
1. price: This is the target variable we want to predict. It represents the price of the laptop.
2. speed: This could refer to the processor speed of the laptop, usually measured in GHz (gigahertz). 
3. hd: This likely represents the size of the hard disk drive (HDD) in the laptop, measured in gigabytes (GB) or terabytes (TB). 
4. ram: This represents the amount of random-access memory (RAM) in the laptop, usually measured in gigabytes (GB) or megabytes (MB).
5. screen: This refers to the size of the laptop screen, typically measured diagonally in inches. 
6. cd: This could indicate whether the laptop has a CD-ROM drive for reading CDs.
7. multi: This might represent whether the laptop has a multi-format DVD/CD-RW drive, meaning it can read and write CDs and DVDs of various formats.
8. premium: This feature could indicate whether the laptop is a premium model or has premium features. 
9. ads: This feature could represent the level of advertising or marketing for the laptop. 
10. trend: This could represent the trendiness or popularity of the laptop model.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge,Lasso
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from feature_engine.outliers import Winsorizer
import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv("C:\Data Set\LassoRidge\Computer_Data.csv")
df.head()

# unnamed col is not import drop it
df.drop("Unnamed: 0",axis=1,inplace=True)

df.head()

df.shape
# rows=6259 and rows=10

df.columns

df.info()


df.isnull().sum()

df.describe()

sns.distplot(df["price"])
plt.show()
# data is right skewed

sns.boxplot(df["price"])
plt.show()
# outliers are present


sns.distplot(df["speed"])
plt.show()
# data is normally distributed

sns.boxplot(df["speed"])
plt.show()

sns.distplot(df["hd"])
plt.show()
# right skewed data


sns.boxplot(df["hd"])
plt.show()
# outliers are present

sns.distplot(df["ram"])
plt.show()
# normally distributed

sns.boxplot(df["ram"])
plt.show()
# outliers are present

sns.distplot(df["screen"])
plt.show()
# normally distributed

sns.boxplot(df["screen"])
plt.show()
# one outliers is present

sns.distplot(df["ads"])
plt.show()
# data is left skewed

sns.boxplot(df["ads"])
plt.show()
# no outliers

#To remove the outlier use winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['price'])
df['price']=winsor.fit_transform(df[['price']])
sns.boxplot(df['price'])
plt.xlabel('price')
plt.show()
#from boxplot we easily see that outliers are removed


#To remove the outlier use winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['hd'])
df['hd']=winsor.fit_transform(df[['hd']])
sns.boxplot(df['hd'])
plt.xlabel('hd')
plt.show()
#from boxplot we easily see that outliers are removed


#To remove the outlier use winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['ram'])
df['ram']=winsor.fit_transform(df[['ram']])
sns.boxplot(df['ram'])
plt.xlabel('ram')
plt.show()
#from boxplot we easily see that outliers are removed


#To remove the outlier use winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['screen'])
df['screen']=winsor.fit_transform(df[['screen']])
sns.boxplot(df['screen'])
plt.xlabel('screen')
plt.show()
#from boxplot we easily see that outliers are removed

df.describe(include="O")

df["cd"].value_counts()
sns.countplot(x=df["cd"])
plt.show()

df["multi"].value_counts()
sns.countplot(x=df["multi"])
plt.show()

df["premium"].value_counts()
sns.countplot(x=df["premium"])
plt.show()

le=LabelEncoder()
df["cd"]=le.fit_transform(df["cd"])
df["multi"]=le.fit_transform(df["multi"])
df["premium"]=le.fit_transform(df["premium"])

df["cd"].value_counts()

df.head()

# split the data
x=df.drop(["price"],axis=1)
y=df["price"]


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

x_train.shape


y_train.shape


model=LinearRegression()
model.fit(x_train,y_train)


y_pred=model.predict(x_test)

mse=mean_squared_error(y_test,y_pred)
print(mse)

r2=r2_score(y_test,y_pred)
print(r2)



model.score(x_test,y_test)

model.score(x_train,y_train)


ridge_alpha=0.02
lasso_alpha=0.01

ridge=Ridge(alpha=ridge_alpha)
lasso=Lasso(alpha=lasso_alpha)


ridge.fit(x_train,y_train)
lasso.fit(x_train,y_train)

ridge_pred=model.predict(x_test)
lasso_pred=model.predict(x_test)


r2_ridge=r2_score(y_test,ridge_pred)
print(r2_ridge)

r2_lasso=r2_score(y_test,lasso_pred)
print(r2_lasso)

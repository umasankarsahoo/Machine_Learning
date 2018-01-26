# Load libraries
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import seaborn as sns
import datetime
from string import ascii_letters

# Load df
# Load df
url1 = "/Users/...../DS1.csv"
url2 = "/Users/...../DS2.csv"
url3 = "/Users/...../DS3.csv"
url4 = "/Users/...../DS4.csv"
url5 = "/Users/...../DS5.csv"
url6 = "/Users/...../DS6.csv"

df1=pd.read_csv(url1)
df2=pd.read_csv(url2)
df3=pd.read_csv(url3)
df4=pd.read_csv(url4)
df5=pd.read_csv(url5)
df6=pd.read_csv(url6)

print(df1.info())
print(df2.info())
print(df3.info())
print(df4.info())
print(df5.info())
print(df6.info())

print(df1.describe())
print(df2.describe())
print(df3.describe())
print(df4.describe())
print(df5.describe())
print(df6.describe())


#print('1.   RO RANDOM W8-1.xlsx	   :  ',pandas.read_csv(url1).shape)
#print('2.   RO RANDOM W8-2.xlsx	   :  ',pandas.read_csv(url2).shape)
#print('3.   RO RANDOM W9-1.xlsx	   :  ',pandas.read_csv(url3).shape)
#print('4.   RO RANDOM W9-2.xlsx	   :  ',pandas.read_csv(url4).shape)
#print('5.   RO RANDOM W10-1.xlsx    :  ',pandas.read_csv(url5).shape)
#print('6.   RO RANDOM W10-2.xlsx    :  ',pandas.read_csv(url6).shape)

#to_delete = ['no','id_outlet','SERIAL','id_auditor','Date','Period']
#df3 = df3.drop(to_delete, axis=1)
#df3=df3.iloc[0:50000]

'''
plt.figure(figsize=(12,5))

plt.hist(df1['Total Selling Price'])
plt.xlabel('Total Selling Price')
plt.ylabel('Frequency')
plt.title('Non gaussian histogram plot')


plt.show()
plt.hist(np.log(df1['Total Selling Price']))
plt.xlabel('Total Selling Price')
plt.ylabel('Frequency')
plt.title('Gaussian histogram plot')

plt.show()

corr = df3.select_dtypes(include = ['float64', 'int64']).iloc[:,1:].corr()
sns.set(font_scale=1)
sns.heatmap(corr, vmax=1, square=True)
#plt.show()

corr_list = corr['Total Selling Price'].sort_values(axis=0,ascending=False).iloc[1:]
#print(corr_list)

plt.figure(figsize=(18,8))
for i in range(6):
    ii = '23'+str(i+1)
    plt.subplot(ii)
    feature = corr_list.index.values[i]
    plt.scatter(df3[feature], df3['Total Selling Price'], facecolors='red',edgecolors='k',s = 75)
    sns.regplot(x = feature, y = 'Total Selling Price', data = df3,scatter=False, color = 'Blue')
    ax=plt.gca()
    ax.set_ylim([0,100])

#plt.show()
plt.figure(figsize=(18,8))

df3.plot(kind='box', subplots=True, layout=(5,5), sharex=False, sharey=False)
plt.show()


'''
tables = [df3,df6]
print ("Delete features with high number of missing values...")
total_missing = df3.isnull().sum()
to_delete = total_missing[total_missing>(df3.shape[0]/3.)]
for table in tables:
    table.drop(list(to_delete.index),axis=1,inplace=True)

numerical_features = df6.select_dtypes(include=["float","int","bool"]).columns.values
categorical_features = df3.select_dtypes(include=["object"]).columns.values

print(to_delete)

'''

origin = pd.DataFrame(df1['Total Selling Price'])
dif = np.abs(df6-origin) > 5000
idx = dif[dif['Total Selling Price']].index.tolist()
df1.drop(df1.index[idx],inplace=True)
print(df1.shape)
print(idx)

'''
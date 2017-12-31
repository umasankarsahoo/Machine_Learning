Kaggle House Prices: Advanced Regression Techniques

House Prices: Advanced Regression Techniques Competition on Kaggle

Install

This project requires Python 2.7 and the following Python libraries installed:

NumPy
matplotlib
seaborn
scikit-learn
XGBoost
You will also need to have software installed to run and execute an iPython Notebook

Code

All ipython notebook are used for data preprocessing, feature transforming and outlier detecting. All core scripts are in code folder, in which the ensemble learning script is in ensemble folder and base model script is in sing_model folder. All input data are in input folder and the detailed description of the data can be found in Kaggle.

Run

For a single model run, navigate to the /code/single_model/ and run the following commands: python base_model.py For a ensemble run, navigate to the /code/ensemble/ and run the following commands: python ensemble.py Make sure to change the data directory and the parameters accordingly before the model run.

Submission

Submission score on Kaggle leaderboard with different approaches.

FlowChart

Flow chart of the code.

Documentation

=========================================== Exploratory_data_analysis-checkpoint.ipynb ===========================================

# This file provide a basic exploration of ames house price dataset
import numpy as np 
import pandas as pd

df = pd.read_csv('/Users/umasankarsahoo/Desktop/PyML/Datasets/house_price_pred_trainingdata.csv')
df.head()
df.describe()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Set up the matplotlib figure
plt.figure(figsize=(12,5))
#f, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
plt.subplot(121)
sns.distplot(df['SalePrice'],kde=False)
plt.xlabel('Sale price')
plt.axis([0,800000,0,180])
plt.subplot(122)
sns.distplot(np.log(df['SalePrice']),kde=False)
plt.xlabel('Log (sale price)')
plt.axis([10,14,0,180])

corr = df.select_dtypes(include = ['float64', 'int64']).iloc[:,1:].corr()
#fig = plt.figure()
sns.set(font_scale=1)  
sns.heatmap(corr, vmax=1, square=True)

corr_list = corr['SalePrice'].sort_values(axis=0,ascending=False).iloc[1:]
corr_list
plt.figure(figsize=(18,8))
for i in range(6):
    ii = '23'+str(i+1)
    plt.subplot(ii)
    feature = corr_list.index.values[i]
    plt.scatter(df[feature], df['SalePrice'], facecolors='none',edgecolors='k',s = 75)
    sns.regplot(x = feature, y = 'SalePrice', data = df,scatter=False, color = 'Blue')
    ax=plt.gca() 
    ax.set_ylim([0,800000])

plt.figure(figsize = (12, 6))
sns.boxplot(x = 'Neighborhood', y = 'SalePrice',  data = df)
xt = plt.xticks(rotation=45)


=========================================== feature_selection-checkpoint.ipynb ===========================================

import numpy as np
import pandas as pd
import datetime
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
import time
from sklearn import preprocessing
from scipy.stats import skew

train = pd.read_csv("/Users/umasankarsahoo/Desktop/PyML/Datasets/house_price_pred_trainingdata.csv") # read train data
test = pd.read_csv("/Users/umasankarsahoo/Desktop/PyML/Datasets/house_price_pred_testdata.csv") # read test data

tables = [train,test]
print ("Delete features with high number of missing values...")
total_missing = train.isnull().sum()
to_delete = total_missing[total_missing>(train.shape[0]/3.)]
for table in tables:
    table.drop(list(to_delete.index),axis=1,inplace=True)

numerical_features = test.select_dtypes(include=["float","int","bool"]).columns.values
categorical_features = train.select_dtypes(include=["object"]).columns.values

to_delete

Alley          1369
FireplaceQu     690
PoolQC         1453
Fence          1179
MiscFeature    1406
dtype: int64



=========================================== outlier_detection.ipynb ===========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn import svm
from sklearn.covariance import EllipticEnvelope
%matplotlib inline


train = pd.read_csv("/Users/umasankarsahoo/Desktop/PyML/Datasets/house_price_pred_trainingdata.csv") 
test = pd.read_csv("prediction_training.csv").drop('Id',axis=1,inplace=False)
origin = pd.DataFrame(train['SalePrice'])

dif = np.abs(test-origin) > 12000

idx = dif[dif['SalePrice']].index.tolist()

train.drop(train.index[idx],inplace=True)
train.shape
idx

=========================================== Score_with_different_approaches-checkpoint.ipynb ===========================================

import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

score = np.array([0.15702,0.14831,0.15233,0.12856,0.12815,0.12459,0.11696])

plt.figure()
plt.plot(score,'o-')
plt.ylabel('Leaderboard score',fontsize=16)
ax = plt.gca()
ax.set_xlim([-0.5,6.3])
ax.set_xticks([i for i in range(7)])
ax.set_xticklabels(['Untuned Random Forest', 'Tuned Random Forest', 'Tuned Extra Trees', 'Tuned Gradient Boosting'\
                    , 'Tuned XGBoost','Stacking 4 best model','Stacking more models'],fontsize=12)
plt.grid()
plt.xticks(rotation=90)


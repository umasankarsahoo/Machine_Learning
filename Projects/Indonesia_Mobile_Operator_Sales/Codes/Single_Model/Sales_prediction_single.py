import numpy as np
import pandas as pd
import datetime
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
import time
from sklearn import preprocessing
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.linear_model import Ridge, LassoCV, LassoLarsCV, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from scipy.stats import skew
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor


def create_submission(prediction, score):
    now = datetime.datetime.now()
    sub_file = '/Users/umasankarsahoo/Desktop/PyML/Housing_price_prediction/submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    # sub_file = 'prediction_training.csv'
    print ('Creating submission: ', sub_file)
    pd.DataFrame({'Id': test['Id'].values, 'SalePrice': prediction}).to_csv(sub_file, index=False)


# train need to be test when do test prediction
def data_preprocess(train, test):
    all_data = pd.concat((train.loc[:, 'Outlet Size':'Selling price'],
                          test.loc[:, 'Outlet Size':'Selling price']))

    to_delete = ['Date']
    all_data = all_data.drop(to_delete, axis=1)
#    all_data["Type of Data/SMS"] = all_data["Type of Data/SMS"].map({None: 0, "GB": 1, "Gb": 2, "MB": 3, "SMS (dalam ribuan)": 4,"gb": 5,"mb": 6}).astype(int)

    train["Total Selling Price"] = np.log1p(train["Total Selling Price"])
    # log transform skewed numeric features
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))  # compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index
    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
    all_data = pd.get_dummies(all_data)
    all_data = all_data.fillna(all_data.mean())
    all_data['Nominal of Data/SMS'] = all_data['Nominal of Data/SMS'].fillna(0)
    X_train = all_data[:train.shape[0]]
    X_test = all_data[train.shape[0]:]
    y = train.loc[:, 'Total Selling Price']

    return X_train, X_test, y


def mean_squared_error_(ground_truth, predictions):
    return mean_squared_error(ground_truth, predictions) ** 0.5


RMSE = make_scorer(mean_squared_error_, greater_is_better=False)


def model_random_forecast(Xtrain,Xtest,ytrain):
    
    X_train = Xtrain
    y_train = ytrain
    rfr = RandomForestRegressor(n_jobs=1, random_state=0)
    param_grid = {}#'n_estimators': [500], 'max_features': [10,15,20,25], 'max_depth':[3,5,7,9,11]}
    model = GridSearchCV(estimator=rfr, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
    model.fit(X_train, y_train)
    print('Random forecast regression...')
    print('Best CV Score:')
    print(-model.best_score_)

    y_pred = model.predict(Xtest)
    return y_pred, -model.best_score_

def model_gradient_boosting_tree(Xtrain,Xtest,ytrain):
    
    X_train = Xtrain
    y_train = ytrain 
    gbr = GradientBoostingRegressor(random_state=0)
    param_grid = {
 #       'n_estimators': [500],
 #       'max_features': [10,15],
#	'max_depth': [6,8,10],
 #       'learning_rate': [0.05,0.1,0.15],
  #      'subsample': [0.8]
    }
    model = GridSearchCV(estimator=gbr, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
    model.fit(X_train, y_train)
    print('Gradient boosted tree regression...')
    print('Best CV Score:')
    print(-model.best_score_)

    y_pred = model.predict(Xtest)
    return y_pred, -model.best_score_

def model_xgb_regression(Xtrain,Xtest,ytrain):
    
    X_train = Xtrain
    y_train = ytrain 
    
    xgbreg = xgb.XGBRegressor(seed=0)
    param_grid = {
#        'n_estimators': [500],
#        'learning_rate': [ 0.05],
#        'max_depth': [ 7, 9, 11],
#        'subsample': [ 0.8],
#        'colsample_bytree': [0.75,0.8,0.85],
    }
    model = GridSearchCV(estimator=xgbreg, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
    model.fit(X_train, y_train)
    print('eXtreme Gradient Boosting regression...')
    print('Best CV Score:')
    print(-model.best_score_)

    y_pred = model.predict(Xtest)
    return y_pred, -model.best_score_

def model_extra_trees_regression(Xtrain,Xtest,ytrain):
    
    X_train = Xtrain
    y_train = ytrain
    
    etr = ExtraTreesRegressor(n_jobs=1, random_state=0)
    param_grid = {}#'n_estimators': [500], 'max_features': [10,15,20]}
    model = GridSearchCV(estimator=etr, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
    model.fit(X_train, y_train)
    print('Extra trees regression...')
    print('Best CV Score:')
    print(-model.best_score_)

    y_pred = model.predict(Xtest)
    return y_pred, -model.best_score_


train = pd.read_csv("/Users/umasankarsahoo/Desktop/PyML/Stay_Top_Mobile_Operator_Sales/DS3.csv")
test = pd.read_csv("/Users/umasankarsahoo/Desktop/PyML/Stay_Top_Mobile_Operator_Sales/DS6.csv")

Xtrain,Xtest,ytrain = data_preprocess(train.iloc[0:100000], test.iloc[0:100000])


test_predict,score = model_random_forecast(Xtrain,Xtest,ytrain)
test_predict,score = model_xgb_regression(Xtrain,Xtest,ytrain)
test_predict,score = model_extra_trees_regression(Xtrain,Xtest,ytrain)
test_predict,score = model_gradient_boosting_tree(Xtrain,Xtest,ytrain)





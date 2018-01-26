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


class ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, train, test , ytr):
        X = train.values
        y = ytr.values
        T = test.values
        folds = list(KFold(len(y), n_folds=self.n_folds, shuffle=True, random_state=0))
        #print(folds)
        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, reg in enumerate(base_models):
            print ("Fitting the base model...")
            S_test_i = np.zeros((T.shape[0], len(folds)))
            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                reg.fit(X_train, y_train)
                y_pred = reg.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = reg.predict(T)[:]
                S_test[:, i] = S_test_i.mean(1)
        print ("Stacking base models...")
        # tuning the stacker
        param_grid = {
        'alpha': [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 0.2, 0.3, 0.4, 0.5, 0.8, 1e0, 3, 5, 7, 1e1],
        }
        grid = GridSearchCV(estimator=self.stacker, param_grid=param_grid, n_jobs=1, cv=5, scoring=RMSE)
        grid.fit(S_train, y)
        try:
            print('Param grid:')
            print(param_grid)
            print('Best Params:')
            print(grid.best_params_)
            print('Best CV Score:')
            print(-grid.best_score_)
            print('Best estimator:')
            print(grid.best_estimator_)
            print(message)
        except:
            pass
        y_pred = grid.predict(S_test)[:]
        return y_pred,grid.best_score_


train = pd.read_csv("/Users/umasankarsahoo/Desktop/PyML/Stay_Top_Mobile_Operator_Sales/DS3.csv")
test = pd.read_csv("/Users/umasankarsahoo/Desktop/PyML/Stay_Top_Mobile_Operator_Sales/DS6.csv")

# build a model library (can be improved)
base_models = [
    RandomForestRegressor(
        n_jobs=1, random_state=0,
        n_estimators=500, max_features=8
    ),
    RandomForestRegressor(
        n_jobs=1, random_state=0,
        n_estimators=500, max_features=12,
        max_depth=7
    ),
    ExtraTreesRegressor(
        n_jobs=1, random_state=0,
        n_estimators=500, max_features=8
    ),
    ExtraTreesRegressor(
        n_jobs=1, random_state=0,
        n_estimators=500, max_features=12
    ),
    GradientBoostingRegressor(
        random_state=0,
        n_estimators=500, max_features=8, max_depth=6,
        learning_rate=0.05, subsample=0.8
    ),
    GradientBoostingRegressor(
        random_state=0,
        n_estimators=500, max_features=12, max_depth=6,
        learning_rate=0.05, subsample=0.8
    ),
    XGBRegressor(
        seed=0,
        n_estimators=500, max_depth=10,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.75
    ),

    XGBRegressor(
        seed=0,
        n_estimators=500, max_depth=7,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.75
    ),
    LassoCV(alphas=[1, 0.1, 0.001, 0.0005]),
    KNeighborsRegressor(n_neighbors=5),
    KNeighborsRegressor(n_neighbors=10),
    KNeighborsRegressor(n_neighbors=15),
    KNeighborsRegressor(n_neighbors=25),
    LassoLarsCV(),
    ElasticNet(),
    SVR()
]

ensem = ensemble(
    n_folds=5,
    stacker=Ridge(),
    base_models=base_models
)

X_train, X_test, y_train = data_preprocess(train.iloc[0:5000], test.iloc[0:5000])
y_pred, score = ensem.fit_predict(X_train, X_test, y_train)






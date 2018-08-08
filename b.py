import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.cross_validation import KFold
import xgboost as xgb
import numpy as np
import scipy.sparse as sp
import pandas as pd


y=np.load('all_label.npy')
df=pd.read_csv('categorical_data.csv')
df.reset_index()
features=df.columns
df['loss']=y
X = df[features].as_matrix()
y =df['loss'].as_matrix()
d = xgb.DMatrix(X, y)
bst = xgb.Booster({'nthread': 4})
bst.load_model('0001.model')
ypred = bst.predict(d)
print(ypred)
print(y)
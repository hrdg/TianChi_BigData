import numpy as np
import time
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

df_id_train = pd.read_csv('official-data/df_id_train.csv', index_col=0, names=['个人编码', 'label'])
df_id_test = pd.read_csv('official-data/df_id_test.csv', index_col=0, names=['个人编码', 'label'])

df_train = pd.read_csv('lin-train.csv', index_col='个人编码')
df_test = pd.read_csv('lin-test.csv', index_col='个人编码')
df_train.drop(['医院编码'], axis=1, inplace=True)
df_test.drop(['医院编码'], axis=1, inplace=True)

X = df_train.copy()
# print(X.info())
y = df_id_train.sort_index()

seed = 42
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# 用xgboost的Python API而非更上层的SciKit封装来训练

dtrain = xgb.DMatrix(X, label=y)
dtest = xgb.DMatrix(X_test, label=y_test)

param = {'max_depth': 4, 'eta': 0.08, 'silent': 1, 'objective': 'binary:logistic', 'min_child_weight':20,
         'scale_pos_weight' : 5}
param['nthread'] = 4
param['eval_metric'] = 'auc'
plst = param.items()

evallist = [(dtrain, 'train'), (dtest, 'eval')]
eval_set = [(dtrain, 'train'), (dtest, 'fuckyou')]

# num_round = 300
# bst = xgb.train(plst, dtrain, num_round, eval_set)
# y_pred = bst.predict(dtest)



def fpreproc(dtrain, dtest, param):
    label = dtrain.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label==1)
    param['scale_pos_weight'] = ratio
    print(ratio)
    return (dtrain, dtest, param)


def f1_zengjinjie(y, t):
    t = t.get_label()
    y_bin = [1. if y_cont > 0.5 else 0. for y_cont in y]
    return 'f1', -1 * f1_score(t, y_bin)

num_round = 300
start = time.clock()
#  feval=f1_zengjinjie
# fpreproc=fpreproc, 这个才是导致f1波动的最终原因
# metrics='auc', stratified=True,
res = xgb.cv(param, dtrain, num_round, nfold=5,
             seed=0,
              feval=f1_zengjinjie,
             callbacks=[xgb.callback.print_evaluation(show_stdv=True),
                        xgb.callback.early_stop(50)])
end = time.clock()

print('cv运行时间为：', end - start)
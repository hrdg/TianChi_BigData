# -*- coding: utf-8 -*-
import pandas as pd
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from xgboost import plot_importance
from matplotlib import pyplot
pd.set_option('max_columns',100)
#导入数据文件
dataset = pd.read_csv("lin-train.csv", index_col='个人编码')
dataset = dataset.drop(["医院编码"], axis=1)
#dataset = dataset.drop(["医院编码标准差"], axis=1)
#dataset = dataset.drop(["医院得分"], axis=1)
hospital = pd.read_csv("医院编码标准差训练集.csv", index_col='个人编码')
dataset["医院编码标准差"] = hospital["医院编码标准差"]
sanmu  = pd.read_csv("三目求和训练集.csv", index_col='个人编码')
dataset["三目各项和"] = sanmu["三目1"] + sanmu["三目2"] + sanmu["三目3"] + sanmu["三目6"]
dataset["药品费占比"] = (dataset['药品费发生金额'])/(dataset['药品费发生金额'] + dataset['检查费发生金额'] + dataset['治疗费发生金额'] + dataset['手术费发生金额']+dataset['床位费发生金额'] + dataset['医用材料发生金额'] + dataset['其它发生金额'])
# illtype = pd.read_csv("出院诊断病例调换门特挂号顺序.csv", index_col='个人编码')
# dataset["出院诊断病例"] = illtype["出院诊断病例"]
# ill = pd.read_csv("几种病训练集.csv", index_col='个人编码')
# dataset["几种病"] = ill["sum"]
sanmugeshu = pd.read_csv('三目服务项目个数改训练集.csv', index_col='个人编码')
dataset['三目服务项目个数'] = sanmugeshu['三目服务项目个数']

# yaopinfangcha1 = pd.read_csv('df_train_sum_std(1).csv', index_col='个人编码')
# dataset['药品费月总发生金额方差'] = yaopinfangcha1['药品费月总发生金额方差']
# yaopinfangcha2 = pd.read_csv('df_train_ave_std(1).csv', index_col='个人编码')
# dataset['药品费月平均发生金额方差'] = yaopinfangcha2['药品费月平均发生金额方差']
yaopinshuliang = pd.read_csv('药品数量求和训练集.csv', index_col='个人编码')
dataset['药品数量'] = yaopinshuliang['药品数量']
X = dataset.copy()
dataset = pd.read_csv('data\df_id_train.csv', index_col='id')
Y = dataset.copy()

dataset = pd.read_csv('lin-test.csv', index_col='个人编码')
dataset = dataset.drop(["医院编码"], axis=1)
#dataset = dataset.drop(["医院编码标准差"], axis=1)
#dataset = dataset.drop(["医院得分"], axis=1)
hospital = pd.read_csv("医院编码标准差测试集.csv", index_col='个人编码')
dataset["医院编码标准差"] = hospital["医院编码标准差"]
sanmu  = pd.read_csv("三目求和测试集.csv", index_col='个人编码')
dataset["三目各项和"] = sanmu["三目1"] + sanmu["三目2"] + sanmu["三目3"]+ sanmu["三目6"]
dataset["药品费占比"] = (dataset['药品费发生金额'])/(dataset['药品费发生金额'] + dataset['检查费发生金额'] + dataset['治疗费发生金额'] + dataset['手术费发生金额']+dataset['床位费发生金额'] + dataset['医用材料发生金额'] + dataset['其它发生金额'])
# ill = pd.read_csv("几种病测试集.csv", index_col='个人编码')
# dataset["几种病"] = ill["sum"]
sanmugeshu = pd.read_csv('三目服务项目个数改测试集.csv', index_col='个人编码')
dataset['三目服务项目个数'] = sanmugeshu['三目服务项目个数']

# yaopinfangcha1 = pd.read_csv('df_test_sum_std(1).csv', index_col='个人编码')
# dataset['药品费月总发生金额方差'] = yaopinfangcha1['药品费月总发生金额方差']
# yaopinfangcha2 = pd.read_csv('df_test_ave_std(1).csv', index_col='个人编码')
# dataset['药品费月平均发生金额方差'] = yaopinfangcha2['药品费月平均发生金额方差']
yaopinshuliang = pd.read_csv('药品数量求和测试集.csv', index_col='个人编码')
dataset['药品数量'] = yaopinshuliang['药品数量']
X_real_test = dataset.copy()

seed = 42
test_size = 0.20
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
#训练分类器，并用GridSearch调参
# param_grid = {'scale_pos_weight': [3], 'learning_rate':[0.2], 'n_estimators':[200], 'min_child_weight':[20], 'max_depth':[3]}
# model_xgb = XGBClassifier()
# grid_search = GridSearchCV(model_xgb, param_grid, cv=5, scoring='f1')
# grid_search.fit(X , Y['yon'])
# print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))


model_xgb = XGBClassifier(scale_pos_weight=3, learning_rate=0.2, n_estimators=200, min_child_weight=20, max_depth=3, base_score=0.5, gamma=0, n_jobs=4)
# eval = [(X_test, y_test)]
# model_xgb.fit(X_train, y_train, eval_set=eval, eval_metric='auc', early_stopping_rounds=20, verbose=True)
model_xgb.fit(X, Y)
y_pred= model_xgb.predict(X_test)

p = precision_score(y_test, y_pred, average='binary')
r = recall_score(y_test, y_pred, average='binary')
f1score = f1_score(y_test, y_pred, average='binary')
print(p)
print(r)
print(f1score)

plot_importance(model_xgb, importance_type='gain')
pyplot.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
pyplot.rcParams['axes.unicode_minus'] = False
pyplot.show()

# y_real_pred = model_xgb.predict(X_real_test)
# y_real_pred = grid_search.predict(X_real_test)
# y_real_id = pd.read_csv('data\df_id_test.csv', index_col=0, names=["个人编码", "result"])
# temp = pd.DataFrame({"result": y_real_pred}, index=X_real_test.index)
# print(temp["result"].sum())
# y_real_id["result"] = temp["result"]
# y_real_id.to_csv('b.csv', header=False)

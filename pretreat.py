# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
pd.set_option('max_columns',100)
# data = pd.read_csv("data\df_train.csv", index_col='个人编码')
# datadrop = data.drop(["顺序号", "住院开始时间", "住院终止时间", "申报受理时间", "操作时间", "药品费拒付金额", "检查费拒付金额", "治疗费拒付金额", "手术费拒付金额", "床位费拒付金额", "医用材料费拒付金额", "输全血申报金额", "成分输血自费金额", "成分输血拒付金额", "其它拒付金额", "一次性医用材料自费金额", "一次性医用材料拒付金额", "输全血按比例自负金额", "统筹拒付金额", "农民工医疗救助计算金额","双笔退费标识", "住院天数","非典补助补助金额","家床起付线剩余"], axis=1)

# 将医院编码用onehot重新编码后，计算一个人医院编码的标准差
# pid = pd.read_csv("data\df_id_test.csv")
# var = 0
# for i in range(0,4000):
#     piddata = pid['id']
#     print(piddata.ix[i])
#     dataid = datadrop.ix[ [ piddata.ix[i] ] ]
#     encoder = OneHotEncoder()
#     dataid_array = np.array(dataid["医院编码"].values)
#     dataid_onehot = encoder.fit_transform(dataid_array.reshape(-1,1))
#     dataid_toarray = np.array(dataid_onehot.toarray())
#     mean = dataid_toarray.mean(axis=0)
#     for j in range(0, dataid["医院编码"].count()):
#         var = var + ((dataid_toarray[j] - mean)**2).sum()
#     var = var/dataid["医院编码"].count()
#     std = math.sqrt(var)
#     dl = {"医院编码标准差": std }
#     tem = pd.DataFrame(dl, index=[ piddata.ix[i] ])
#     tem.to_csv('a.csv', sep=',', mode='a', header=False)
#     var = 0

# # 根据一个医院产生的所有就医记录中诈骗人员产生就医记录所占的比例的10倍作为该医院的评分，计算一个人产生的所有就医记录中对应医院的评分的和
# m = datadrop["医院编码"].value_counts()
# dataset = pd.read_csv("cheat_infor.csv", index_col='个人编码')
# n = dataset["医院编码"].value_counts()
# res = n/m
# res = res.fillna(0)
# pid = pd.read_csv("data\df_id_train.csv")
# score = 0
# for i in range(0,20000):
#     piddata = pid['id']
#     print(piddata.ix[i])
#     dataid = datadrop.ix[[ piddata.ix[i] ]]
#     vm = dataid['医院编码'].values
#     for j in vm:
#         if j not in res.index:
#             score = score + 0
#         else:
#             score = score + res.ix[j] * 10
#     dl = {'去过医院总得分': score}
#     tem = pd.DataFrame(dl, index=[ piddata.ix[i] ])
#     tem.to_csv('a.csv', sep=',', mode='a', header=False)
#     score = 0

#提取差欺骗人就医信息
# tem = datadrop.ix[352120001523108]
# tem.to_csv('cheat_infor.csv', sep=',')
# #id = pd.read_csv("data\df_id_train.csv", index_col='yon')
# #id = id.ix[1]
# #id.to_csv('cheat_id.csv', index=False, sep=',')
# id = pd.read_csv("cheat_id.csv")
# for i in range(1, 1000):
#     print(id.ix[i])
#     tem = datadrop.ix[ id.ix[i] ]
#     tem.to_csv('cheat_infor.csv', sep=',', header=False, mode='a')
#
# 统计一个人一共去过几家医院，然后除以就医次数，反应医院的更换频率
# pid = pd.read_csv("data\df_id_train.csv")
# for i in range(0,20000):
#     piddata = pid['id']
#     print(piddata.ix[i])
#     dataid = datadrop.ix[ [ piddata.ix[i] ] ]
#     m = dataid['医院编码'].value_counts().count()
#     n = dataid['医院编码'].count()
#     dl = {'去过几家不同的医院': m/n }
#     tem = pd.DataFrame(dl, index=[ piddata.ix[i] ])
#     tem.to_csv('a.csv', sep=',', mode='a', header=False)

'''
#将交易时间和出院诊断病种名称两个文字属性转换为离散数值
#统计每个人的出院病种名称
encoder = LabelEncoder()
#datadrop['交易时间'] = encoder.fit_transform(datadrop['交易时间'])
#datadrop['出院诊断病种名称'] = encoder.fit_transform( datadrop['出院诊断病种名称'].factorize()[0] )
datadrop['出院诊断病种名称'] = datadrop['出院诊断病种名称'].fillna(0)
illtype = ['肺心病', '心脏病', '尿毒症', '精神病', '糖尿病', '偏瘫', '肾', '门特挂号', '挂号']

for i in range(0, 1830386):
    str = datadrop['出院诊断病种名称'].values[i]
    if(str == 0):
        datadrop['出院诊断病种名称'].values[i] = 0
        continue
    for j in range(0, 10):
        if(j == 9):
            break
        str_type = illtype[j]
        result = str.find(str_type)
        if(result >= 0):
            datadrop['出院诊断病种名称'].values[i] = 9-j
            break
    if (j==9):
        datadrop['出院诊断病种名称'].values[i] = 9 - j
print(datadrop['出院诊断病种名称'])


pid = pd.read_csv("data\df_id_train.csv")
piddata = pid['id']
dl = []
for i in range(0,20000):
    print(i)
    dataid = datadrop.ix[ [ piddata.ix[i] ] ]
    first = dataid['出院诊断病种名称'].value_counts().index[0]
    if(dataid['出院诊断病种名称'].value_counts().count() == 1):
        dl.append(first)
    else:
        if( first != 0 ):
            dl.append(first)
        else:
            second = dataid['出院诊断病种名称'].value_counts().index[1]
            dl.append(second)
tem = pd.DataFrame({'出院诊断病例': dl}, index=pid['id'].values)
tem.to_csv('a.csv', sep=',')

#以人为索引，统计一个人所有的就医信息，其中医院编码，交易时间和出院诊断病种名称求标准差，其余为求和，个人编码为352120002836161的人暂时删除
dataid = datadrop.ix[352120001523108]
dl = {'就医次数':dataid.index.value_counts()[352120001523108], '医院编码标准差': dataid['医院编码'].std(), '药品费发生金额':dataid['药品费发生金额'].mean(),'贵重药品发生金额':dataid['贵重药品发生金额'].mean(), '中成药费发生金额':dataid['中成药费发生金额'].mean(), '中草药费发生金额':dataid['中草药费发生金额'].mean(),'药品费自费金额':dataid[ '药品费自费金额'].mean(), '药品费申报金额':dataid['药品费申报金额'].mean(), '检查费发生金额':dataid['检查费发生金额'].mean(),'贵重检查费金额':dataid[ '贵重检查费金额'].mean(), '检查费自费金额':dataid['检查费自费金额'].mean(), '检查费申报金额':dataid['检查费申报金额'].mean(),'治疗费发生金额':dataid['治疗费发生金额'].mean(), '治疗费自费金额':dataid['治疗费自费金额'].mean(), '治疗费申报金额':dataid['治疗费申报金额'].mean(), '手术费发生金额':dataid[ '手术费发生金额'].mean(),'手术费自费金额':dataid['手术费自费金额'].mean(), '手术费申报金额':dataid['手术费申报金额'].mean(),'床位费发生金额':dataid['床位费发生金额'].mean(),'床位费申报金额':dataid['床位费申报金额'].mean(), '医用材料发生金额':dataid[ '医用材料发生金额'].mean(),'高价材料发生金额':dataid['高价材料发生金额'].mean(),'医用材料费自费金额':dataid['医用材料费自费金额'].mean(), '成分输血申报金额':dataid['成分输血申报金额'].mean(),'其它发生金额':dataid['其它发生金额'].mean(),'其它申报金额':dataid['其它申报金额'].mean(), '一次性医用材料申报金额':dataid[ '一次性医用材料申报金额'].mean(), '起付线标准金额':dataid['起付线标准金额'].mean(),'起付标准以上自负比例金额':dataid['起付标准以上自负比例金额'].mean(), '医疗救助个人按比例负担金额':dataid['医疗救助个人按比例负担金额'].mean(),'最高限额以上金额':dataid['最高限额以上金额'].mean(), '基本医疗保险统筹基金支付金额':dataid['基本医疗保险统筹基金支付金额'].mean(), '交易时间标准差':dataid[ '交易时间'].std(),'公务员医疗补助基金支付金额':dataid['公务员医疗补助基金支付金额'].mean(), '城乡救助补助金额':dataid[ '城乡救助补助金额'].mean(), '可用账户报销金额':dataid['可用账户报销金额'].mean(),'基本医疗保险个人账户支付金额':dataid['基本医疗保险个人账户支付金额'].mean(), '非账户支付金额':dataid['非账户支付金额'].mean(), '出院诊断病种名称标准差':dataid[ '出院诊断病种名称'].std(),'本次审批金额':dataid['本次审批金额'].mean(), '补助审批金额':dataid['补助审批金额'].mean(), '医疗救助医院申请':dataid['医疗救助医院申请'].mean(), '残疾军人医疗补助基金支付金额':dataid['残疾军人医疗补助基金支付金额'].mean(),'民政救助补助金额':dataid['民政救助补助金额'].mean(), '城乡优抚补助金额':dataid['城乡优抚补助金额'].mean()}
tem = pd.DataFrame(dl, index=[352120001523108])
tem.to_csv('df_train_mean.csv', sep=',')
pid = pd.read_csv("data\df_id_train.csv")
for i in range(1, 19999):
    piddata = pid['id']
    print(piddata.ix[i])
    dataid = datadrop.ix[ piddata.ix[i] ]
    dl = {'就医次数': dataid.index.value_counts()[piddata.ix[i]], '医院编码标准差': dataid['医院编码'].std(), '药品费发生金额': dataid['药品费发生金额'].mean(), '贵重药品发生金额': dataid['贵重药品发生金额'].mean(),'中成药费发生金额': dataid['中成药费发生金额'].mean(), '中草药费发生金额': dataid['中草药费发生金额'].mean(),'药品费自费金额': dataid['药品费自费金额'].mean(), '药品费申报金额': dataid['药品费申报金额'].mean(), '检查费发生金额': dataid['检查费发生金额'].mean(),'贵重检查费金额': dataid['贵重检查费金额'].mean(), '检查费自费金额': dataid['检查费自费金额'].mean(), '检查费申报金额': dataid['检查费申报金额'].mean(),'治疗费发生金额': dataid['治疗费发生金额'].mean(), '治疗费自费金额': dataid['治疗费自费金额'].mean(), '治疗费申报金额': dataid['治疗费申报金额'].mean(),'手术费发生金额': dataid['手术费发生金额'].mean(), '手术费自费金额': dataid['手术费自费金额'].mean(), '手术费申报金额': dataid['手术费申报金额'].mean(),'床位费发生金额': dataid['床位费发生金额'].mean(), '床位费申报金额': dataid['床位费申报金额'].mean(),'医用材料发生金额': dataid['医用材料发生金额'].mean(), '高价材料发生金额': dataid['高价材料发生金额'].mean(),'医用材料费自费金额': dataid['医用材料费自费金额'].mean(), '成分输血申报金额': dataid['成分输血申报金额'].mean(),'其它发生金额': dataid['其它发生金额'].mean(), '其它申报金额': dataid['其它申报金额'].mean(),'一次性医用材料申报金额': dataid['一次性医用材料申报金额'].mean(), '起付线标准金额': dataid['起付线标准金额'].mean(),'起付标准以上自负比例金额': dataid['起付标准以上自负比例金额'].mean(), '医疗救助个人按比例负担金额': dataid['医疗救助个人按比例负担金额'].mean(),'最高限额以上金额': dataid['最高限额以上金额'].mean(), '基本医疗保险统筹基金支付金额': dataid['基本医疗保险统筹基金支付金额'].mean(),'交易时间标准差': dataid['交易时间'].std(), '公务员医疗补助基金支付金额': dataid['公务员医疗补助基金支付金额'].mean(),'城乡救助补助金额': dataid['城乡救助补助金额'].mean(), '可用账户报销金额': dataid['可用账户报销金额'].mean(),'基本医疗保险个人账户支付金额': dataid['基本医疗保险个人账户支付金额'].mean(), '非账户支付金额': dataid['非账户支付金额'].mean(),'出院诊断病种名称标准差': dataid['出院诊断病种名称'].std(), '本次审批金额': dataid['本次审批金额'].mean(), '补助审批金额': dataid['补助审批金额'].mean(),'医疗救助医院申请': dataid['医疗救助医院申请'].mean(), '残疾军人医疗补助基金支付金额': dataid['残疾军人医疗补助基金支付金额'].mean(),'民政救助补助金额': dataid['民政救助补助金额'].mean(), '城乡优抚补助金额': dataid['城乡优抚补助金额'].mean()}
    tem = pd.DataFrame(dl, index=[ piddata.ix[i] ])
    tem.to_csv('df_train_mean.csv', sep=',', mode='a', header=False)
'''
# 求三目医院对应标号的和
# column = ['个人编码', '顺序号', '三目1', '三目2', '三目3', '三目4', '三目5', '三目6', '三目7', '三目9']
# csv_reader = pd.read_csv('deal_medicine\df_train_sanmu_bianma.csv', usecols=column)
# df = csv_reader.set_index('个人编码', '顺序号')
# df.sort_index(inplace=True)
# print(df)
# df = df.var(level=0)
# df["三目全部"] = df['三目1'] + df['三目2'] + df['三目3'] + df['三目4'] + df['三目5'] + df['三目6'] + df['三目7'] + df['三目9']
# df.to_csv('a.csv')

#统计三目服务项目个数，不包括三目1,7
column = ['个人编码', '顺序号', '三目2', '三目3', '三目4', '三目5', '三目6', '三目9']
csv_reader = pd.read_csv('deal_medicine\df_test_sanmu_bianma.csv', usecols=column)
tempdata = csv_reader.drop(['个人编码', '顺序号'], axis=1)
dl = []
for i in range(0, 362607):
    print(i)
    if tempdata.ix[i].count()>0:
        m = tempdata.ix[i].count()-tempdata.ix[i].value_counts()[0]
    else:
        m=0
    dl.append(m)
tem = pd.DataFrame({'三目服务项目个数': dl}, index=csv_reader.index)
print(tem)
csv_reader['三目服务项目个数'] = tem['三目服务项目个数']
print("fjg")
df = csv_reader.set_index('个人编码', '顺序号')
df.sort_index(inplace=True)
df = df.sum(level=0)
df.to_csv('a.csv')

# 统计药品费占比，计算每次就医的药品费占比，然后求一个人各次就医药品费占比的平均值
# data = pd.read_csv("data/df_train.csv")
# df = data.set_index('个人编码', '顺序号')
# df.sort_index(inplace=True)
# df["药品费占比"] = (df['药品费发生金额'])/(df['药品费发生金额'] + df['检查费发生金额'] + df['治疗费发生金额'] + df['手术费发生金额'] + df['床位费发生金额'] + df['医用材料发生金额'] + df['其它发生金额'])
# df = df.mean(level=0)
# df['药品费占比'].to_csv('c.csv')

#统计药品数量
# dfid = pd.read_csv("deal_medicine/df_test_sanmu_bianma.csv", index_col='顺序号')
# dfid.sort_index(inplace=True)

# data = pd.read_csv("data/fee_detail.csv", index_col='顺序号')
# data.sort_index(inplace=True)
# data = data[ data.三目统计项目.isin([1]) ]
# data = data.sum(level=0)
# dfid['数量'] = data['数量']
# tem = dfid['数量']
# print(tem)
# tem.to_csv('deal_medicine/df_test_yaopin_shuliang.csv', header=False)

# data = pd.read_csv("deal_medicine/df_test_yaopin_shuliang.csv", index_col='顺序号')
# data.sort_index(inplace=True)
# data['个人编码'] = dfid['个人编码']
# data.to_csv('deal_medicine/df_test_yaopin_shuliang.csv')

#将药品数量按个人合并在一起，求和或按就医次数取平均平均值
# data = pd.read_csv("deal_medicine/df_test_yaopin_shuliang.csv")
# df = data.set_index('个人编码', '顺序号')
# df.sort_index(inplace=True)
# df = df.sum(level=0)
# df.to_csv('药品数量求和测试集.csv')

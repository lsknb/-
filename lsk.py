import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from matplotlib.pyplot import plot, show
from sklearn.metrics import roc_auc_score
import json
from gensim.models.word2vec import Word2Vec

# 合并数据集
df_train = pd.read_csv('data/train_dataset.csv', sep='\t')  # 读取训练样本
df_test = pd.read_csv('data/test_dataset.csv', sep='\t')  # 读取测试样本
# 在训练集中，共有15016条数据，其中每条数据有19个特征
# 在测试集中，共有9999条数据，其中每条数据有18个特征
sub = pd.read_csv('data/submit_example.csv')
df_test['id'] = sub['id']
df = pd.concat([df_train, df_test])  # 列对其合并数据集
print(df.dtypes)
# 特征转换，location列转成多列
df['location_first_lvl'] = df['location'].astype(str).apply(lambda x: json.loads(x)['first_lvl'])
df['location_sec_lvl'] = df['location'].astype(str).apply(lambda x: json.loads(x)['sec_lvl'])
df['location_third_lvl'] = df['location'].astype(str).apply(lambda x: json.loads(x)['third_lvl'])
# 标签编码
feats = ['user_name', 'action', 'auth_type', 'ip_location_type_keyword', 'ip_risk_level', 'ip', 'location',
         'device_model', 'os_type', 'os_version', 'browser_type', 'browser_version', 'bus_system_code', 'op_target',
         'location_first_lvl', 'location_sec_lvl', 'location_third_lvl',
         ]

cat = []

LABEL = 'risk_label'

# 时间的处理
df['sec'] = df['session_id'].apply(lambda x: int(x[-7:-5]))
# 将时间特征分割为正弦余弦，反应数据循环性
df['sec_sin'] = np.sin(df['sec'] / 60 * 2 * np.pi)
df['sec_cos'] = np.cos(df['sec'] / 60 * 2 * np.pi)
df['op_date'] = pd.to_datetime(df['op_date'])  # 时间格式转换
df['hour'] = df['op_date'].dt.hour  # 添加时间
df['weekday'] = df['op_date'].dt.weekday  # 添加每周的哪一天
df['year'] = df['op_date'].dt.year
df['month'] = df['op_date'].dt.month
df['day'] = df['op_date'].dt.day

df['op_ts'] = df["op_date"].values.astype(np.int64) // 10 ** 9  # 将时间转换为int类型，并取名为op_ts，10 ** 9 转化为纳秒
df = df.sort_values(by=['user_name', 'op_ts']).reset_index(drop=True)  # # 重排数据，根据用户名，日期排序
df['last_ts'] = df.groupby(['user_name'])['op_ts'].shift(1)  # 以用户名做聚合，ts字段下移一位
df['last_ts2'] = df.groupby(['user_name'])['op_ts'].shift(2)  # 以用户名做聚合，ts字段下移二位
df['ts_diff'] = df['op_ts'] - df['last_ts']  # 同一用户第二次操作时间与第一次操作时间的间隔
df['ts_diff2'] = df['op_ts'] - df['last_ts2']  # 同一用户第三次操作时间与第一次操作时间的间隔

feats += ['sec',
          'sec_sin', 'sec_cos',
          'op_ts', 'last_ts', 'ts_diff',
          'ts_diff2',
          ]

# 词嵌入特征
for name in ['auth_type']:
    df[name + '_fillna'] = df[name].astype('str')
    sent = df.groupby(['user_name', 'year', 'month', 'day'])[name + '_fillna'].agg(list).values
    vec_size = 6
    w2v_model = Word2Vec(sentences=sent, vector_size=vec_size, window=12, min_count=1, workers=1)
    tmp = df[name + '_fillna'].map(lambda x: w2v_model.wv[x])
    tmp = pd.DataFrame(list(tmp))
    tmp.columns = ['_'.join([name, 'emb', str(i)]) for i in range(vec_size)]
    df = pd.concat([df, tmp], axis=1)
    feats += list(tmp.columns)

for w in w2v_model.wv.key_to_index:
    print(w, w2v_model.wv[w])

# 生成交叉统计量特征
for name in ['mean', 'std', 'max', 'min', 'median', 'skew']:
    for name1 in ['user_name', 'bus_system_code', 'auth_type', 'action',
                  ]:
        df[name1 + '_ts_diff_' + name] = df.groupby([name1])['ts_diff'].transform(name)
        feats.append(name1 + '_ts_diff_' + name)

df['if_out'] = (df['location'] == '{"first_lvl":"成都分公司","sec_lvl":"9楼","third_lvl":"销售部"}')
feats.append('if_out')

# 标签编码
for name in ['user_name', 'action', 'auth_type', 'ip', 'ip_location_type_keyword', 'ip_risk_level', 'location',
             'device_model', 'os_type', 'os_version', 'browser_type', 'browser_version', 'bus_system_code', 'op_target',
             'location_first_lvl', 'location_sec_lvl', 'location_third_lvl',
             ] + cat:
    # 获取一个LabelEncoder
    le = LabelEncoder()
    # fit_transform训练LabelEncoder并使用训练好的LabelEncoder进行编码
    df[name] = le.fit_transform(df[name])

df_train = df[~df[LABEL].isna()].reset_index(drop=True)
df_test = df[df[LABEL].isna()].reset_index(drop=True)

# 训练模型
params = {
    'learning_rate': 0.08,  # 学习速率
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'binary',  # 目标函数
    'metric': 'auc',  # 评估函数auc
    'verbose': -1,  # 一个整数，表示是否输出中间信息 小于0，则仅仅输出critical 信息
    'seed': 2222,  # 随机数种子
    'n_jobs': -1,
}

print(feats)
print(df_train[feats].shape, df_test[feats].shape)

seeds = [2022]
oof = np.zeros(len(df_train))
importance = 0  # 特征重要性
fold_num = 11
pred_y = pd.DataFrame()
for seed in seeds:  # k折交叉验证 : 在KFold的基础上，加入了分层抽样的思想，使得测试集和训练集有相同的数据分布
    print('############################', seed)
    kf = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=seed)
    # shuffle不会打乱样本顺序，它返回的只是index
    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train[feats], df_train[LABEL])):
        print('-----------', fold)
        train = lgb.Dataset(df_train.loc[train_idx, feats],
                            df_train.loc[train_idx, LABEL])  # 获取lgb训练数据
        val = lgb.Dataset(df_train.loc[val_idx, feats],
                          df_train.loc[val_idx, LABEL])  # 获取lgb验证数据
        model = lgb.train(params, train, valid_sets=val, num_boost_round=10000,
                          early_stopping_rounds=100, verbose_eval=100)  # 100轮内验证集指标不提升就停止迭代

        # 预测数据集
        oof[val_idx] += model.predict(df_train.loc[val_idx, feats]) / len(seeds)  # 保存每一次验证集的预测结果
        pred_y['fold_%d_seed_%d' % (fold, seed)] = model.predict(df_test[feats])  # 预测测试集
        importance += model.feature_importance(importance_type='gain') / fold_num  # 信息增益，获取特征重要程度

df_train['oof'] = oof
score = roc_auc_score(df_train[LABEL], df_train['oof'])  # 计算auc的值
print(score)
# 打印特征字段的重要性排名（前十个）
feats_importance = pd.DataFrame()
feats_importance['name'] = feats
feats_importance['importance'] = importance
print(feats_importance.sort_values('importance', ascending=False)[:10])

sub = pd.read_csv('data/submit_example.csv')

pred_y = pred_y.mean(axis=1)
sub['ret'] = pred_y
# 输出csv文件
sub[['id', 'ret']].to_csv('ans/lsk.csv', index=False)

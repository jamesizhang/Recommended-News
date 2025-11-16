import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import gc, os
import time
from datetime import datetime
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

data_path = '../data_raw/'
save_path = '../temp_results/'
offline = False

# 重新读取数据的时候，发现click_article_id是一个浮点数，所以将其转换成int类型
trn_user_item_feats_df = pd.read_csv(save_path + 'trn_user_item_feats_df.csv')
trn_user_item_feats_df['click_article_id'] = trn_user_item_feats_df['click_article_id'].astype(int)

if offline:
    val_user_item_feats_df = pd.read_csv(save_path + 'val_user_item_feats_df.csv')
    val_user_item_feats_df['click_article_id'] = val_user_item_feats_df['click_article_id'].astype(int)
else:
    val_user_item_feats_df = None

tst_user_item_feats_df = pd.read_csv(save_path + 'tst_user_item_feats_df.csv')
tst_user_item_feats_df['click_article_id'] = tst_user_item_feats_df['click_article_id'].astype(int)

# 做特征的时候为了方便，给测试集也打上了一个无效的标签，这里直接删掉就行
del tst_user_item_feats_df['label']


def submit(recall_df, topk=5, model_name=None):
    recall_df = recall_df.sort_values(by=['user_id', 'pred_score'])
    recall_df['rank'] = recall_df.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')

    # 判断是不是每个用户都有5篇文章及以上
    tmp = recall_df.groupby('user_id').apply(lambda x: x['rank'].max())
    
    # 检查是否有用户少于topk篇文章
    users_with_insufficient_items = tmp[tmp < topk]
    if len(users_with_insufficient_items) > 0:
        print(f'[Warning] {len(users_with_insufficient_items)} users have fewer than {topk} items:')
        print(f'  Min items per user: {tmp.min()}, Max items per user: {tmp.max()}')
        print(f'  Users with < {topk} items: {users_with_insufficient_items.index.tolist()[:10]}...' if len(users_with_insufficient_items) > 10 else f'  Users with < {topk} items: {users_with_insufficient_items.index.tolist()}')
        # 过滤掉少于topk篇文章的用户
        valid_users = tmp[tmp >= topk].index
        recall_df = recall_df[recall_df['user_id'].isin(valid_users)]
        print(f'  Filtered to {len(valid_users)} users with at least {topk} items')
    
    # 重新计算rank（因为可能过滤了用户）
    recall_df['rank'] = recall_df.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')

    del recall_df['pred_score']
    submit = recall_df[recall_df['rank'] <= topk].set_index(['user_id', 'rank']).unstack(-1).reset_index()

    submit.columns = [int(col) if isinstance(col, int) else col for col in submit.columns.droplevel(0)]
    # 按照提交格式定义列名
    submit = submit.rename(columns={'': 'user_id', 1: 'article_1', 2: 'article_2',
                                    3: 'article_3', 4: 'article_4', 5: 'article_5'})

    save_name = save_path + model_name + '_' + datetime.today().strftime('%m-%d') + '.csv'
    submit.to_csv(save_name, index=False, header=True)

# 排序结果归一化
def norm_sim(sim_df, weight=0.0):
    # print(sim_df.head())
    min_sim = sim_df.min()
    max_sim = sim_df.max()
    if max_sim == min_sim:
        sim_df = sim_df.apply(lambda sim: 1.0)
    else:
        sim_df = sim_df.apply(lambda sim: 1.0 * (sim - min_sim) / (max_sim - min_sim))

    sim_df = sim_df.apply(lambda sim: sim + weight)  # plus one
    return sim_df


# 防止中间出错之后重新读取数据
trn_user_item_feats_df_rank_model = trn_user_item_feats_df.copy()

if offline:
    val_user_item_feats_df_rank_model = val_user_item_feats_df.copy()

tst_user_item_feats_df_rank_model = tst_user_item_feats_df.copy()

# 定义特征列
lgb_cols = ['sim0', 'time_diff0', 'word_diff0','sim_max', 'sim_min', 'sim_sum',
            'sim_mean', 'score','click_size', 'time_diff_mean', 'active_level',
            'click_environment','click_deviceGroup', 'click_os', 'click_country',
            'click_region','click_referrer_type', 'user_time_hob1', 'user_time_hob2',
            'words_hbo', 'category_id', 'created_at_ts','words_count']

# 排序模型分组
trn_user_item_feats_df_rank_model.sort_values(by=['user_id'], inplace=True)
g_train = trn_user_item_feats_df_rank_model.groupby(['user_id'], as_index=False).count()["label"].values

if offline:
    val_user_item_feats_df_rank_model.sort_values(by=['user_id'], inplace=True)
    g_val = val_user_item_feats_df_rank_model.groupby(['user_id'], as_index=False).count()["label"].values

# 排序模型定义
lgb_ranker = lgb.LGBMRanker(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
                            max_depth=-1, n_estimators=100, subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
                            learning_rate=0.01, min_child_weight=20, min_gain_to_split=0.0, random_state=2018, n_jobs= 16)

# 排序模型训练
if offline:
    lgb_ranker.fit(
        trn_user_item_feats_df_rank_model[lgb_cols], trn_user_item_feats_df_rank_model['label'], group=g_train,
        eval_set=[(val_user_item_feats_df_rank_model[lgb_cols], val_user_item_feats_df_rank_model['label'])],
        eval_group=[g_val], eval_at=[1, 2, 3, 4, 5], eval_metric=['ndcg'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=10)
        ]
    )
else:
    lgb_ranker.fit(trn_user_item_feats_df[lgb_cols], trn_user_item_feats_df['label'], group=g_train)

# 模型预测（增加快速排查）

# 模型预测
tst_user_item_feats_df['pred_score'] = lgb_ranker.predict(tst_user_item_feats_df[lgb_cols], num_iteration=lgb_ranker.best_iteration_)

# missing_cols = list(set(lgb_cols) - set(tst_user_item_feats_df.columns))
# print('[LGB DEBUG] tst rows:', len(tst_user_item_feats_df))
# print('[LGB DEBUG] lgb_cols size:', len(lgb_cols))
# print('[LGB DEBUG] missing cols:', missing_cols)
# X_tst = tst_user_item_feats_df.loc[:, [c for c in lgb_cols if c in tst_user_item_feats_df.columns]]
# print('[LGB DEBUG] X_tst shape:', X_tst.shape, 'is_empty:', X_tst.empty)
# if X_tst.empty or X_tst.shape[1] == 0:
#     raise ValueError('Input data must be 2 dimensional and non empty. Check recall/features building; missing cols: %s' % missing_cols)

# best_iter = getattr(lgb_ranker, 'best_iteration_', None)
# tst_user_item_feats_df['pred_score'] = lgb_ranker.predict(X_tst, num_iteration=best_iter)

# 将这里的排序结果保存一份，用户后面的模型融合
tst_user_item_feats_df[['user_id', 'click_article_id', 'pred_score']].to_csv(save_path + 'lgb_ranker_score.csv', index=False)

# 预测结果重新排序, 及生成提交结果
rank_results = tst_user_item_feats_df[['user_id', 'click_article_id', 'pred_score']]
rank_results['click_article_id'] = rank_results['click_article_id'].astype(int)
submit(rank_results, topk=5, model_name='lgb_ranker')

# 预测结果重新排序, 及生成提交结果
# rank_results = tst_user_item_feats_df[['user_id', 'click_article_id', 'pred_score']]
# rank_results['click_article_id'] = rank_results['click_article_id'].astype(int)
# submit(rank_results, topk=5, model_name='lgb_ranker')


# 五折交叉验证，这里的五折交叉是以用户为目标进行五折划分
#  这一部分与前面的单独训练和验证是分开的
def get_kfold_users(trn_df, n=5):
    user_ids = trn_df['user_id'].unique()
    user_set = [user_ids[i::n] for i in range(n)]
    return user_set


k_fold = 5
trn_df = trn_user_item_feats_df_rank_model
user_set = get_kfold_users(trn_df, n=k_fold)

score_list = []
score_df = trn_df[['user_id', 'click_article_id', 'label']]
sub_preds = np.zeros(tst_user_item_feats_df_rank_model.shape[0])

# 五折交叉验证，并将中间结果保存用于staking
for n_fold, valid_user in enumerate(user_set):
    train_idx = trn_df[~trn_df['user_id'].isin(valid_user)]  # add slide user
    valid_idx = trn_df[trn_df['user_id'].isin(valid_user)]

    # 训练集与验证集的用户分组
    train_idx.sort_values(by=['user_id'], inplace=True)
    g_train = train_idx.groupby(['user_id'], as_index=False).count()["label"].values

    valid_idx.sort_values(by=['user_id'], inplace=True)
    g_val = valid_idx.groupby(['user_id'], as_index=False).count()["label"].values

    # 定义模型
    lgb_ranker = lgb.LGBMRanker(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
                            max_depth=-1, n_estimators=100, subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
                            learning_rate=0.01, min_child_weight=20, min_gain_to_split=0.0, random_state=2018, n_jobs= 16)
    # 训练模型
    lgb_ranker.fit(
        train_idx[lgb_cols], train_idx['label'], group=g_train,
        eval_set=[(valid_idx[lgb_cols], valid_idx['label'])], eval_group=[g_val],
        eval_at=[1, 2, 3, 4, 5], eval_metric=['ndcg'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=10)
        ]
    )

    # 预测验证集结果
    valid_idx['pred_score'] = lgb_ranker.predict(valid_idx[lgb_cols], num_iteration=lgb_ranker.best_iteration_)

    # 对输出结果进行归一化
    valid_idx['pred_score'] = valid_idx[['pred_score']].transform(lambda x: norm_sim(x))

    valid_idx.sort_values(by=['user_id', 'pred_score'])
    valid_idx['pred_rank'] = valid_idx.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')

    # 将验证集的预测结果放到一个列表中，后面进行拼接
    score_list.append(valid_idx[['user_id', 'click_article_id', 'pred_score', 'pred_rank']])

    # 如果是线上测试，需要计算每次交叉验证的结果相加，最后求平均
    if not offline:
        sub_preds += lgb_ranker.predict(tst_user_item_feats_df_rank_model[lgb_cols], lgb_ranker.best_iteration_)

score_df_ = pd.concat(score_list, axis=0)
score_df = score_df.merge(score_df_, how='left', on=['user_id', 'click_article_id'])
# 保存训练集交叉验证产生的新特征
score_df[['user_id', 'click_article_id', 'pred_score', 'pred_rank', 'label']].to_csv(
    save_path + 'trn_lgb_ranker_feats.csv', index=False)

# 测试集的预测结果，多次交叉验证求平均,将预测的score和对应的rank特征保存，可以用于后面的staking，这里还可以构造其他更多的特征
tst_user_item_feats_df_rank_model['pred_score'] = sub_preds / k_fold
tst_user_item_feats_df_rank_model['pred_score'] = tst_user_item_feats_df_rank_model['pred_score'].transform(
    lambda x: norm_sim(x))
tst_user_item_feats_df_rank_model.sort_values(by=['user_id', 'pred_score'])
tst_user_item_feats_df_rank_model['pred_rank'] = tst_user_item_feats_df_rank_model.groupby(['user_id'])[
    'pred_score'].rank(ascending=False, method='first')

# 保存测试集交叉验证的新特征
tst_user_item_feats_df_rank_model[['user_id', 'click_article_id', 'pred_score', 'pred_rank']].to_csv(
    save_path + 'tst_lgb_ranker_feats.csv', index=False)

# 预测结果重新排序, 及生成提交结果
# 单模型生成提交结果
rank_results = tst_user_item_feats_df_rank_model[['user_id', 'click_article_id', 'pred_score']]
rank_results['click_article_id'] = rank_results['click_article_id'].astype(int)
submit(rank_results, topk=5, model_name='lgb_ranker')

# 模型及参数的定义
# lgb_Classfication = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
#                             max_depth=-1, n_estimators=500, subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
#                             learning_rate=0.01, min_child_weight=50, random_state=2018, n_jobs= 16, verbose=10)

# # 模型及参数的定义
# lgb_Classfication = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
#                             max_depth=-1, n_estimators=500, subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
#                             learning_rate=0.01, min_child_weight=50, random_state=2018, n_jobs= 16, verbose=10)

# 模型及参数的定义
lgb_Classfication = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
                            max_depth=-1, n_estimators=500, subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
                            learning_rate=0.01, min_child_weight=20, min_gain_to_split=0.0, random_state=2018, n_jobs= 16, verbose=10)

# 模型训练
if offline:
    lgb_Classfication.fit(
        trn_user_item_feats_df_rank_model[lgb_cols], trn_user_item_feats_df_rank_model['label'],
        eval_set=[(val_user_item_feats_df_rank_model[lgb_cols], val_user_item_feats_df_rank_model['label'])],
        eval_metric=['auc'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=10)
        ]
    )
else:
    lgb_Classfication.fit(trn_user_item_feats_df_rank_model[lgb_cols], trn_user_item_feats_df_rank_model['label'])

# 模型预测
tst_user_item_feats_df['pred_score'] = lgb_Classfication.predict_proba(tst_user_item_feats_df[lgb_cols])[:,1]

# 将这里的排序结果保存一份，用户后面的模型融合
tst_user_item_feats_df[['user_id', 'click_article_id', 'pred_score']].to_csv(save_path + 'lgb_cls_score.csv', index=False)

# 预测结果重新排序, 及生成提交结果
rank_results = tst_user_item_feats_df[['user_id', 'click_article_id', 'pred_score']]
rank_results['click_article_id'] = rank_results['click_article_id'].astype(int)
submit(rank_results, topk=5, model_name='lgb_cls')


# 五折交叉验证，这里的五折交叉是以用户为目标进行五折划分
#  这一部分与前面的单独训练和验证是分开的
def get_kfold_users(trn_df, n=5):
    user_ids = trn_df['user_id'].unique()
    user_set = [user_ids[i::n] for i in range(n)]
    return user_set


k_fold = 5
trn_df = trn_user_item_feats_df_rank_model
user_set = get_kfold_users(trn_df, n=k_fold)

score_list = []
score_df = trn_df[['user_id', 'click_article_id', 'label']]
sub_preds = np.zeros(tst_user_item_feats_df_rank_model.shape[0])

# 五折交叉验证，并将中间结果保存用于staking
for n_fold, valid_user in enumerate(user_set):
    train_idx = trn_df[~trn_df['user_id'].isin(valid_user)]  # add slide user
    valid_idx = trn_df[trn_df['user_id'].isin(valid_user)]

    # 模型及参数的定义
    lgb_Classfication = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
                                           max_depth=-1, n_estimators=100, subsample=0.7, colsample_bytree=0.7,
                                           subsample_freq=1,
                                           learning_rate=0.01, min_child_weight=20, min_gain_to_split=0.0, random_state=2018, n_jobs=16,
                                           verbose=10)
    # 训练模型
    lgb_Classfication.fit(
        train_idx[lgb_cols], train_idx['label'],
        eval_set=[(valid_idx[lgb_cols], valid_idx['label'])],
        eval_metric=['auc'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=10)
        ]
    )

    # 预测验证集结果
    valid_idx['pred_score'] = lgb_Classfication.predict_proba(valid_idx[lgb_cols],
                                                              num_iteration=lgb_Classfication.best_iteration_)[:, 1]

    # 对输出结果进行归一化 分类模型输出的值本身就是一个概率值不需要进行归一化
    # valid_idx['pred_score'] = valid_idx[['pred_score']].transform(lambda x: norm_sim(x))

    valid_idx.sort_values(by=['user_id', 'pred_score'])
    valid_idx['pred_rank'] = valid_idx.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')

    # 将验证集的预测结果放到一个列表中，后面进行拼接
    score_list.append(valid_idx[['user_id', 'click_article_id', 'pred_score', 'pred_rank']])

    # 如果是线上测试，需要计算每次交叉验证的结果相加，最后求平均
    if not offline:
        sub_preds += lgb_Classfication.predict_proba(tst_user_item_feats_df_rank_model[lgb_cols],
                                                     num_iteration=lgb_Classfication.best_iteration_)[:, 1]

score_df_ = pd.concat(score_list, axis=0)
score_df = score_df.merge(score_df_, how='left', on=['user_id', 'click_article_id'])
# 保存训练集交叉验证产生的新特征
score_df[['user_id', 'click_article_id', 'pred_score', 'pred_rank', 'label']].to_csv(
    save_path + 'trn_lgb_cls_feats.csv', index=False)

# 测试集的预测结果，多次交叉验证求平均,将预测的score和对应的rank特征保存，可以用于后面的staking，这里还可以构造其他更多的特征
tst_user_item_feats_df_rank_model['pred_score'] = sub_preds / k_fold
tst_user_item_feats_df_rank_model['pred_score'] = tst_user_item_feats_df_rank_model['pred_score'].transform(
    lambda x: norm_sim(x))
tst_user_item_feats_df_rank_model.sort_values(by=['user_id', 'pred_score'])
tst_user_item_feats_df_rank_model['pred_rank'] = tst_user_item_feats_df_rank_model.groupby(['user_id'])[
    'pred_score'].rank(ascending=False, method='first')

# 保存测试集交叉验证的新特征
tst_user_item_feats_df_rank_model[['user_id', 'click_article_id', 'pred_score', 'pred_rank']].to_csv(
    save_path + 'tst_lgb_cls_feats.csv', index=False)

# 预测结果重新排序, 及生成提交结果
rank_results = tst_user_item_feats_df_rank_model[['user_id', 'click_article_id', 'pred_score']]
rank_results['click_article_id'] = rank_results['click_article_id'].astype(int)
submit(rank_results, topk=5, model_name='lgb_cls')

# ===================== 修复时间泄露：训练时只能用训练集数据构建历史行为 =====================
if offline:
    all_data = pd.read_csv('../data_raw/train_click_log.csv')
    # offline 模式：用所有训练数据构建历史
    # ⚠️ 重要：必须先按时间排序，确保历史序列按时间顺序（DIEN 等序列模型需要）
    all_data = all_data.sort_values(by=['user_id', 'click_timestamp'])
    hist_click = all_data.groupby('user_id')['click_article_id'].agg(list).reset_index()
else:
    # online 模式：训练集和测试集的用户不交叉（20万用户 vs 5万用户）
    # 因此训练集和测试集的历史行为应该分别只用各自的数据
    trn_data = pd.read_csv('../data_raw/train_click_log.csv')
    tst_data = pd.read_csv('../data_raw/testA_click_log.csv')
    
    # 训练集的历史行为：只用训练集数据
    # ⚠️ 重要：必须先按时间排序，确保历史序列按时间顺序
    trn_data = trn_data.sort_values(by=['user_id', 'click_timestamp'])
    trn_hist_click = trn_data.groupby('user_id')['click_article_id'].agg(list).reset_index()
    
    # 测试集的历史行为：只用测试集数据（因为用户不交叉，训练集用户的历史对测试集用户没有意义）
    # ⚠️ 重要：必须先按时间排序，确保历史序列按时间顺序
    tst_data = tst_data.sort_values(by=['user_id', 'click_timestamp'])
    tst_hist_click = tst_data.groupby('user_id')['click_article_id'].agg(list).reset_index()
    
    hist_click = trn_hist_click  # 先用于训练集

his_behavior_df = pd.DataFrame()
his_behavior_df['user_id'] = hist_click['user_id']
his_behavior_df['hist_click_article_id'] = hist_click['click_article_id']

trn_user_item_feats_df_din_model = trn_user_item_feats_df.copy()

if offline:
    val_user_item_feats_df_din_model = val_user_item_feats_df.copy()
else:
    val_user_item_feats_df_din_model = None

tst_user_item_feats_df_din_model = tst_user_item_feats_df.copy()

# 训练集：只用训练集的历史行为（避免泄露）
trn_user_item_feats_df_din_model = trn_user_item_feats_df_din_model.merge(his_behavior_df, on='user_id', how='left')

# 测试集：使用测试集自身的历史行为
if not offline:
    # online 模式：测试集使用测试集自身的历史行为
    # 注意：因为训练集和测试集的用户不交叉，所以测试集用户的历史行为只来自测试集数据
    tst_his_behavior_df = pd.DataFrame()
    tst_his_behavior_df['user_id'] = tst_hist_click['user_id']
    tst_his_behavior_df['hist_click_article_id'] = tst_hist_click['click_article_id']
    tst_user_item_feats_df_din_model = tst_user_item_feats_df_din_model.merge(tst_his_behavior_df, on='user_id', how='left')
else:
    # offline 模式：测试集使用训练集的历史行为（offline 模式下测试集可能是从训练集划分出来的）
    tst_user_item_feats_df_din_model = tst_user_item_feats_df_din_model.merge(his_behavior_df, on='user_id', how='left')

if offline:
    val_user_item_feats_df_din_model = val_user_item_feats_df_din_model.merge(his_behavior_df, on='user_id', how='left')
else:
    val_user_item_feats_df_din_model = None



# ===================== DIEN (DeepCTR) 训练与预测（不影响 LightGBM 流程） =====================
try:
    from deepctr.feature_column import SparseFeat, VarLenSparseFeat
    from deepctr.models import DIEN
    import numpy as np
    from sklearn.preprocessing import LabelEncoder

    print('[DIEN] start preparing data')

    # 检查必要的列是否存在
    required_cols_trn = ['user_id', 'click_article_id', 'label', 'hist_click_article_id']
    required_cols_tst = ['user_id', 'click_article_id', 'hist_click_article_id']
    
    missing_cols_trn = [col for col in required_cols_trn if col not in trn_user_item_feats_df_din_model.columns]
    missing_cols_tst = [col for col in required_cols_tst if col not in tst_user_item_feats_df_din_model.columns]
    
    if missing_cols_trn or missing_cols_tst:
        raise KeyError(f"Missing columns: train={missing_cols_trn}, test={missing_cols_tst}")

    # 仅取必要字段，拷贝以免影响上游 DataFrame
    trn_dien = trn_user_item_feats_df_din_model[required_cols_trn].copy()
    tst_dien = tst_user_item_feats_df_din_model[required_cols_tst].copy()
    
    # 填充缺失的历史行为（如果某些用户没有历史行为）
    # hist_click_article_id 应该是列表类型，缺失值填充为空列表
    def ensure_list(x):
        if pd.isna(x) or x is None:
            return []
        if isinstance(x, list):
            return x
        # 如果是其他类型（如字符串），尝试转换
        return [] if not x else x
    
    if trn_dien['hist_click_article_id'].isna().any():
        trn_dien['hist_click_article_id'] = trn_dien['hist_click_article_id'].apply(ensure_list)
    if tst_dien['hist_click_article_id'].isna().any():
        tst_dien['hist_click_article_id'] = tst_dien['hist_click_article_id'].apply(ensure_list)

    # 连续编号编码（稀疏 id 更稳定）
    # ⚠️ 注意：由于训练集和测试集的用户不交叉，测试集中会出现未见过的用户ID
    # 需要先合并所有可能的ID，然后再编码，或者使用 OOV 处理
    
    # 方法1：合并训练集和测试集的所有ID，然后统一编码（推荐）
    all_user_ids = pd.concat([
        trn_dien['user_id'].astype(str),
        tst_dien['user_id'].astype(str)
    ]).unique()
    all_item_ids = pd.concat([
        trn_dien['click_article_id'].astype(str),
        tst_dien['click_article_id'].astype(str)
    ]).unique()
    
    user_lbe = LabelEncoder()
    item_lbe = LabelEncoder()
    
    # 先 fit 所有可能的ID（包括训练集和测试集）
    user_lbe.fit(all_user_ids)
    item_lbe.fit(all_item_ids)
    
    # 然后 transform
    trn_dien['user_id_enc'] = user_lbe.transform(trn_dien['user_id'].astype(str))
    tst_dien['user_id_enc'] = user_lbe.transform(tst_dien['user_id'].astype(str))

    trn_dien['item_id_enc'] = item_lbe.transform(trn_dien['click_article_id'].astype(str))
    tst_dien['item_id_enc'] = item_lbe.transform(tst_dien['click_article_id'].astype(str))

    # 将历史序列映射为编码后的 item_id
    # 映射表：包含所有 item 的映射（训练集+测试集）
    # 由于已经用所有ID fit 了 LabelEncoder，所以可以直接构建完整映射
    all_item_ids_list = pd.concat([
        trn_dien['click_article_id'].astype(str),
        tst_dien['click_article_id'].astype(str)
    ]).unique()
    all_item_enc_list = item_lbe.transform(all_item_ids_list)
    item_map = {str(orig): int(enc) for orig, enc in zip(all_item_ids_list, all_item_enc_list)}

    def map_hist(seq):
        if not isinstance(seq, list):
            return []
        res = []
        for s in seq:
            enc = item_map.get(str(s))
            if enc is not None:
                res.append(int(enc))
        return res

    trn_dien['hist_item_id'] = trn_dien['hist_click_article_id'].apply(map_hist)
    tst_dien['hist_item_id'] = tst_dien['hist_click_article_id'].apply(map_hist)

    # 长度与 padding
    max_seq_len = 50
    trn_dien['hist_len'] = trn_dien['hist_item_id'].apply(lambda x: len(x))
    tst_dien['hist_len'] = tst_dien['hist_item_id'].apply(lambda x: len(x))

    def pad_seq(seq, maxlen):
        seq = seq if isinstance(seq, list) else []
        if len(seq) >= maxlen:
            return np.array(seq[-maxlen:])
        return np.array(seq + [0] * (maxlen - len(seq)))

    X_trn = {
        'user_id': trn_dien['user_id_enc'].values,
        'item_id': trn_dien['item_id_enc'].values,
        'hist_item_id': np.stack(trn_dien['hist_item_id'].apply(lambda s: pad_seq(s, max_seq_len)).values),
        'hist_len': trn_dien['hist_len'].values
    }
    y_trn = trn_dien['label'].values

    X_tst = {
        'user_id': tst_dien['user_id_enc'].values,
        'item_id': tst_dien['item_id_enc'].values,
        'hist_item_id': np.stack(tst_dien['hist_item_id'].apply(lambda s: pad_seq(s, max_seq_len)).values),
        'hist_len': tst_dien['hist_len'].values
    }

    # 特征列
    user_num = int(max(trn_dien['user_id_enc'].max(), tst_dien['user_id_enc'].max())) + 1
    item_num = int(max(trn_dien['item_id_enc'].max(), tst_dien['item_id_enc'].max())) + 1

    d_model = 16
    feature_columns = [
        SparseFeat('user_id', vocabulary_size=user_num, embedding_dim=d_model),
        SparseFeat('item_id', vocabulary_size=item_num, embedding_dim=d_model),
        VarLenSparseFeat(SparseFeat('hist_item_id', vocabulary_size=item_num, embedding_dim=d_model),
                         maxlen=max_seq_len, length_name='hist_len')
    ]
    behavior_feature_list = ['item_id']

    # 构建与训练（轻量参数以避免影响现有流程时间）
    print('[DIEN] building model')
    model = DIEN(feature_columns, behavior_feature_list, dnn_hidden_units=(128, 64), gru_type='AUGRU')
    model.compile(optimizer='adagrad', loss='binary_crossentropy')
    print('[DIEN] training')
    model.fit(X_trn, y_trn, batch_size=1024, epochs=1, verbose=2, validation_split=0.0)

    print('[DIEN] predicting')
    dien_pred = model.predict(X_tst, batch_size=2048).reshape(-1)

    # 保存为供后续融合读取的文件（与下方读取路径一致：din_rank_score.csv）
    din_out = pd.DataFrame({
        'user_id': tst_user_item_feats_df_din_model['user_id'].values,
        'click_article_id': tst_user_item_feats_df_din_model['click_article_id'].values,
        'pred_score': dien_pred
    })
    din_out.to_csv(save_path + 'din_rank_score.csv', index=False)
    print('[DIEN] saved:', save_path + 'din_rank_score.csv')

except Exception as e:
    # 若 deepctr 不可用或训练异常，回退生成一个占位分数，避免后续流程报错
    print('[DIEN] skip due to error:', e)
    placeholder = pd.DataFrame({
        'user_id': tst_user_item_feats_df_din_model['user_id'].values,
        'click_article_id': tst_user_item_feats_df_din_model['click_article_id'].values,
        'pred_score': np.zeros(len(tst_user_item_feats_df_din_model))
    })
    placeholder.to_csv(save_path + 'din_rank_score.csv', index=False)
    print('[DIEN] wrote placeholder:', save_path + 'din_rank_score.csv')

# DIEN模型




# 读取多个模型的排序结果文件
lgb_ranker = pd.read_csv(save_path + 'lgb_ranker_score.csv')
lgb_cls = pd.read_csv(save_path + 'lgb_cls_score.csv')
din_ranker = pd.read_csv(save_path + 'din_rank_score.csv')

# 这里也可以换成交叉验证输出的测试结果进行加权融合
rank_model = {'lgb_ranker': lgb_ranker,
              'lgb_cls': lgb_cls,
              'din_ranker': din_ranker}


def get_ensumble_predict_topk(rank_model, topk=5):
    # 使用 pd.concat 替代已废弃的 append 方法（pandas 2.0+）
    final_recall = pd.concat([rank_model['lgb_cls'], rank_model['din_ranker']], ignore_index=True)
    rank_model['lgb_ranker']['pred_score'] = rank_model['lgb_ranker']['pred_score'].transform(lambda x: norm_sim(x))

    final_recall = pd.concat([final_recall, rank_model['lgb_ranker']], ignore_index=True)
    final_recall = final_recall.groupby(['user_id', 'click_article_id'])['pred_score'].sum().reset_index()

    submit(final_recall, topk=topk, model_name='ensemble_fuse')

get_ensumble_predict_topk(rank_model)

# 读取多个模型的交叉验证生成的结果文件
# 训练集
trn_lgb_ranker_feats = pd.read_csv(save_path + 'trn_lgb_ranker_feats.csv')
trn_lgb_cls_feats = pd.read_csv(save_path + 'trn_lgb_cls_feats.csv')

# 测试集
tst_lgb_ranker_feats = pd.read_csv(save_path + 'tst_lgb_ranker_feats.csv')
tst_lgb_cls_feats = pd.read_csv(save_path + 'tst_lgb_cls_feats.csv')

# DIEN 特征文件（如果不存在则创建占位符）
# ⚠️ 重要：DIEN 模型目前只训练一次并只预测测试集，没有进行交叉验证
# 因此需要为训练集和测试集创建占位符，确保所有样本都有 DIEN 特征（即使为0）
def load_or_create_din_feats(filepath, base_df, is_train=False):
    """加载 DIEN 特征文件，如果不存在则创建占位符"""
    if os.path.exists(filepath):
        din_feats = pd.read_csv(filepath)
        print(f'[Info] Loaded DIEN features from {filepath}: {len(din_feats)} rows')
        return din_feats
    else:
        print(f'[Warning] {filepath} not found, creating placeholder')
        # 创建占位符：使用 base_df 的结构，pred_score 和 pred_rank 都设为 0
        # 这确保所有样本都有 DIEN 特征，避免 NaN
        placeholder = base_df[['user_id', 'click_article_id']].copy()
        placeholder['pred_score'] = 0.0
        placeholder['pred_rank'] = 0.0
        if is_train and 'label' in base_df.columns:
            placeholder['label'] = base_df['label'].values
        placeholder.to_csv(filepath, index=False)
        print(f'[Info] Created placeholder: {filepath} with {len(placeholder)} rows')
        return placeholder

# ⚠️ 重要：使用 trn_lgb_ranker_feats 作为基础结构（最完整的特征文件）
# 确保占位符包含所有样本，避免 NaN
print("\n[DIEN] 加载或创建 DIEN 交叉验证特征文件...")
trn_din_cls_feats = load_or_create_din_feats(
    save_path + 'trn_din_cls_feats.csv', 
    trn_lgb_ranker_feats,  # 使用最完整的特征文件作为基础
    is_train=True
)
tst_din_cls_feats = load_or_create_din_feats(
    save_path + 'tst_din_cls_feats.csv', 
    tst_lgb_ranker_feats,  # 使用最完整的特征文件作为基础
    is_train=False
)

# 检查 DIEN 特征文件是否包含所有样本
print(f"\n[DIEN] 检查 DIEN 特征文件完整性：")
print(f"  trn_lgb_ranker_feats: {len(trn_lgb_ranker_feats)} 行")
print(f"  trn_din_cls_feats: {len(trn_din_cls_feats)} 行")
print(f"  训练集覆盖率: {len(trn_din_cls_feats) / len(trn_lgb_ranker_feats) * 100:.2f}%")
print(f"  tst_lgb_ranker_feats: {len(tst_lgb_ranker_feats)} 行")
print(f"  tst_din_cls_feats: {len(tst_din_cls_feats)} 行")
print(f"  测试集覆盖率: {len(tst_din_cls_feats) / len(tst_lgb_ranker_feats) * 100:.2f}%")

if len(trn_din_cls_feats) < len(trn_lgb_ranker_feats):
    print(f"  ⚠️  警告：训练集 DIEN 特征不完整，将创建完整占位符")
    # 基于 trn_lgb_ranker_feats 创建完整占位符
    trn_din_cls_feats = trn_lgb_ranker_feats[['user_id', 'click_article_id', 'label']].copy()
    trn_din_cls_feats['pred_score'] = 0.0
    # 计算 pred_rank（虽然都是 0，但需要保持格式一致）
    trn_din_cls_feats.sort_values(by=['user_id', 'pred_score'])
    trn_din_cls_feats['pred_rank'] = trn_din_cls_feats.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')
    trn_din_cls_feats.to_csv(save_path + 'trn_din_cls_feats.csv', index=False)
    print(f"  ✓ 已创建完整的训练集 DIEN 占位符: {len(trn_din_cls_feats)} 行")

if len(tst_din_cls_feats) < len(tst_lgb_ranker_feats):
    print(f"  ⚠️  警告：测试集 DIEN 特征不完整，将创建完整占位符")
    # 基于 tst_lgb_ranker_feats 创建完整占位符
    tst_din_cls_feats = tst_lgb_ranker_feats[['user_id', 'click_article_id']].copy()
    tst_din_cls_feats['pred_score'] = 0.0
    # 计算 pred_rank（虽然都是 0，但需要保持格式一致）
    tst_din_cls_feats.sort_values(by=['user_id', 'pred_score'])
    tst_din_cls_feats['pred_rank'] = tst_din_cls_feats.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')
    tst_din_cls_feats.to_csv(save_path + 'tst_din_cls_feats.csv', index=False)
    print(f"  ✓ 已创建完整的测试集 DIEN 占位符: {len(tst_din_cls_feats)} 行")

# 将多个模型输出的特征进行拼接
# ⚠️ 重要：使用 merge 而不是直接赋值，避免因为行数或顺序不一致导致的 NaN 问题
# 原因：不同模型的特征文件可能基于不同的召回结果或负采样，导致 (user_id, click_article_id) 对不一致

print("\n" + "=" * 80)
print("诊断：检查特征文件的样本对齐情况")
print("=" * 80)

# 检查各个特征文件的样本数量和用户-物品对
print(f"\n[诊断] 训练集特征文件样本数：")
print(f"  trn_lgb_ranker_feats: {len(trn_lgb_ranker_feats)} 行")
print(f"  trn_lgb_cls_feats: {len(trn_lgb_cls_feats)} 行")
print(f"  trn_din_cls_feats: {len(trn_din_cls_feats)} 行")

# 检查用户-物品对的唯一性
trn_base_keys = set(zip(trn_lgb_ranker_feats['user_id'], trn_lgb_ranker_feats['click_article_id']))
trn_cls_keys = set(zip(trn_lgb_cls_feats['user_id'], trn_lgb_cls_feats['click_article_id']))
trn_din_keys = set(zip(trn_din_cls_feats['user_id'], trn_din_cls_feats['click_article_id']))

print(f"\n[诊断] 训练集用户-物品对数量：")
print(f"  trn_lgb_ranker_feats: {len(trn_base_keys)} 个唯一对")
print(f"  trn_lgb_cls_feats: {len(trn_cls_keys)} 个唯一对")
print(f"  trn_din_cls_feats: {len(trn_din_keys)} 个唯一对")

# 检查交集和差集
print(f"\n[诊断] 训练集用户-物品对交集：")
print(f"  ranker & cls: {len(trn_base_keys & trn_cls_keys)} 个共同对")
print(f"  ranker & din: {len(trn_base_keys & trn_din_keys)} 个共同对")
print(f"  cls & din: {len(trn_cls_keys & trn_din_keys)} 个共同对")

print(f"\n[诊断] 训练集用户-物品对差集：")
print(f"  ranker 独有: {len(trn_base_keys - trn_cls_keys - trn_din_keys)} 个")
print(f"  cls 独有: {len(trn_cls_keys - trn_base_keys - trn_din_keys)} 个")
print(f"  din 独有: {len(trn_din_keys - trn_base_keys - trn_cls_keys)} 个")

# 检查测试集
print(f"\n[诊断] 测试集特征文件样本数：")
print(f"  tst_lgb_ranker_feats: {len(tst_lgb_ranker_feats)} 行")
print(f"  tst_lgb_cls_feats: {len(tst_lgb_cls_feats)} 行")
print(f"  tst_din_cls_feats: {len(tst_din_cls_feats)} 行")

tst_base_keys = set(zip(tst_lgb_ranker_feats['user_id'], tst_lgb_ranker_feats['click_article_id']))
tst_cls_keys = set(zip(tst_lgb_cls_feats['user_id'], tst_lgb_cls_feats['click_article_id']))
tst_din_keys = set(zip(tst_din_cls_feats['user_id'], tst_din_cls_feats['click_article_id']))

print(f"\n[诊断] 测试集用户-物品对数量：")
print(f"  tst_lgb_ranker_feats: {len(tst_base_keys)} 个唯一对")
print(f"  tst_lgb_cls_feats: {len(tst_cls_keys)} 个唯一对")
print(f"  tst_din_cls_feats: {len(tst_din_keys)} 个唯一对")

print(f"\n[诊断] 测试集用户-物品对交集：")
print(f"  ranker & cls: {len(tst_base_keys & tst_cls_keys)} 个共同对")
print(f"  ranker & din: {len(tst_base_keys & tst_din_keys)} 个共同对")
print(f"  cls & din: {len(tst_cls_keys & tst_din_keys)} 个共同对")

print(f"\n[诊断] 测试集用户-物品对差集：")
print(f"  ranker 独有: {len(tst_base_keys - tst_cls_keys - tst_din_keys)} 个")
print(f"  cls 独有: {len(tst_cls_keys - tst_base_keys - tst_din_keys)} 个")
print(f"  din 独有: {len(tst_din_keys - tst_base_keys - tst_cls_keys)} 个")

finall_trn_ranker_feats = trn_lgb_ranker_feats[['user_id', 'click_article_id', 'label']].copy()
finall_tst_ranker_feats = tst_lgb_ranker_feats[['user_id', 'click_article_id']].copy()

# 诊断：检查基础特征文件的用户物品数量
print("\n[诊断] 基础特征文件统计（合并前）:")
print(f"  finall_tst_ranker_feats 基础用户数: {finall_tst_ranker_feats['user_id'].nunique()}")
print(f"  finall_tst_ranker_feats 基础用户-物品对数: {len(finall_tst_ranker_feats)}")
base_user_counts = finall_tst_ranker_feats.groupby('user_id').size()
print(f"  基础文件中每个用户的物品数统计:")
print(f"    最小值: {base_user_counts.min()}, 最大值: {base_user_counts.max()}")
print(f"    平均值: {base_user_counts.mean():.2f}, 中位数: {base_user_counts.median():.2f}")
insufficient_base = (base_user_counts < 5).sum()
print(f"    少于5个物品的用户数: {insufficient_base} ({insufficient_base/len(base_user_counts)*100:.2f}%)")

# 训练集特征合并
print(f"\n[诊断] 开始合并训练集特征...")
for idx, trn_model in enumerate([trn_lgb_ranker_feats, trn_lgb_cls_feats, trn_din_cls_feats]):
    model_name = ['ranker', 'cls', 'din'][idx]
    for feat in ['pred_score', 'pred_rank']:
        col_name = feat + '_' + str(idx)
        # 使用 merge 确保基于 (user_id, click_article_id) 对齐，避免 NaN
        feat_df = trn_model[['user_id', 'click_article_id', feat]].rename(columns={feat: col_name})
        
        # 检查合并前的情况
        before_keys = set(zip(finall_trn_ranker_feats['user_id'], finall_trn_ranker_feats['click_article_id']))
        feat_keys = set(zip(feat_df['user_id'], feat_df['click_article_id']))
        missing_in_feat = before_keys - feat_keys
        
        if len(missing_in_feat) > 0:
            print(f"  ⚠️  {model_name}.{feat}: {len(missing_in_feat)} 个用户-物品对在特征文件中缺失（将产生NaN）")
            if len(missing_in_feat) <= 10:
                print(f"     示例缺失对: {list(missing_in_feat)[:10]}")
        
        finall_trn_ranker_feats = finall_trn_ranker_feats.merge(
            feat_df, 
            on=['user_id', 'click_article_id'], 
            how='left'
        )
        
        # 检查合并后的NaN情况
        nan_count = finall_trn_ranker_feats[col_name].isna().sum()
        if nan_count > 0:
            print(f"  ⚠️  {model_name}.{feat}: 合并后产生 {nan_count} 个NaN值")

# 测试集特征合并
print(f"\n[诊断] 开始合并测试集特征...")
for idx, tst_model in enumerate([tst_lgb_ranker_feats, tst_lgb_cls_feats, tst_din_cls_feats]):
    model_name = ['ranker', 'cls', 'din'][idx]
    for feat in ['pred_score', 'pred_rank']:
        col_name = feat + '_' + str(idx)
        # 使用 merge 确保基于 (user_id, click_article_id) 对齐，避免 NaN
        feat_df = tst_model[['user_id', 'click_article_id', feat]].rename(columns={feat: col_name})
        
        # 检查合并前的情况
        before_keys = set(zip(finall_tst_ranker_feats['user_id'], finall_tst_ranker_feats['click_article_id']))
        feat_keys = set(zip(feat_df['user_id'], feat_df['click_article_id']))
        missing_in_feat = before_keys - feat_keys
        
        if len(missing_in_feat) > 0:
            print(f"  ⚠️  {model_name}.{feat}: {len(missing_in_feat)} 个用户-物品对在特征文件中缺失（将产生NaN）")
            if len(missing_in_feat) <= 10:
                print(f"     示例缺失对: {list(missing_in_feat)[:10]}")
        
        finall_tst_ranker_feats = finall_tst_ranker_feats.merge(
            feat_df, 
            on=['user_id', 'click_article_id'], 
            how='left'
        )
        
        # 检查合并后的NaN情况
        nan_count = finall_tst_ranker_feats[col_name].isna().sum()
        if nan_count > 0:
            print(f"  ⚠️  {model_name}.{feat}: 合并后产生 {nan_count} 个NaN值")

print("\n" + "=" * 80)

# 定义一个逻辑回归模型再次拟合交叉验证产生的特征对测试集进行预测
# 这里需要注意的是，在做交叉验证的时候可以构造多一些与输出预测值相关的特征，来丰富这里简单模型的特征
from sklearn.linear_model import LogisticRegression

feat_cols = ['pred_score_0', 'pred_rank_0', 'pred_score_1', 'pred_rank_1', 'pred_score_2', 'pred_rank_2']

# 检查并处理可能的 NaN 值
print(f"\n[诊断] 最终检查特征中的NaN值...")
print(f"  训练集形状: {finall_trn_ranker_feats.shape}")
print(f"  训练集NaN统计:\n{finall_trn_ranker_feats[feat_cols].isna().sum()}")
if finall_trn_ranker_feats[feat_cols].isna().any().any():
    print(f"  ⚠️  训练集特征中发现NaN值，使用0填充")
    finall_trn_ranker_feats[feat_cols] = finall_trn_ranker_feats[feat_cols].fillna(0.0)

print(f"  测试集形状: {finall_tst_ranker_feats.shape}")
print(f"  测试集NaN统计:\n{finall_tst_ranker_feats[feat_cols].isna().sum()}")
if finall_tst_ranker_feats[feat_cols].isna().any().any():
    print(f"  ⚠️  测试集特征中发现NaN值，使用0填充")
    finall_tst_ranker_feats[feat_cols] = finall_tst_ranker_feats[feat_cols].fillna(0.0)

trn_x = finall_trn_ranker_feats[feat_cols]
trn_y = finall_trn_ranker_feats['label']

tst_x = finall_tst_ranker_feats[feat_cols]

# 定义模型
lr = LogisticRegression()

# 模型训练
lr.fit(trn_x, trn_y)

# 模型预测
finall_tst_ranker_feats['pred_score'] = lr.predict_proba(tst_x)[:, 1]

# 预测结果重新排序, 及生成提交结果
rank_results = finall_tst_ranker_feats[['user_id', 'click_article_id', 'pred_score']]

# 诊断：检查提交前每个用户的物品数量
print("\n" + "=" * 80)
print("诊断：提交前的用户物品数量分析")
print("=" * 80)
user_item_counts = rank_results.groupby('user_id').size()
print(f"\n[诊断] 提交前统计：")
print(f"  总用户数: {len(user_item_counts)}")
print(f"  总用户-物品对数: {len(rank_results)}")
print(f"  每个用户的物品数统计:")
print(f"    最小值: {user_item_counts.min()}")
print(f"    最大值: {user_item_counts.max()}")
print(f"    平均值: {user_item_counts.mean():.2f}")
print(f"    中位数: {user_item_counts.median():.2f}")
print(f"  物品数分布:")
for threshold in [1, 2, 3, 4, 5, 10, 20, 50]:
    count = (user_item_counts < threshold).sum()
    pct = count / len(user_item_counts) * 100
    print(f"    < {threshold} 个物品的用户: {count} ({pct:.2f}%)")

# 检查少于5个物品的用户
users_with_insufficient = user_item_counts[user_item_counts < 5]
if len(users_with_insufficient) > 0:
    print(f"\n[诊断] 少于5个物品的用户详情（前20个）:")
    for user_id, count in users_with_insufficient.head(20).items():
        user_items = rank_results[rank_results['user_id'] == user_id]
        print(f"  用户 {user_id}: {count} 个物品")
        print(f"    物品列表: {user_items['click_article_id'].tolist()}")
        print(f"    预测分数范围: [{user_items['pred_score'].min():.4f}, {user_items['pred_score'].max():.4f}]")
    
    # 检查这些用户是否在原始特征文件中
    print(f"\n[诊断] 检查这些用户在原始特征文件中的情况:")
    insufficient_user_ids = users_with_insufficient.index.tolist()[:10]
    for user_id in insufficient_user_ids:
        in_ranker = (tst_lgb_ranker_feats['user_id'] == user_id).sum()
        in_cls = (tst_lgb_cls_feats['user_id'] == user_id).sum()
        in_din = (tst_din_cls_feats['user_id'] == user_id).sum()
        in_original = (tst_user_item_feats_df['user_id'] == user_id).sum()
        print(f"  用户 {user_id}:")
        print(f"    原始特征文件: {in_original} 个物品")
        print(f"    ranker特征: {in_ranker} 个物品")
        print(f"    cls特征: {in_cls} 个物品")
        print(f"    din特征: {in_din} 个物品")

submit(rank_results, topk=5, model_name='ensumble_staking')

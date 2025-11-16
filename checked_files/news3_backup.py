import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os, math, warnings, math, pickle
from tqdm import tqdm
import faiss
import collections
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import backend as K
# from tensorflow.python.keras.models import Model
# from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from deepmatch.utils import sampledsoftmaxloss, NegativeSampler
from tensorflow.keras.utils import plot_model

from collections import Counter
from deepmatch.models import *
from deepmatch.utils import sampledsoftmaxloss

warnings.filterwarnings('ignore')

data_path = '../data_raw/'
save_path = '../temp_results/'
# 做召回评估的一个标志, 如果不进行评估就是直接使用全量数据进行召回
metric_recall = False


# debug模式： 从训练集中划出一部分数据来调试代码
def get_all_click_sample(data_path, sample_nums=10000):
    """
        训练集中采样一部分数据调试
        data_path: 原数据的存储路径
        sample_nums: 采样数目（这里由于机器的内存限制，可以采样用户做）
    """
    all_click = pd.read_csv(data_path + 'train_click_log.csv')
    all_user_ids = all_click.user_id.unique()

    sample_user_ids = np.random.choice(all_user_ids, size=sample_nums, replace=False)
    all_click = all_click[all_click['user_id'].isin(sample_user_ids)]

    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    return all_click


# 读取点击数据，这里分成线上和线下，如果是为了获取线上提交结果应该讲测试集中的点击数据合并到总的数据中
# 如果是为了线下验证模型的有效性或者特征的有效性，可以只使用训练集
# def get_all_click_df(data_path='../data_raw/', offline=True):
#     if offline:
#         all_click = pd.read_csv(data_path + 'train_click_log.csv')
#     else:
#         trn_click = pd.read_csv(data_path + 'train_click_log.csv')
#         tst_click = pd.read_csv(data_path + 'testA_click_log.csv')
#
#         all_click = trn_click.append(tst_click)
#
#     all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
#     return all_click

def get_all_click_df(data_path='../data_raw/', offline=True, sample_size=100000):
    if offline:
        all_click = pd.read_csv(data_path + 'train_click_log.csv')
    else:
        trn_click = pd.read_csv(data_path + 'train_click_log.csv')
        tst_click = pd.read_csv(data_path + 'testA_click_log.csv')
        all_click = pd.concat([trn_click, tst_click], ignore_index=True)

    # 去重
    all_click = all_click.drop_duplicates(['user_id', 'click_article_id', 'click_timestamp'])
    
    print(f"all_click.shape: {all_click.shape}")
    print(f"all_click length: {len(all_click)}")
    print(f"unique users: {all_click['user_id'].nunique()}")
    print(f"unique items: {all_click['click_article_id'].nunique()}")

    # 改进的采样策略：保证所有用户和物品都被包含
    if sample_size and len(all_click) > sample_size:
        # 获取所有唯一的用户和物品数量
        unique_users_count = all_click['user_id'].nunique()
        unique_items_count = all_click['click_article_id'].nunique()
        
        # 计算最少需要的记录数：max(unique_users, unique_items)
        # 理论上，最少需要 max(unique_users, unique_items) 条记录才能保证所有用户和物品都被包含
        min_records_needed = max(unique_users_count, unique_items_count)
        
        print(f"最少需要的记录数（保证所有用户和物品）: {min_records_needed}")
        print(f"  unique users: {unique_users_count}")
        print(f"  unique items: {unique_items_count}")
        
        # 如果最少记录数已经超过 sample_size，调整 sample_size
        if min_records_needed >= sample_size:
            print(f"警告: 最少记录数({min_records_needed}) >= sample_size({sample_size})")
            print(f"调整 sample_size 为: {min_records_needed}")
            sample_size = min_records_needed
        
        # 第一步：确保所有用户和物品的记录都被包含
        # 使用 groupby 更高效地获取每个用户和物品的第一条记录
        user_item_records = set()
        
        # 为每个用户保留至少一条记录
        for user_id, group in all_click.groupby('user_id'):
            if len(group) > 0:
                user_item_records.add(group.index[0])
        
        # 为每个物品保留至少一条记录（如果还没有被包含）
        for item_id, group in all_click.groupby('click_article_id'):
            if len(group) > 0:
                # 检查是否已经有包含该物品的记录
                item_idx = group.index[0]
                if item_idx not in user_item_records:
                    user_item_records.add(item_idx)
        
        # 计算实际保留的记录数
        actual_min_records = len(user_item_records)
        print(f"实际保留的记录数（保证所有用户和物品）: {actual_min_records}")
        
        # 第二步：从剩余记录中随机采样，直到达到 sample_size
        if actual_min_records < sample_size:
            remaining_records = all_click.drop(index=list(user_item_records))
            remaining_sample_size = sample_size - actual_min_records
            
            if len(remaining_records) > 0:
                if len(remaining_records) > remaining_sample_size:
                    remaining_sampled = remaining_records.sample(n=remaining_sample_size, random_state=1)
                else:
                    remaining_sampled = remaining_records
                
                # 合并：保证的记录 + 随机采样的记录
                sampled_click = pd.concat([
                    all_click.loc[list(user_item_records)],
                    remaining_sampled
                ], ignore_index=True)
            else:
                sampled_click = all_click.loc[list(user_item_records)]
        else:
            # 如果实际保留的记录数已经达到或超过 sample_size，直接使用
            if actual_min_records == sample_size:
                sampled_click = all_click.loc[list(user_item_records)]
            else:
                # 如果超过，随机选择 sample_size 条记录（但保证所有用户和物品）
                # 这种情况理论上不应该发生，但为了安全起见
                sampled_click = all_click.loc[list(user_item_records)].sample(n=sample_size, random_state=1)
        
        # 验证采样后的用户和物品覆盖率（应该是100%）
        sampled_users = sampled_click['user_id'].nunique()
        sampled_items = sampled_click['click_article_id'].nunique()
        user_coverage = sampled_users / unique_users_count * 100
        item_coverage = sampled_items / unique_items_count * 100
        
        print(f"采样后:")
        print(f"  all_click.shape: {sampled_click.shape}")
        print(f"  all_click length: {len(sampled_click)}")
        print(f"  unique users: {sampled_users}/{unique_users_count} (覆盖率: {user_coverage:.2f}%)")
        print(f"  unique items: {sampled_items}/{unique_items_count} (覆盖率: {item_coverage:.2f}%)")
        
        if user_coverage < 100.0 or item_coverage < 100.0:
            print(f"  ⚠️  警告: 用户或物品覆盖率未达到100%，可能存在数据问题")
        
        all_click = sampled_click

    return all_click


# 读取文章的基本属性
def get_item_info_df(data_path):
    item_info_df = pd.read_csv(data_path + 'articles.csv')

    # 为了方便与训练集中的click_article_id拼接，需要把article_id修改成click_article_id
    item_info_df = item_info_df.rename(columns={'article_id': 'click_article_id'})

    return item_info_df


# 读取文章的Embedding数据
def get_item_emb_dict(data_path):
    item_emb_df = pd.read_csv(data_path + 'articles_emb.csv')

    item_emb_cols = [x for x in item_emb_df.columns if 'emb' in x]
    item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols])
    # 进行归一化
    item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)

    item_emb_dict = dict(zip(item_emb_df['article_id'], item_emb_np))
    pickle.dump(item_emb_dict, open(save_path + 'item_content_emb.pkl', 'wb'))

    return item_emb_dict


# 采样数据
# all_click_df = get_all_click_sample(data_path)

# 全量训练集
# 注意：采样会严重影响召回效果（文章保留率只有55%，用户保留率72.8%）
# 如果内存允许，建议使用全量数据（sample_size=None）以获得更好的召回效果
# all_click_df = get_all_click_df(offline=False, sample_size=None)  # 使用全量数据，不采样
# all_click_df = get_all_click_df(offline=False, sample_size=10000)  # 使用全量数据，不采样
all_click_df = get_all_click_df(offline=False, sample_size=None)  # 如果内存不足，可以采样

max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))

# 对时间戳进行归一化,用于在关联规则的时候计算权重
all_click_df['click_timestamp'] = all_click_df[['click_timestamp']].apply(max_min_scaler)

item_info_df = get_item_info_df(data_path)
item_emb_dict = get_item_emb_dict(data_path)


# 根据点击时间获取用户的点击文章序列   {user1: [(item1, time1), (item2, time2)..]...}
def get_user_item_time(click_df):
    click_df = click_df.sort_values('click_timestamp')

    def make_item_time_pair(df):
        return list(zip(df['click_article_id'], df['click_timestamp']))

    user_item_time_df = click_df.groupby('user_id')[['click_article_id', 'click_timestamp']].apply(
        lambda x: make_item_time_pair(x)) \
        .reset_index().rename(columns={0: 'item_time_list'})
    user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))

    return user_item_time_dict


# 根据时间获取商品被点击的用户序列  {item1: [(user1, time1), (user2, time2)...]...}
# 这里的时间是用户点击当前商品的时间，好像没有直接的关系。
def get_item_user_time_dict(click_df):
    def make_user_time_pair(df):
        return list(zip(df['user_id'], df['click_timestamp']))

    click_df = click_df.sort_values('click_timestamp')
    item_user_time_df = click_df.groupby('click_article_id')[['user_id', 'click_timestamp']].apply(
        lambda x: make_user_time_pair(x)) \
        .reset_index().rename(columns={0: 'user_time_list'})

    item_user_time_dict = dict(zip(item_user_time_df['click_article_id'], item_user_time_df['user_time_list']))
    return item_user_time_dict


# 获取当前数据的历史点击和最后一次点击
def get_hist_and_last_click(all_click):
    all_click = all_click.sort_values(by=['user_id', 'click_timestamp'])
    click_last_df = all_click.groupby('user_id').tail(1)

    # 如果用户只有一个点击，hist为空了，会导致训练的时候这个用户不可见，此时默认泄露一下
    def hist_func(user_df):
        if len(user_df) == 1:
            return user_df
        else:
            return user_df[:-1]

    click_hist_df = all_click.groupby('user_id').apply(hist_func).reset_index(drop=True)

    return click_hist_df, click_last_df


# 获取文章id对应的基本属性，保存成字典的形式，方便后面召回阶段，冷启动阶段直接使用
def get_item_info_dict(item_info_df):
    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    item_info_df['created_at_ts'] = item_info_df[['created_at_ts']].apply(max_min_scaler)

    item_type_dict = dict(zip(item_info_df['click_article_id'], item_info_df['category_id']))
    item_words_dict = dict(zip(item_info_df['click_article_id'], item_info_df['words_count']))
    item_created_time_dict = dict(zip(item_info_df['click_article_id'], item_info_df['created_at_ts']))

    return item_type_dict, item_words_dict, item_created_time_dict


def get_user_hist_item_info_dict(all_click):
    # 获取user_id对应的用户历史点击文章类型的集合字典
    user_hist_item_typs = all_click.groupby('user_id')['category_id'].agg(set).reset_index()
    user_hist_item_typs_dict = dict(zip(user_hist_item_typs['user_id'], user_hist_item_typs['category_id']))

    # 获取user_id对应的用户点击文章的集合
    user_hist_item_ids_dict = all_click.groupby('user_id')['click_article_id'].agg(set).reset_index()
    user_hist_item_ids_dict = dict(zip(user_hist_item_ids_dict['user_id'], user_hist_item_ids_dict['click_article_id']))

    # 获取user_id对应的用户历史点击的文章的平均字数字典
    user_hist_item_words = all_click.groupby('user_id')['words_count'].agg('mean').reset_index()
    user_hist_item_words_dict = dict(zip(user_hist_item_words['user_id'], user_hist_item_words['words_count']))

    # 获取user_id对应的用户最后一次点击的文章的创建时间
    all_click_ = all_click.sort_values('click_timestamp')
    user_last_item_created_time = all_click_.groupby('user_id')['created_at_ts'].apply(
        lambda x: x.iloc[-1]).reset_index()

    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    user_last_item_created_time['created_at_ts'] = user_last_item_created_time[['created_at_ts']].apply(max_min_scaler)

    user_last_item_created_time_dict = dict(zip(user_last_item_created_time['user_id'], \
                                                user_last_item_created_time['created_at_ts']))

    return user_hist_item_typs_dict, user_hist_item_ids_dict, user_hist_item_words_dict, user_last_item_created_time_dict


# 获取近期点击最多的文章
def get_item_topk_click(click_df, k):
    topk_click = click_df['click_article_id'].value_counts().index[:k]
    return topk_click


# 获取文章的属性信息，保存成字典的形式方便查询
item_type_dict, item_words_dict, item_created_time_dict = get_item_info_dict(item_info_df)

# 定义一个多路召回的字典，将各路召回的结果都保存在这个字典当中
user_multi_recall_dict = {'itemcf_sim_itemcf_recall': {},
                          'embedding_sim_item_recall': {},
                          'youtubednn_recall': {},
                          'youtubednn_usercf_recall': {},
                          'cold_start_recall': {}}

# 提取最后一次点击作为召回评估，如果不需要做召回评估直接使用全量的训练集进行召回(线下验证模型)
# 如果不是召回评估，直接使用全量数据进行召回，不用将最后一次提取出来
trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)


# 依次评估召回的前10, 20, 30, 40, 50个文章中的击中率
def metrics_recall(user_recall_items_dict, trn_last_click_df, topk=5):
    last_click_item_dict = dict(zip(trn_last_click_df['user_id'], trn_last_click_df['click_article_id']))
    user_num = len(user_recall_items_dict)

    for k in range(10, topk + 1, 10):
        hit_num = 0
        for user, item_list in user_recall_items_dict.items():
            # 获取前k个召回的结果
            tmp_recall_items = [x[0] for x in user_recall_items_dict[user][:k]]
            if last_click_item_dict[user] in set(tmp_recall_items):
                hit_num += 1

        hit_rate = round(hit_num * 1.0 / user_num, 5)
        print(' topk: ', k, ' : ', 'hit_num: ', hit_num, 'hit_rate: ', hit_rate, 'user_num : ', user_num)


def itemcf_sim(df, item_created_time_dict):
    """
        文章与文章之间的相似性矩阵计算
        :param df: 数据表
        :item_created_time_dict:  文章创建时间的字典
        return : 文章与文章的相似性矩阵

        思路: 基于物品的协同过滤(详细请参考上一期推荐系统基础的组队学习) + 关联规则
    """

    user_item_time_dict = get_user_item_time(df)

    # 计算物品相似度
    i2i_sim = {}
    item_cnt = defaultdict(int)
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        # 在基于商品的协同过滤优化的时候可以考虑时间因素
        for loc1, (i, i_click_time) in enumerate(item_time_list):
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})
            for loc2, (j, j_click_time) in enumerate(item_time_list):
                if (i == j):
                    continue

                # 考虑文章的正向顺序点击和反向顺序点击
                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                # 位置信息权重，其中的参数可以调节
                loc_weight = loc_alpha * (0.9 ** (np.abs(loc2 - loc1) - 1))
                # 点击时间权重，其中的参数可以调节
                click_time_weight = np.exp(0.7 ** np.abs(i_click_time - j_click_time))
                # 两篇文章创建时间的权重，其中的参数可以调节
                created_time_weight = np.exp(0.8 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))
                i2i_sim[i].setdefault(j, 0)
                # 考虑多种因素的权重计算最终的文章之间的相似度
                i2i_sim[i][j] += loc_weight * click_time_weight * created_time_weight / math.log(
                    len(item_time_list) + 1)

    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])

    # 将得到的相似性矩阵保存到本地
    pickle.dump(i2i_sim_, open(save_path + 'itemcf_i2i_sim.pkl', 'wb'))

    return i2i_sim_


def get_user_activate_degree_dict(all_click_df):
    all_click_df_ = all_click_df.groupby('user_id')['click_article_id'].count().reset_index()

    # 用户活跃度归一化
    mm = MinMaxScaler()
    all_click_df_['click_article_id'] = mm.fit_transform(all_click_df_[['click_article_id']])
    user_activate_degree_dict = dict(zip(all_click_df_['user_id'], all_click_df_['click_article_id']))

    return user_activate_degree_dict


def usercf_sim(all_click_df, user_activate_degree_dict):
    """
        用户相似性矩阵计算
        :param all_click_df: 数据表
        :param user_activate_degree_dict: 用户活跃度的字典
        return 用户相似性矩阵

        思路: 基于用户的协同过滤(详细请参考上一期推荐系统基础的组队学习) + 关联规则
    """
    item_user_time_dict = get_item_user_time_dict(all_click_df)

    u2u_sim = {}
    user_cnt = defaultdict(int)
    for item, user_time_list in tqdm(item_user_time_dict.items()):
        for u, click_time in user_time_list:
            user_cnt[u] += 1
            u2u_sim.setdefault(u, {})
            for v, click_time in user_time_list:
                u2u_sim[u].setdefault(v, 0)
                if u == v:
                    continue
                # 用户平均活跃度作为活跃度的权重，这里的式子也可以改善
                activate_weight = 100 * 0.5 * (user_activate_degree_dict[u] + user_activate_degree_dict[v])
                u2u_sim[u][v] += activate_weight / math.log(len(user_time_list) + 1)

    u2u_sim_ = u2u_sim.copy()
    for u, related_users in u2u_sim.items():
        for v, wij in related_users.items():
            u2u_sim_[u][v] = wij / math.sqrt(user_cnt[u] * user_cnt[v])

    # 将得到的相似性矩阵保存到本地
    pickle.dump(u2u_sim_, open(save_path + 'usercf_u2u_sim.pkl', 'wb'))

    return u2u_sim_


# 由于usercf计算时候太耗费内存了，这里就不直接运行了
# 如果是采样的话，是可以运行的
user_activate_degree_dict = get_user_activate_degree_dict(all_click_df)
u2u_sim = usercf_sim(all_click_df, user_activate_degree_dict)


# 向量检索相似度计算
# topk指的是每个item, faiss搜索后返回最相似的topk个item
def embdding_sim(click_df, item_emb_df, save_path, topk):
    """
        基于内容的文章embedding相似性矩阵计算
        :param click_df: 数据表
        :param item_emb_df: 文章的embedding
        :param save_path: 保存路径
        :patam topk: 找最相似的topk篇
        return 文章相似性矩阵

        思路: 对于每一篇文章， 基于embedding的相似性返回topk个与其最相似的文章， 只不过由于文章数量太多，这里用了faiss进行加速
    """

    # 文章索引与文章id的字典映射
    item_idx_2_rawid_dict = dict(zip(item_emb_df.index, item_emb_df['article_id']))

    item_emb_cols = [x for x in item_emb_df.columns if 'emb' in x]
    item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols].values, dtype=np.float32)
    # 向量进行单位化
    item_emb_np = item_emb_np / np.linalg.norm(item_emb_np, axis=1, keepdims=True)

    # 建立faiss索引
    item_index = faiss.IndexFlatIP(item_emb_np.shape[1])
    item_index.add(item_emb_np)
    # 相似度查询，给每个索引位置上的向量返回topk个item以及相似度
    sim, idx = item_index.search(item_emb_np, topk)  # 返回的是列表

    # 将向量检索的结果保存成原始id的对应关系
    item_sim_dict = collections.defaultdict(dict)
    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(range(len(item_emb_np)), sim, idx)):
        target_raw_id = item_idx_2_rawid_dict[target_idx]
        # 从1开始是为了去掉商品本身, 所以最终获得的相似商品只有topk-1
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
            rele_raw_id = item_idx_2_rawid_dict[rele_idx]
            item_sim_dict[target_raw_id][rele_raw_id] = item_sim_dict.get(target_raw_id, {}).get(rele_raw_id,
                                                                                                 0) + sim_value

    # 保存i2i相似度矩阵
    pickle.dump(item_sim_dict, open(save_path + 'emb_i2i_sim.pkl', 'wb'))

    return item_sim_dict


# 获取双塔召回时的训练验证数据
# negsample指的是通过滑窗构建样本的时候，负样本的数量
def gen_data_set(data, negsample=0, item_feature_cols=None):
    """
    生成训练和测试数据集
    
    Args:
        data: 点击数据，包含 user_id, click_article_id, click_timestamp 等
        negsample: 负样本数量
        item_feature_cols: 物品辅助特征列名列表（如 ['category_id']），用于保存物品特征
    """
    data.sort_values("click_timestamp", inplace=True)
    item_ids = data['click_article_id'].unique()
    
    # 如果没有指定物品特征列，默认为空列表
    if item_feature_cols is None:
        item_feature_cols = []

    train_set = []
    test_set = []
    for reviewerID, hist in tqdm(data.groupby('user_id')):
        pos_list = hist['click_article_id'].tolist()
        
        # 保存物品的辅助特征（如果有）
        item_features_dict = {}
        if item_feature_cols:
            for item_id in pos_list:
                item_row = hist[hist['click_article_id'] == item_id].iloc[0]
                item_features_dict[item_id] = {feat: item_row[feat] for feat in item_feature_cols if feat in hist.columns}

        if negsample > 0:
            candidate_set = list(set(item_ids) - set(pos_list))  # 用户没看过的文章里面选择负样本
            neg_list = np.random.choice(candidate_set, size=len(pos_list) * negsample, replace=True)  # 对于每个正样本，选择n个负样本

        # 长度只有一个的时候，需要把这条数据也放到训练集中，不然的话最终学到的embedding就会有缺失
        if len(pos_list) == 1:
            item_id = pos_list[0]
            item_feat = item_features_dict.get(item_id, {}) if item_feature_cols else {}
            train_set.append((reviewerID, [item_id], item_id, 1, len(pos_list), item_feat))
            test_set.append((reviewerID, [item_id], item_id, 1, len(pos_list), item_feat))

        # 滑窗构造正负样本
        for i in range(1, len(pos_list)):
            hist_items = pos_list[:i]
            pos_item = pos_list[i]
            pos_item_feat = item_features_dict.get(pos_item, {}) if item_feature_cols else {}

            if i != len(pos_list) - 1:
                train_set.append((reviewerID, hist_items[::-1], pos_item, 1,
                                  len(hist_items[::-1]), pos_item_feat))  # 正样本 [user_id, his_item, pos_item, label, len(his_item), item_feat]
                for negi in range(negsample):
                    neg_item = neg_list[i * negsample + negi]
                    # 对于负样本，尝试从数据中获取特征，如果不存在则使用默认值
                    neg_item_feat = {}
                    if item_feature_cols:
                        neg_item_rows = data[data['click_article_id'] == neg_item]
                        if len(neg_item_rows) > 0:
                            neg_item_row = neg_item_rows.iloc[0]
                            neg_item_feat = {feat: neg_item_row[feat] for feat in item_feature_cols if feat in neg_item_rows.columns}
                        else:
                            # 如果负样本不在数据中，使用默认值（OOV）
                            neg_item_feat = {feat: -1 if feat == 'category_id' else 0 for feat in item_feature_cols}
                    train_set.append((reviewerID, hist_items[::-1], neg_item, 0,
                                      len(hist_items[::-1]), neg_item_feat))  # 负样本
            else:
                # 将最长的那一个序列长度作为测试数据
                test_set.append((reviewerID, hist_items[::-1], pos_item, 1, len(hist_items[::-1]), pos_item_feat))

    random.shuffle(train_set)
    random.shuffle(test_set)

    return train_set, test_set


# 将输入的数据进行padding，使得序列特征的长度都一致
def gen_model_input(train_set, user_profile, seq_max_len, item_feature_cols=None):
    """
    生成模型输入数据
    
    Args:
        train_set: 训练数据集，每个样本为 (user_id, hist_items, item_id, label, hist_len, item_feat_dict)
        user_profile: 用户画像
        seq_max_len: 序列最大长度
        item_feature_cols: 物品辅助特征列名列表（如 ['category_id']）
    """
    train_uid = np.array([line[0] for line in train_set])
    train_seq = [line[1] for line in train_set]
    train_iid = np.array([line[2] for line in train_set])
    train_label = np.array([line[3] for line in train_set])
    train_hist_len = np.array([line[4] for line in train_set])
    
    # 如果没有指定物品特征列，默认为空列表
    if item_feature_cols is None:
        item_feature_cols = []

    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)
    train_model_input = {"user_id": train_uid, "click_article_id": train_iid, "hist_article_id": train_seq_pad,
                         "hist_len": train_hist_len}
    
    # 添加物品辅助特征
    if item_feature_cols:
        for feat in item_feature_cols:
            # 从 train_set 中提取物品特征
            feat_values = []
            for line in train_set:
                item_feat_dict = line[5] if len(line) > 5 else {}
                feat_value = item_feat_dict.get(feat, -1 if feat == 'category_id' else 0)
                feat_values.append(feat_value)
            train_model_input[feat] = np.array(feat_values)

    return train_model_input, train_label


def youtubednn_u2i_dict(data, item_info_df=None, topk=20, use_aux_features=True):
    """
    YouTubeDNN 召回函数（双塔模型）
    
    Args:
        data: 点击数据，需要包含 user_id, click_article_id, click_timestamp
        item_info_df: 物品信息表，包含 category_id 等（可选，用于处理冷启动）
        topk: 召回数量
        use_aux_features: 是否使用辅助特征（物品特征）来帮助处理冷启动
    """
    # 拷贝数据，避免修改原始 DataFrame
    data = data.copy()
    
    SEQ_LEN = 30  # 用户点击序列的长度，短的填充，长的截断

    # 添加物品辅助特征（用于处理新物品的冷启动问题）
    item_aux_features = []
    if use_aux_features and item_info_df is not None:
        # 合并物品特征（category_id）
        if 'category_id' in item_info_df.columns:
            data = data.merge(item_info_df[['click_article_id', 'category_id']], 
                            on='click_article_id', how='left')
            # 填充缺失值：-1 表示未知类别（新物品或缺失信息）
            data['category_id'] = data['category_id'].fillna(-1).astype(int)
            item_aux_features.append('category_id')
            print(f"✓ 已添加物品辅助特征: category_id（用于处理新物品冷启动）")
        else:
            print(f"⚠️ 警告: item_info_df 中未找到 category_id 列，跳过物品辅助特征")
    elif not use_aux_features:
        print(f"ℹ️ 未启用辅助特征，仅使用 ID 特征（可能无法处理新用户/新物品）")
    
    user_profile_ = data[["user_id"]].drop_duplicates('user_id')
    item_profile_ = data[["click_article_id"]].drop_duplicates('click_article_id')

    # 特征编码
    # 基础特征：user_id, click_article_id（必需）
    # 辅助特征：物品特征（category_id），帮助处理新物品的冷启动
    features = ["click_article_id", "user_id"] + item_aux_features
    feature_max_idx = {}
    label_encoders = {}  # 保存编码器，以便在线服务时使用

    for feature in features:
        lbe = LabelEncoder()
        
        if feature == 'category_id':
            # category_id 的特殊处理：-1 表示未知类别（新物品）
            unique_values = sorted(data[feature].unique())
            if -1 in unique_values:
                unique_values.remove(-1)
                unique_values = [-1] + unique_values  # -1 放在最前面，编码为 0（OOV token）
            lbe.fit(unique_values)
            data[feature] = data[feature].apply(lambda x: x if x in lbe.classes_ else -1)
            data[feature] = lbe.transform(data[feature])
            # category_id 的 -1 已经是 OOV，所以只需要 +1
            feature_max_idx[feature] = data[feature].max() + 1
        else:
            # 标准 ID 特征编码
            data[feature] = lbe.fit_transform(data[feature])
            # 为 OOV 预留一个索引（通常是 max_index + 1）
            # 在在线服务时，新用户/物品可以使用这个索引，使用默认 embedding
            feature_max_idx[feature] = data[feature].max() + 2  # +2：正常索引 + OOV索引
        
        label_encoders[feature] = lbe
    
    # 保存 LabelEncoder，以便在线服务时使用
    # 在线服务时的使用方式：
    # 1. 加载 LabelEncoder: label_encoders = pickle.load(open(save_path + 'youtube_label_encoders.pkl', 'rb'))
    # 2. 对于已知用户/物品: encoded_id = label_encoders['user_id'].transform([user_id])[0]
    # 3. 对于新用户/物品（OOV）: 
    #    - user_id/click_article_id: 使用 feature_max_idx[feature] - 1 作为 OOV 索引
    #    - category_id: 使用 -1（编码为 0）作为 OOV 索引
    pickle.dump(label_encoders, open(save_path + 'youtube_label_encoders.pkl', 'wb'))
    pickle.dump(feature_max_idx, open(save_path + 'youtube_feature_max_idx.pkl', 'wb'))  # 保存 vocabulary_size
    print(f"\n✓ LabelEncoder 已保存到 {save_path}youtube_label_encoders.pkl")
    print(f"✓ Vocabulary size 已保存到 {save_path}youtube_feature_max_idx.pkl")
    print(f"\n使用的特征：")
    print(f"  基础特征: user_id, click_article_id")
    if item_aux_features:
        print(f"  物品辅助特征: {item_aux_features}（帮助处理新物品冷启动）")
    print(f"\n⚠️ 重要提示：")
    print(f"1. 如果在采样数据上训练，只能对采样中的用户/物品进行召回")
    print(f"2. 对于新用户：使用 OOV token（feature_max_idx['user_id'] - 1）")
    print(f"3. 对于新物品：")
    print(f"   - 如果知道 category_id：使用 category_id 特征帮助泛化")
    print(f"   - 如果不知道 category_id：使用 OOV token（feature_max_idx['click_article_id'] - 1）")
    print(f"4. 工业界最佳实践：使用全量数据（或至少大部分数据）训练 LabelEncoder，确保覆盖大部分用户/物品")

    # 提取user和item的画像，这里具体选择哪些特征还需要进一步的分析和考虑
    user_profile = data[["user_id"]].drop_duplicates('user_id')
    item_profile = data[["click_article_id"]].drop_duplicates('click_article_id')

    user_index_2_rawid = dict(zip(user_profile['user_id'], user_profile_['user_id']))
    item_index_2_rawid = dict(zip(item_profile['click_article_id'], item_profile_['click_article_id']))

    # 划分训练和测试集
    # 由于深度学习需要的数据量通常都是非常大的，所以为了保证召回的效果，往往会通过滑窗的形式扩充训练样本
    train_set, test_set = gen_data_set(data, 0, item_feature_cols=item_aux_features)
    # 整理输入数据，具体的操作可以看上面的函数
    train_model_input, train_label = gen_model_input(train_set, user_profile, SEQ_LEN, item_feature_cols=item_aux_features)
    test_model_input, test_label = gen_model_input(test_set, user_profile, SEQ_LEN, item_feature_cols=item_aux_features)

    # 确定Embedding的维度
    embedding_dim = 16

    # 将数据整理成模型可以直接输入的形式
    # 用户塔特征：user_id + 历史序列
    user_feature_columns = [SparseFeat('user_id', feature_max_idx['user_id'], embedding_dim),
                            VarLenSparseFeat(
                                SparseFeat('hist_article_id', feature_max_idx['click_article_id'], embedding_dim,
                                           embedding_name="click_article_id"), SEQ_LEN, 'mean', 'hist_len'), ]
    
    # 物品塔特征：click_article_id + 物品辅助特征
    item_feature_columns = [SparseFeat('click_article_id', feature_max_idx['click_article_id'], embedding_dim)]
    # 添加物品辅助特征到物品塔
    for feat in item_aux_features:
        item_feature_columns.append(SparseFeat(feat, feature_max_idx[feat], embedding_dim))

    # 模型的定义
    # num_sampled: 负采样时的样本数量
    # model = YoutubeDNN(user_feature_columns, item_feature_columns, num_sampled=5,
    #                    user_dnn_hidden_units=(64, embedding_dim))
    train_counter = Counter(train_model_input['click_article_id'])
    item_count = [train_counter.get(i, 0) for i in range(item_feature_columns[0].vocabulary_size)]
    sampler_config = NegativeSampler('frequency', num_sampled=5, item_name="click_article_id", item_count=item_count)

    import tensorflow as tf
    print(tf.__version__)
    # deepmatch 库的某些组件（如 get_item_embedding 中的 Lambda 层）在 Eager Execution 模式下不兼容
    # 因此必须使用 Graph 模式（禁用 Eager Execution）
    # 在 Graph 模式下，需要使用 K.function() 而不是 Model() 来提取 embedding
    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()

    # TensorFlow 1.x 才需要手动初始化 Session
    if tf.__version__ < '2.0.0':
        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())

    model = YoutubeDNN(user_feature_columns, item_feature_columns, user_dnn_hidden_units=(64, 16, embedding_dim),
                       sampler_config=sampler_config)
    # 模型编译
    model.compile(optimizer="adam", loss=sampledsoftmaxloss)

    # 模型训练，这里可以定义验证集的比例，如果设置为0的话就是全量数据直接进行训练
    history = model.fit(train_model_input, train_label, batch_size=256, epochs=5, verbose=1, validation_split=0.0)

    # 训练完模型之后,提取训练的Embedding，包括user端和item端
    test_user_model_input = test_model_input
    # 构建所有物品的输入（包含物品辅助特征）
    all_item_model_input = {"click_article_id": item_profile['click_article_id'].values}
    # 添加物品辅助特征
    if item_aux_features:
        # 需要从原始数据中获取物品特征（已编码）
        for feat in item_aux_features:
            feat_values = []
            for item_id in item_profile['click_article_id']:
                # 从 data 中查找该物品的特征值（已编码）
                item_rows = data[data['click_article_id'] == item_id]
                if len(item_rows) > 0:
                    feat_value = item_rows[feat].iloc[0]
                else:
                    # 如果找不到，使用 OOV 值
                    feat_value = -1 if feat == 'category_id' else 0
                    # 需要编码 OOV 值
                    if feat == 'category_id':
                        # category_id 的 -1 应该已经被编码为 0
                        feat_value = 0
                    else:
                        # 其他特征的 OOV 值
                        feat_value = feature_max_idx[feat] - 1
                feat_values.append(feat_value)
            all_item_model_input[feat] = np.array(feat_values)
    
    print(f"test_user_model_input length: {len(test_user_model_input)}")
    print(f"all_item_model_input length: {len(all_item_model_input)}")
    if item_aux_features:
        print(f"物品辅助特征: {item_aux_features}")

    # Model() 的作用：从已训练的完整模型中提取子模型，创建一个新的模型
    # 这个新模型只包含指定的输入层和输出层，用于单独获取user或item的embedding
    # 在 TensorFlow 2.x 中，即使禁用了 eager execution（使用 Graph 模式），
    # 只要保持输入和原模型一致，使用 Model() 创建子模型也是可以的
    # 关键是要确保输入格式正确（列表格式，按顺序对应模型的输入层）
    
    # 从输入层的名称提取特征名称
    def get_feature_name(input_layer):
        """从Tensor名称（如 'user_id:0'）中提取特征名称（如 'user_id'）"""
        name = input_layer.name
        # 去除可能的 ':0', ':1' 等后缀
        if ':' in name:
            name = name.split(':')[0]
        return name
    
    # 将字典格式转换为列表格式，按照 model.user_input 的顺序
    # model.user_input 的顺序：['user_id:0', 'hist_article_id:0', 'hist_len:0']
    # 对应的字典键：['user_id', 'hist_article_id', 'hist_len']
    def get_feature_data(input_layer, input_dict, aux_features=None):
        """从输入字典中提取对应特征的数据，确保维度正确"""
        if aux_features is None:
            aux_features = []
        feature_name = get_feature_name(input_layer)
        data = input_dict[feature_name]
        # 确保数据是 numpy 数组，并且维度正确
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        # 确保维度正确
        # user_id 和 hist_len 应该是 (batch_size, 1) 的形状
        if feature_name in ['user_id', 'hist_len']:
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            elif len(data.shape) > 2:
                # 如果维度过多，压缩到2D
                data = data.reshape(data.shape[0], -1)[:, :1]
        # hist_article_id 应该是 (batch_size, seq_len) 的形状，确保是2D
        elif feature_name == 'hist_article_id':
            if len(data.shape) == 1:
                # 如果是一维，可能需要重新reshape
                data = data.reshape(-1, 1)
            elif len(data.shape) > 2:
                # 如果维度过多（可能是3D），压缩到2D
                # 保持 batch_size 维度，将其他维度展平
                data = data.reshape(data.shape[0], -1)
        # click_article_id 和物品辅助特征应该是 (batch_size, 1) 的形状
        elif feature_name == 'click_article_id' or feature_name in aux_features:
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            elif len(data.shape) > 2:
                data = data.reshape(data.shape[0], -1)[:, :1]
        
        return data
    
    # deepmatch 库需要在 Graph 模式下运行（由于 get_item_embedding 中的 Lambda 层在 Eager 模式下不兼容）
    # 在 Graph 模式下，必须使用 K.function() 而不是 Model() 来提取 embedding
    # 因为 Model() 会创建新的计算图，无法访问原始模型的变量权重
    
    # 将字典格式转换为列表格式，按照 model.user_input 的顺序
    test_user_model_input_list = [get_feature_data(inp, test_user_model_input, aux_features=item_aux_features) for inp in model.user_input]
    all_item_model_input_list = [get_feature_data(inp, all_item_model_input, aux_features=item_aux_features) for inp in model.item_input]
    
    # 使用 K.function 创建函数来获取 embedding（Graph 模式下必须使用这个方法）
    # K.function 在同一个计算图中执行，可以访问原始模型的权重
    user_embedding_func = K.function(model.user_input, [model.user_embedding])
    item_embedding_func = K.function(model.item_input, [model.item_embedding])
    
    # 使用函数来获取 embedding，分批处理以避免内存问题
    batch_size = 2 ** 12
    n_samples = len(test_user_model_input_list[0])
    
    # 分批处理 user embedding
    user_embs_list = []
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_inputs = [inp[i:end_idx] for inp in test_user_model_input_list]
        batch_emb = user_embedding_func(batch_inputs)[0]
        user_embs_list.append(batch_emb)
    user_embs = np.vstack(user_embs_list)
    
    # 分批处理 item embedding
    n_items = len(all_item_model_input_list[0])
    item_embs_list = []
    for i in range(0, n_items, batch_size):
        end_idx = min(i + batch_size, n_items)
        batch_inputs = [inp[i:end_idx] for inp in all_item_model_input_list]
        batch_emb = item_embedding_func(batch_inputs)[0]
        item_embs_list.append(batch_emb)
    item_embs = np.vstack(item_embs_list)

    # embedding保存之前归一化一下
    user_embs = user_embs / np.linalg.norm(user_embs, axis=1, keepdims=True)
    item_embs = item_embs / np.linalg.norm(item_embs, axis=1, keepdims=True)

    # embedding保存之前归一化一下
    user_embs = user_embs / np.linalg.norm(user_embs, axis=1, keepdims=True)
    item_embs = item_embs / np.linalg.norm(item_embs, axis=1, keepdims=True)

    # 将Embedding转换成字典的形式方便查询
    raw_user_id_emb_dict = {user_index_2_rawid[k]: \
                                v for k, v in zip(user_profile['user_id'], user_embs)}
    raw_item_id_emb_dict = {item_index_2_rawid[k]: \
                                v for k, v in zip(item_profile['click_article_id'], item_embs)}
    # 将Embedding保存到本地
    pickle.dump(raw_user_id_emb_dict, open(save_path + 'user_youtube_emb.pkl', 'wb'))
    pickle.dump(raw_item_id_emb_dict, open(save_path + 'item_youtube_emb.pkl', 'wb'))

    # faiss紧邻搜索，通过user_embedding 搜索与其相似性最高的topk个item
    index = faiss.IndexFlatIP(embedding_dim)

    # 示例数据
    item_embs = np.random.rand(8787, 16).astype(np.float32)
    user_embs = np.random.rand(70274, 16).astype(np.float32)  # 确保维度一致

    # 创建 FAISS 索引
    num_features = item_embs.shape[1]
    index = faiss.IndexFlatIP(num_features)

    # 执行搜索
    topk = 20
    sim, idx = index.search(user_embs, topk)

    # 打印结果
    print("Similarities:", sim)
    print("Indices:", idx)

    # 上面已经进行了归一化，这里可以不进行归一化了
    #     faiss.normalize_L2(user_embs)
    #     faiss.normalize_L2(item_embs)
    index.add(item_embs)  # 将item向量构建索引
    sim, idx = index.search(np.ascontiguousarray(user_embs), topk)  # 通过user去查询最相似的topk个item
    print("Keys in item_index_2_rawid:", list(item_index_2_rawid.keys()))
    print("Sample idx:", idx[:5])  # 打印前几个索引

    user_recall_items_dict = collections.defaultdict(dict)
    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(test_user_model_input['user_id'], sim, idx)):
        target_raw_id = user_index_2_rawid[target_idx]
        # 从1开始是为了去掉商品本身, 所以最终获得的相似商品只有topk-1
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
            if rele_idx in item_index_2_rawid:
                rele_raw_id = item_index_2_rawid[rele_idx]
                user_recall_items_dict[target_raw_id][rele_raw_id] = user_recall_items_dict.get(target_raw_id, {}) \
                                                                         .get(rele_raw_id, 0) + sim_value
            else:
                print(f"Warning: {rele_idx} not found in item_index_2_rawid")

    user_recall_items_dict = {k: sorted(v.items(), key=lambda x: x[1], reverse=True) for k, v in
                              user_recall_items_dict.items()}
    # 将召回的结果进行排序

    # 保存召回的结果
    # 这里是直接通过向量的方式得到了召回结果，相比于上面的召回方法，上面的只是得到了i2i及u2u的相似性矩阵，还需要进行协同过滤召回才能得到召回结果
    # 可以直接对这个召回结果进行评估，为了方便可以统一写一个评估函数对所有的召回结果进行评估
    pickle.dump(user_recall_items_dict, open(save_path + 'youtube_u2i_dict.pkl', 'wb'))
    return user_recall_items_dict


# 由于这里需要做召回评估，所以讲训练集中的最后一次点击都提取了出来
metric_recall = True
if not metric_recall:
    user_multi_recall_dict['youtubednn_recall'] = youtubednn_u2i_dict(all_click_df, item_info_df=item_info_df, topk=20, use_aux_features=True)
else:
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
    user_multi_recall_dict['youtubednn_recall'] = youtubednn_u2i_dict(trn_hist_click_df, item_info_df=item_info_df, topk=20, use_aux_features=True)
    # 召回效果评估
    metrics_recall(user_multi_recall_dict['youtubednn_recall'], trn_last_click_df, topk=20)
metric_recall = False

# 基于商品的召回i2i
def item_based_recommend(user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, item_topk_click,
                         item_created_time_dict, emb_i2i_sim):
    """
        基于文章协同过滤的召回
        :param user_id: 用户id
        :param user_item_time_dict: 字典, 根据点击时间获取用户的点击文章序列   {user1: [(item1, time1), (item2, time2)..]...}
        :param i2i_sim: 字典，文章相似性矩阵
        :param sim_item_topk: 整数， 选择与当前文章最相似的前k篇文章
        :param recall_item_num: 整数， 最后的召回文章数量
        :param item_topk_click: 列表，点击次数最多的文章列表，用户召回补全
        :param emb_i2i_sim: 字典基于内容embedding算的文章相似矩阵

        return: 召回的文章列表 [(item1, score1), (item2, score2)...]

    """
    # 获取用户历史交互的文章
    user_hist_items = user_item_time_dict[user_id]
    user_hist_items_ = {user_id for user_id, _ in user_hist_items}

    if 3941 not in i2i_sim:
        print("Key 3941 not found in i2i_sim")

    item_rank = {}
    for loc, (i, click_time) in enumerate(user_hist_items):
        # 检查文章是否在相似度矩阵中，如果不在则跳过
        if i not in i2i_sim:
            continue
        for j, wij in sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
            if j in user_hist_items_:
                continue

            # 文章创建时间差权重
            created_time_weight = np.exp(0.8 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))
            # 相似文章和历史点击文章序列中历史文章所在的位置权重
            loc_weight = (0.9 ** (len(user_hist_items) - loc))

            content_weight = 1.0
            if emb_i2i_sim.get(i, {}).get(j, None) is not None:
                content_weight += emb_i2i_sim[i][j]
            if emb_i2i_sim.get(j, {}).get(i, None) is not None:
                content_weight += emb_i2i_sim[j][i]

            item_rank.setdefault(j, 0)
            item_rank[j] += created_time_weight * loc_weight * content_weight * wij
    # 不足10个，用热门商品补全
    if len(item_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in item_rank:  # 修复：检查key而不是items，填充的item应该不在原来的列表中
                continue
            item_rank[item] = - i - 100  # 随便给个负数就行
            if len(item_rank) == recall_item_num:
                break

    item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]

    return item_rank


# 先进行itemcf召回, 为了召回评估，所以提取最后一次点击

if metric_recall:
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
else:
    trn_hist_click_df = all_click_df

user_recall_items_dict = collections.defaultdict(dict)
user_item_time_dict = get_user_item_time(trn_hist_click_df)

i2i_sim = pickle.load(open(save_path + 'itemcf_i2i_sim.pkl', 'rb'))
emb_i2i_sim = pickle.load(open(save_path + 'emb_i2i_sim.pkl', 'rb'))

# 增加召回数量和覆盖率以提高召回命中率
sim_item_topk = 50  # 从20提高到50，每个历史item召回更多相似文章
recall_item_num = 50  # 从10提高到50，提高召回覆盖率
item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)

for user in tqdm(trn_hist_click_df['user_id'].unique()):
    user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, i2i_sim, sim_item_topk,
                                                        recall_item_num, item_topk_click, item_created_time_dict,
                                                        emb_i2i_sim)

user_multi_recall_dict['itemcf_sim_itemcf_recall'] = user_recall_items_dict
pickle.dump(user_multi_recall_dict['itemcf_sim_itemcf_recall'], open(save_path + 'itemcf_recall_dict.pkl', 'wb'))

if metric_recall:
    # 召回效果评估
    metrics_recall(user_multi_recall_dict['itemcf_sim_itemcf_recall'], trn_last_click_df, topk=recall_item_num)

# 这里是为了召回评估，所以提取最后一次点击
if metric_recall:
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
else:
    trn_hist_click_df = all_click_df

user_recall_items_dict = collections.defaultdict(dict)
user_item_time_dict = get_user_item_time(trn_hist_click_df)
i2i_sim = pickle.load(open(save_path + 'emb_i2i_sim.pkl', 'rb'))

# Embedding相似度召回：增加召回数量以提高覆盖率
sim_item_topk = 50  # 从20提高到50
recall_item_num = 50  # 从10提高到50

item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)

for user in tqdm(trn_hist_click_df['user_id'].unique()):
    user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, i2i_sim, sim_item_topk,
                                                        recall_item_num, item_topk_click, item_created_time_dict,
                                                        emb_i2i_sim)

user_multi_recall_dict['embedding_sim_item_recall'] = user_recall_items_dict
pickle.dump(user_multi_recall_dict['embedding_sim_item_recall'],
            open(save_path + 'embedding_sim_item_recall.pkl', 'wb'))

if metric_recall:
    # 召回效果评估
    metrics_recall(user_multi_recall_dict['embedding_sim_item_recall'], trn_last_click_df, topk=recall_item_num)

# 这里是为了召回评估，所以提取最后一次点击
if metric_recall:
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
else:
    trn_hist_click_df = all_click_df

user_recall_items_dict = collections.defaultdict(dict)
user_item_time_dict = get_user_item_time(trn_hist_click_df)
i2i_sim = pickle.load(open(save_path + 'emb_i2i_sim.pkl', 'rb'))

# Embedding相似度召回：增加召回数量以提高覆盖率
sim_item_topk = 50  # 从20提高到50
recall_item_num = 50  # 从10提高到50

item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)

for user in tqdm(trn_hist_click_df['user_id'].unique()):
    user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, i2i_sim, sim_item_topk,
                                                        recall_item_num, item_topk_click, item_created_time_dict,
                                                        emb_i2i_sim)

user_multi_recall_dict['embedding_sim_item_recall'] = user_recall_items_dict
pickle.dump(user_multi_recall_dict['embedding_sim_item_recall'],
            open(save_path + 'embedding_sim_item_recall.pkl', 'wb'))

if metric_recall:
    # 召回效果评估
    metrics_recall(user_multi_recall_dict['embedding_sim_item_recall'], trn_last_click_df, topk=recall_item_num)


# 基于用户的召回 u2u2i
def user_based_recommend(user_id, user_item_time_dict, u2u_sim, sim_user_topk, recall_item_num,
                         item_topk_click, item_created_time_dict, emb_i2i_sim):
    """
        基于文章协同过滤的召回
        :param user_id: 用户id
        :param user_item_time_dict: 字典, 根据点击时间获取用户的点击文章序列   {user1: [(item1, time1), (item2, time2)..]...}
        :param u2u_sim: 字典，文章相似性矩阵
        :param sim_user_topk: 整数， 选择与当前用户最相似的前k个用户
        :param recall_item_num: 整数， 最后的召回文章数量
        :param item_topk_click: 列表，点击次数最多的文章列表，用户召回补全
        :param item_created_time_dict: 文章创建时间列表
        :param emb_i2i_sim: 字典基于内容embedding算的文章相似矩阵

        return: 召回的文章列表 [(item1, score1), (item2, score2)...]
    """

    # 历史交互
    user_item_time_list = user_item_time_dict[user_id]  # {item1: time1, item2: time2...}
    user_hist_items = set([i for i, t in user_item_time_list])  # 存在一个用户与某篇文章的多次交互， 这里得去重

    items_rank = {}
    try:
        for sim_u, wuv in sorted(u2u_sim[user_id].items(), key=lambda x: x[1], reverse=True)[:sim_user_topk]:
            for i, click_time in user_item_time_dict[sim_u]:
                if i in user_hist_items:
                    continue
                items_rank.setdefault(i, 0)

                loc_weight = 1.0
                content_weight = 1.0
                created_time_weight = 1.0

                # 当前文章与该用户看的历史文章进行一个权重交互
                for loc, (j, click_time) in enumerate(user_item_time_list):
                    # 点击时的相对位置权重
                    loc_weight += 0.9 ** (len(user_item_time_list) - loc)
                    # 内容相似性权重
                    if emb_i2i_sim.get(i, {}).get(j, None) is not None:
                        content_weight += emb_i2i_sim[i][j]
                    if emb_i2i_sim.get(j, {}).get(i, None) is not None:
                        content_weight += emb_i2i_sim[j][i]

                    # 创建时间差权重
                    created_time_weight += np.exp(0.8 * np.abs(item_created_time_dict[i] - item_created_time_dict[j]))

                items_rank[i] += loc_weight * content_weight * created_time_weight * wuv
    except KeyError as e:
        print()

    # 热度补全
    if len(items_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in items_rank:  # 修复：检查key而不是items，填充的item应该不在原来的列表中
                continue
            items_rank[item] = - i - 100  # 随便给个复数就行
            if len(items_rank) == recall_item_num:
                break

    items_rank = sorted(items_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]

    return items_rank


# 这里是为了召回评估，所以提取最后一次点击
# 由于usercf中计算user之间的相似度的过程太费内存了，全量数据这里就没有跑，跑了一个采样之后的数据
if metric_recall:
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
else:
    trn_hist_click_df = all_click_df

user_recall_items_dict = collections.defaultdict(dict)
user_item_time_dict = get_user_item_time(trn_hist_click_df)

u2u_sim = pickle.load(open(save_path + 'usercf_u2u_sim.pkl', 'rb'))

# UserCF召回：增加召回数量以提高覆盖率
sim_user_topk = 20
recall_item_num = 50  # 从10提高到50
item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)

for user in tqdm(trn_hist_click_df['user_id'].unique()):
    user_recall_items_dict[user] = user_based_recommend(user, user_item_time_dict, u2u_sim, sim_user_topk, \
                                                        recall_item_num, item_topk_click, item_created_time_dict,
                                                        emb_i2i_sim)

pickle.dump(user_recall_items_dict, open(save_path + 'usercf_u2u2i_recall.pkl', 'wb'))

if metric_recall:
    # 召回效果评估
    metrics_recall(user_recall_items_dict, trn_last_click_df, topk=recall_item_num)


# 使用Embedding的方式获取u2u的相似性矩阵
# topk指的是每个user, faiss搜索后返回最相似的topk个user
def u2u_embdding_sim(click_df, user_emb_dict, save_path, topk):
    user_list = []
    user_emb_list = []
    for user_id, user_emb in user_emb_dict.items():
        user_list.append(user_id)
        user_emb_list.append(user_emb)

    user_index_2_rawid_dict = {k: v for k, v in zip(range(len(user_list)), user_list)}

    user_emb_np = np.array(user_emb_list, dtype=np.float32)

    # 建立faiss索引
    user_index = faiss.IndexFlatIP(user_emb_np.shape[1])
    user_index.add(user_emb_np)
    # 相似度查询，给每个索引位置上的向量返回topk个item以及相似度
    sim, idx = user_index.search(user_emb_np, topk)  # 返回的是列表

    # 将向量检索的结果保存成原始id的对应关系
    user_sim_dict = collections.defaultdict(dict)
    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(range(len(user_emb_np)), sim, idx)):
        target_raw_id = user_index_2_rawid_dict[target_idx]
        # 从1开始是为了去掉商品本身, 所以最终获得的相似商品只有topk-1
        try:
            for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
                rele_raw_id = user_index_2_rawid_dict[rele_idx]
                user_sim_dict[target_raw_id][rele_raw_id] = user_sim_dict.get(target_raw_id, {}).get(rele_raw_id,
                                                                                                     0) + sim_value

        except KeyError as e:
            print()

    # 保存i2i相似度矩阵
    pickle.dump(user_sim_dict, open(save_path + 'youtube_u2u_sim.pkl', 'wb'))
    return user_sim_dict


# 读取YoutubeDNN过程中产生的user embedding, 然后使用faiss计算用户之间的相似度
# 这里需要注意，这里得到的user embedding其实并不是很好，因为YoutubeDNN中使用的是用户点击序列来训练的user embedding,
# 如果序列普遍都比较短的话，其实效果并不是很好
user_emb_dict = pickle.load(open(save_path + 'user_youtube_emb.pkl', 'rb'))
u2u_sim = u2u_embdding_sim(all_click_df, user_emb_dict, save_path, topk=10)

# 使用召回评估函数验证当前召回方式的效果
if metric_recall:
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
else:
    trn_hist_click_df = all_click_df

user_recall_items_dict = collections.defaultdict(dict)
user_item_time_dict = get_user_item_time(trn_hist_click_df)
u2u_sim = pickle.load(open(save_path + 'youtube_u2u_sim.pkl', 'rb'))

# YouTubeDNN+UserCF召回：增加召回数量以提高覆盖率
sim_user_topk = 20
recall_item_num = 50  # 从10提高到50

item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)
for user in tqdm(trn_hist_click_df['user_id'].unique()):
    user_recall_items_dict[user] = user_based_recommend(user, user_item_time_dict, u2u_sim, sim_user_topk, \
                                                        recall_item_num, item_topk_click, item_created_time_dict,
                                                        emb_i2i_sim)

user_multi_recall_dict['youtubednn_usercf_recall'] = user_recall_items_dict
pickle.dump(user_multi_recall_dict['youtubednn_usercf_recall'], open(save_path + 'youtubednn_usercf_recall.pkl', 'wb'))

if metric_recall:
    # 召回效果评估
    metrics_recall(user_multi_recall_dict['youtubednn_usercf_recall'], trn_last_click_df, topk=recall_item_num)

# 先进行itemcf召回，这里不需要做召回评估，这里只是一种策略
trn_hist_click_df = all_click_df

user_recall_items_dict = collections.defaultdict(dict)
user_item_time_dict = get_user_item_time(trn_hist_click_df)
i2i_sim = pickle.load(open(save_path + 'emb_i2i_sim.pkl', 'rb'))

sim_item_topk = 150
recall_item_num = 100  # 稍微召回多一点文章，便于后续的规则筛选

item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)
for user in tqdm(trn_hist_click_df['user_id'].unique()):
    user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, i2i_sim, sim_item_topk,
                                                        recall_item_num, item_topk_click, item_created_time_dict,
                                                        emb_i2i_sim)
pickle.dump(user_recall_items_dict, open(save_path + 'cold_start_items_raw_dict.pkl', 'wb'))


# 基于规则进行文章过滤
# 保留文章主题与用户历史浏览主题相似的文章
# 保留文章字数与用户历史浏览文章字数相差不大的文章
# 保留最后一次点击当天的文章
# 按照相似度返回最终的结果

def get_click_article_ids_set(all_click_df):
    return set(all_click_df.click_article_id.values)


def cold_start_items(user_recall_items_dict, user_hist_item_typs_dict, user_hist_item_words_dict, \
                     user_last_item_created_time_dict, item_type_dict, item_words_dict,
                     item_created_time_dict, click_article_ids_set, recall_item_num):
    """
        冷启动的情况下召回一些文章
        :param user_recall_items_dict: 基于内容embedding相似性召回来的很多文章， 字典， {user1: [item1, item2, ..], }
        :param user_hist_item_typs_dict: 字典， 用户点击的文章的主题映射
        :param user_hist_item_words_dict: 字典， 用户点击的历史文章的字数映射
        :param user_last_item_created_time_idct: 字典，用户点击的历史文章创建时间映射
        :param item_tpye_idct: 字典，文章主题映射
        :param item_words_dict: 字典，文章字数映射
        :param item_created_time_dict: 字典， 文章创建时间映射
        :param click_article_ids_set: 集合，用户点击过得文章, 也就是日志里面出现过的文章
        :param recall_item_num: 召回文章的数量， 这个指的是没有出现在日志里面的文章数量
    """

    cold_start_user_items_dict = {}
    for user, item_list in tqdm(user_recall_items_dict.items()):
        cold_start_user_items_dict.setdefault(user, [])
        for item, score in item_list:
            # 获取历史文章信息
            hist_item_type_set = user_hist_item_typs_dict[user]
            hist_mean_words = user_hist_item_words_dict[user]
            hist_last_item_created_time = user_last_item_created_time_dict[user]
            hist_last_item_created_time = datetime.fromtimestamp(hist_last_item_created_time)

            # 获取当前召回文章的信息
            curr_item_type = item_type_dict[item]
            curr_item_words = item_words_dict[item]
            curr_item_created_time = item_created_time_dict[item]
            curr_item_created_time = datetime.fromtimestamp(curr_item_created_time)

            # 首先，文章不能出现在用户的历史点击中， 然后根据文章主题，文章单词数，文章创建时间进行筛选
            if curr_item_type not in hist_item_type_set or \
                    item in click_article_ids_set or \
                    abs(curr_item_words - hist_mean_words) > 200 or \
                    abs((curr_item_created_time - hist_last_item_created_time).days) > 90:
                continue

            cold_start_user_items_dict[user].append((item, score))  # {user1: [(item1, score1), (item2, score2)..]...}

    # 需要控制一下冷启动召回的数量
    cold_start_user_items_dict = {k: sorted(v, key=lambda x: x[1], reverse=True)[:recall_item_num] \
                                  for k, v in cold_start_user_items_dict.items()}

    pickle.dump(cold_start_user_items_dict, open(save_path + 'cold_start_user_items_dict.pkl', 'wb'))

    return cold_start_user_items_dict


all_click_df_ = all_click_df.copy()
all_click_df_ = all_click_df_.merge(item_info_df, how='left', on='click_article_id')
user_hist_item_typs_dict, user_hist_item_ids_dict, user_hist_item_words_dict, user_last_item_created_time_dict = get_user_hist_item_info_dict(
    all_click_df_)
click_article_ids_set = get_click_article_ids_set(all_click_df)
# 需要注意的是
# 这里使用了很多规则来筛选冷启动的文章，所以前面再召回的阶段就应该尽可能的多召回一些文章，否则很容易被删掉
cold_start_user_items_dict = cold_start_items(user_recall_items_dict, user_hist_item_typs_dict,
                                              user_hist_item_words_dict, \
                                              user_last_item_created_time_dict, item_type_dict, item_words_dict, \
                                              item_created_time_dict, click_article_ids_set, recall_item_num)

user_multi_recall_dict['cold_start_recall'] = cold_start_user_items_dict


def combine_recall_results(user_multi_recall_dict, weight_dict=None, topk=25):
    final_recall_items_dict = {}

    # 对每一种召回结果按照用户进行归一化，方便后面多种召回结果，相同用户的物品之间权重相加
    def norm_user_recall_items_sim(sorted_item_list):
        # 如果冷启动中没有文章或者只有一篇文章，直接返回，出现这种情况的原因可能是冷启动召回的文章数量太少了，
        # 基于规则筛选之后就没有文章了, 这里还可以做一些其他的策略性的筛选
        if len(sorted_item_list) < 2:
            return sorted_item_list

        min_sim = sorted_item_list[-1][1]
        max_sim = sorted_item_list[0][1]

        norm_sorted_item_list = []
        for item, score in sorted_item_list:
            if max_sim > 0:
                norm_score = 1.0 * (score - min_sim) / (max_sim - min_sim) if max_sim > min_sim else 1.0
            else:
                norm_score = 0.0
            norm_sorted_item_list.append((item, norm_score))

        return norm_sorted_item_list

    print('多路召回合并...')
    for method, user_recall_items in tqdm(user_multi_recall_dict.items()):
        print(method + '...')
        # 在计算最终召回结果的时候，也可以为每一种召回结果设置一个权重
        if weight_dict == None:
            recall_method_weight = 1
        else:
            recall_method_weight = weight_dict[method]

        for user_id, sorted_item_list in user_recall_items.items():  # 进行归一化
            user_recall_items[user_id] = norm_user_recall_items_sim(sorted_item_list)

        for user_id, sorted_item_list in user_recall_items.items():
            # print('user_id')
            final_recall_items_dict.setdefault(user_id, {})
            for item, score in sorted_item_list:
                final_recall_items_dict[user_id].setdefault(item, 0)
                final_recall_items_dict[user_id][item] += recall_method_weight * score

    final_recall_items_dict_rank = {}
    # 多路召回时也可以控制最终的召回数量
    for user, recall_item_dict in final_recall_items_dict.items():
        final_recall_items_dict_rank[user] = sorted(recall_item_dict.items(), key=lambda x: x[1], reverse=True)[:topk]

    # 将多路召回后的最终结果字典保存到本地
    pickle.dump(final_recall_items_dict, open(os.path.join(save_path, 'final_recall_items_dict.pkl'), 'wb'))

    return final_recall_items_dict_rank


# 根据召回效果调整权重：YouTubeDNN U2I召回效果最好（命中率0.15%），提高其权重
# ItemCF召回效果较差（命中率0.02%），降低其权重
weight_dict = {'itemcf_sim_itemcf_recall': 0.5,  # 降低权重：命中率低，且文章ID不匹配
               'embedding_sim_item_recall': 1.0,
               'youtubednn_recall': 2.0,  # 提高权重：命中率最高（0.15%），召回文章在数据集中的比例93.8%
               'youtubednn_usercf_recall': 0.5,  # 降低权重：命中率0%，且文章ID不匹配
               'cold_start_recall': 1.0}

# 最终合并之后每个用户召回150个商品进行排序
final_recall_items_dict_rank = combine_recall_results(user_multi_recall_dict, weight_dict, topk=150)

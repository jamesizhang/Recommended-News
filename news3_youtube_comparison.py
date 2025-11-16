"""
YouTubeDNN 召回效果对比测试
对比使用辅助特征前后的召回命中率提升
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os, math, warnings, pickle
import faiss
import collections
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from deepctr.feature_column import SparseFeat, VarLenSparseFeat
from tensorflow.python.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
# 使用 funRec 的 YouTubeDNN 模型
import sys
import os
funrec_path = os.path.join(os.path.dirname(__file__), 'funRec', 'src')
sys.path.insert(0, funrec_path)

from funrec.models.youtubednn import build_youtubednn_model
from funrec.features.feature_column import FeatureColumn
from collections import Counter
import tensorflow as tf

# 保留 deepmatch 的导入用于基线版本（如果需要）
# from deepmatch.utils import sampledsoftmaxloss, NegativeSampler
# from deepmatch.models import *

warnings.filterwarnings('ignore')

data_path = '../data_raw/'
save_path = '../temp_results/'
metric_recall = True  # 启用召回评估

# 设置随机种子，确保采样结果一致
np.random.seed(42)
random.seed(42)

# ===================== 数据加载函数 =====================
def get_all_click_df(data_path='../data_raw/', offline=True, sample_size=500000):
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

    # 进行随机采样（使用相同的随机种子）
    if sample_size and len(all_click) > sample_size:
        all_click = all_click.sample(n=sample_size, random_state=42)  # 固定随机种子
        print(f"after sampling, all_click.shape: {all_click.shape}")
        print(f"after sampling, all_click length: {len(all_click)}")

    return all_click

def get_item_info_df(data_path):
    item_info_df = pd.read_csv(data_path + 'articles.csv')
    item_info_df = item_info_df.rename(columns={'article_id': 'click_article_id'})
    return item_info_df

def get_hist_and_last_click(all_click):
    all_click = all_click.sort_values(by=['user_id', 'click_timestamp'])
    click_last_df = all_click.groupby('user_id').tail(1)

    def hist_func(user_df):
        if len(user_df) == 1:
            return user_df
        else:
            return user_df[:-1]

    click_hist_df = all_click.groupby('user_id').apply(hist_func).reset_index(drop=True)
    return click_hist_df, click_last_df

def metrics_recall(user_recall_items_dict, trn_last_click_df, topk=50):
    """评估召回命中率"""
    last_click_item_dict = dict(zip(trn_last_click_df['user_id'], trn_last_click_df['click_article_id']))
    user_num = len(user_recall_items_dict)

    results = {}
    for k in range(10, topk + 1, 10):
        hit_num = 0
        for user, item_list in user_recall_items_dict.items():
            if user not in last_click_item_dict:
                continue
            # 获取前k个召回的结果
            tmp_recall_items = [x[0] for x in user_recall_items_dict[user][:k]]
            if last_click_item_dict[user] in set(tmp_recall_items):
                hit_num += 1

        hit_rate = round(hit_num * 1.0 / user_num, 5)
        results[k] = {'hit_num': hit_num, 'hit_rate': hit_rate, 'user_num': user_num}
        print(f' topk: {k:2d} : hit_num: {hit_num:5d} hit_rate: {hit_rate:.5f} user_num: {user_num}')
    
    return results

# ===================== 数据准备函数 =====================
def gen_data_set(data, negsample=0, user_feature_cols=None, item_feature_cols=None):
    """生成训练/测试数据，支持辅助特征"""
    data.sort_values("click_timestamp", inplace=True)
    item_ids = data['click_article_id'].unique()

    train_set = []
    test_set = []
    
    # 获取用户特征字典
    user_features_dict = {}
    if user_feature_cols:
        for user_id, user_df in data.groupby('user_id'):
            user_features_dict[user_id] = {col: user_df[col].iloc[0] for col in user_feature_cols}
    
    # 获取物品特征字典
    item_features_dict = {}
    if item_feature_cols:
        for item_id, item_df in data.groupby('click_article_id'):
            item_features_dict[item_id] = {col: item_df[col].iloc[0] for col in item_feature_cols}
    
    for reviewerID, hist in tqdm(data.groupby('user_id')):
        pos_list = hist['click_article_id'].tolist()
        user_features = user_features_dict.get(reviewerID, {}) if user_feature_cols else {}

        if negsample > 0:
            candidate_set = list(set(item_ids) - set(pos_list))
            neg_list = np.random.choice(candidate_set, size=len(pos_list) * negsample, replace=True)

        if len(pos_list) == 1:
            item_features = item_features_dict.get(pos_list[0], {}) if item_feature_cols else {}
            train_set.append((reviewerID, [pos_list[0]], pos_list[0], 1, len(pos_list), user_features.copy(), item_features.copy()))
            test_set.append((reviewerID, [pos_list[0]], pos_list[0], 1, len(pos_list), user_features.copy(), item_features.copy()))

        for i in range(1, len(pos_list)):
            hist_seq = pos_list[:i]
            item_features = item_features_dict.get(pos_list[i], {}) if item_feature_cols else {}

            if i != len(pos_list) - 1:
                train_set.append((reviewerID, hist_seq[::-1], pos_list[i], 1,
                                  len(hist_seq[::-1]), user_features.copy(), item_features.copy()))
                for negi in range(negsample):
                    neg_item_features = item_features_dict.get(neg_list[i * negsample + negi], {}) if item_feature_cols else {}
                    train_set.append((reviewerID, hist_seq[::-1], neg_list[i * negsample + negi], 0,
                                      len(hist_seq[::-1]), user_features.copy(), neg_item_features.copy()))
            else:
                test_set.append((reviewerID, hist_seq[::-1], pos_list[i], 1, len(hist_seq[::-1]), user_features.copy(), item_features.copy()))

    random.shuffle(train_set)
    random.shuffle(test_set)
    return train_set, test_set

def gen_model_input(train_set, user_profile, seq_max_len, user_feature_cols=None, item_feature_cols=None):
    """生成模型输入，支持辅助特征"""
    train_uid = np.array([line[0] for line in train_set])
    train_seq = [line[1] for line in train_set]
    train_iid = np.array([line[2] for line in train_set])
    train_label = np.array([line[3] for line in train_set])
    train_hist_len = np.array([line[4] for line in train_set])
    
    user_features_data = {}
    item_features_data = {}
    
    if len(train_set) > 0 and len(train_set[0]) > 6:
        if user_feature_cols:
            for col in user_feature_cols:
                user_features_data[col] = np.array([line[5].get(col, 0) for line in train_set])
        if item_feature_cols:
            for col in item_feature_cols:
                item_features_data[col] = np.array([line[6].get(col, 0) for line in train_set])
    else:
        if user_feature_cols:
            for col in user_feature_cols:
                user_features_data[col] = np.zeros(len(train_set), dtype=np.int32)
        if item_feature_cols:
            for col in item_feature_cols:
                item_features_data[col] = np.zeros(len(train_set), dtype=np.int32)

    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)
    train_model_input = {"user_id": train_uid, "click_article_id": train_iid, "hist_article_id": train_seq_pad,
                         "hist_len": train_hist_len}
    train_model_input.update(user_features_data)
    train_model_input.update(item_features_data)

    return train_model_input, train_label

# ===================== YouTubeDNN 召回函数（无辅助特征版本）=====================
def youtubednn_u2i_dict_baseline(data, topk=20):
    """基线版本：只使用 user_id 和 click_article_id"""
    data = data.copy()
    
    SEQ_LEN = 30
    user_profile_ = data[["user_id"]].drop_duplicates('user_id')
    item_profile_ = data[["click_article_id"]].drop_duplicates('click_article_id')

    features = ["click_article_id", "user_id"]
    feature_max_idx = {}
    label_encoders = {}

    for feature in features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature])
        feature_max_idx[feature] = data[feature].max() + 2
        label_encoders[feature] = lbe

    user_profile = data[["user_id"]].drop_duplicates('user_id')
    item_profile = data[["click_article_id"]].drop_duplicates('click_article_id')

    user_index_2_rawid = dict(zip(user_profile['user_id'], user_profile_['user_id']))
    item_index_2_rawid = dict(zip(item_profile['click_article_id'], item_profile_['click_article_id']))

    train_set, test_set = gen_data_set(data, 0)
    train_model_input, train_label = gen_model_input(train_set, user_profile, SEQ_LEN)
    test_model_input, test_label = gen_model_input(test_set, user_profile, SEQ_LEN)

    embedding_dim = 16
    user_feature_columns = [SparseFeat('user_id', feature_max_idx['user_id'], embedding_dim),
                            VarLenSparseFeat(
                                SparseFeat('hist_article_id', feature_max_idx['click_article_id'], embedding_dim,
                                           embedding_name="click_article_id"), SEQ_LEN, 'mean', 'hist_len'), ]
    item_feature_columns = [SparseFeat('click_article_id', feature_max_idx['click_article_id'], embedding_dim)]

    train_counter = Counter(train_model_input['click_article_id'])
    item_count = [train_counter.get(i, 0) for i in range(item_feature_columns[0].vocabulary_size)]
    sampler_config = NegativeSampler('frequency', num_sampled=5, item_name="click_article_id", item_count=item_count)

    import tensorflow as tf
    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()

    model = YoutubeDNN(user_feature_columns, item_feature_columns, user_dnn_hidden_units=(64, 16, embedding_dim),
                       sampler_config=sampler_config)
    model.compile(optimizer="adam", loss=sampledsoftmaxloss)
    history = model.fit(train_model_input, train_label, batch_size=256, epochs=10, verbose=1, validation_split=0.0)

    test_user_model_input = test_model_input
    all_item_model_input = {"click_article_id": item_profile['click_article_id'].values}

    def get_feature_name(input_layer):
        name = input_layer.name
        if ':' in name:
            name = name.split(':')[0]
        return name
    
    def get_feature_data(input_layer, input_dict):
        feature_name = get_feature_name(input_layer)
        if feature_name not in input_dict:
            if 'user_id' in input_dict:
                batch_size = len(input_dict['user_id'])
            elif 'click_article_id' in input_dict:
                batch_size = len(input_dict['click_article_id'])
            else:
                batch_size = 1
            return np.zeros((batch_size, 1), dtype=np.int32)
        
        data = input_dict[feature_name]
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        if feature_name in ['user_id', 'hist_len']:
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            elif len(data.shape) > 2:
                data = data.reshape(data.shape[0], -1)[:, :1]
        elif feature_name == 'hist_article_id':
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            elif len(data.shape) > 2:
                data = data.reshape(data.shape[0], -1)
        elif feature_name == 'click_article_id':
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            elif len(data.shape) > 2:
                data = data.reshape(data.shape[0], -1)[:, :1]
        
        return data
    
    test_user_model_input_list = [get_feature_data(inp, test_user_model_input) for inp in model.user_input]
    all_item_model_input_list = [get_feature_data(inp, all_item_model_input) for inp in model.item_input]
    
    user_embedding_func = K.function(model.user_input, [model.user_embedding])
    item_embedding_func = K.function(model.item_input, [model.item_embedding])
    
    batch_size = 2 ** 12
    n_samples = len(test_user_model_input_list[0])
    
    user_embs_list = []
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_inputs = [inp[i:end_idx] for inp in test_user_model_input_list]
        batch_emb = user_embedding_func(batch_inputs)[0]
        user_embs_list.append(batch_emb)
    user_embs = np.vstack(user_embs_list)
    
    n_items = len(all_item_model_input_list[0])
    item_embs_list = []
    for i in range(0, n_items, batch_size):
        end_idx = min(i + batch_size, n_items)
        batch_inputs = [inp[i:end_idx] for inp in all_item_model_input_list]
        batch_emb = item_embedding_func(batch_inputs)[0]
        item_embs_list.append(batch_emb)
    item_embs = np.vstack(item_embs_list)

    user_embs = user_embs / np.linalg.norm(user_embs, axis=1, keepdims=True)
    item_embs = item_embs / np.linalg.norm(item_embs, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(embedding_dim)
    index.add(item_embs)
    sim, idx = index.search(np.ascontiguousarray(user_embs), topk)

    user_recall_items_dict = collections.defaultdict(dict)
    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(test_user_model_input['user_id'], sim, idx)):
        target_raw_id = user_index_2_rawid[target_idx]
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
            if rele_idx in item_index_2_rawid:
                rele_raw_id = item_index_2_rawid[rele_idx]
                user_recall_items_dict[target_raw_id][rele_raw_id] = user_recall_items_dict.get(target_raw_id, {}) \
                                                                         .get(rele_raw_id, 0) + sim_value

    user_recall_items_dict = {k: sorted(v.items(), key=lambda x: x[1], reverse=True) for k, v in
                              user_recall_items_dict.items()}

    return user_recall_items_dict

# ===================== YouTubeDNN 召回函数（有辅助特征版本）=====================
def youtubednn_u2i_dict_with_features(data, item_info_df=None, all_click_df=None, topk=20):
    """带辅助特征版本：使用 user_id, click_article_id + 用户特征 + 物品特征"""
    data = data.copy()
    
    SEQ_LEN = 30
    
    # 添加用户特征
    # 改进：只使用分布相对均衡且稳定的特征，避免引入噪声
    user_aux_features = []
    if all_click_df is not None:
        # 优先使用：分布相对均衡、稳定性好的特征
        # 1. click_os: 6个唯一值，分布相对均衡
        # 2. click_deviceGroup: 3个唯一值，分布较好
        # 3. click_region: 27个唯一值，但分布还可以
        # 避免使用：click_environment (99%是值4), click_country (96.8%是值1), click_referrer_type (7.18%不一致)
        user_feature_cols = ['click_os', 'click_deviceGroup', 'click_region']  # 只使用相对稳定的特征
        
        available_users = set(data['user_id'].unique())
        user_features_df = all_click_df[all_click_df['user_id'].isin(available_users)]
        
        if len(user_features_df) > 0:
            for col in user_feature_cols:
                if col in user_features_df.columns:
                    # 如果 data 中已经有这个列（比如从原始数据来的），直接使用
                    if col in data.columns:
                        user_aux_features.append(col)
                    else:
                        # 否则从 all_click_df 中获取并 merge
                        user_mode = user_features_df.groupby('user_id')[col].agg(lambda x: x.value_counts().index[0] if len(x) > 0 else x.iloc[0]).reset_index()
                        user_mode.columns = ['user_id', col]
                        data = data.merge(user_mode, on='user_id', how='left')
                        data[col] = data[col].fillna(0)
                        user_aux_features.append(col)
    
    # 添加物品特征
    # 修改：现在使用本地版本的 YouTubeDNN，支持多个物品特征！
    item_aux_features = []
    if item_info_df is not None:
        item_feature_cols = ['category_id']  # 可以添加更多物品特征
        for col in item_feature_cols:
            if col in item_info_df.columns:
                data = data.merge(item_info_df[['click_article_id', col]], 
                                on='click_article_id', how='left')
                data[col] = data[col].fillna(-1).astype(int)
                item_aux_features.append(col)
    
    user_profile_ = data[["user_id"]].drop_duplicates('user_id')
    item_profile_ = data[["click_article_id"]].drop_duplicates('click_article_id')

    features = ["click_article_id", "user_id"] + user_aux_features + item_aux_features
    feature_max_idx = {}
    label_encoders = {}

    for feature in features:
        lbe = LabelEncoder()
        if feature in item_aux_features and feature == 'category_id':
            unique_values = sorted(data[feature].unique())
            if -1 in unique_values:
                unique_values.remove(-1)
                unique_values = [-1] + unique_values
            lbe.fit(unique_values)
            data[feature] = data[feature].apply(lambda x: x if x in lbe.classes_ else -1)
        else:
            unique_values = data[feature].unique()
            lbe.fit(unique_values)
        
        data[feature] = lbe.transform(data[feature])
        if feature == 'category_id':
            feature_max_idx[feature] = data[feature].max() + 1
        else:
            feature_max_idx[feature] = data[feature].max() + 2
        label_encoders[feature] = lbe

    user_profile = data[["user_id"]].drop_duplicates('user_id')
    item_profile = data[["click_article_id"]].drop_duplicates('click_article_id')

    user_index_2_rawid = dict(zip(user_profile['user_id'], user_profile_['user_id']))
    item_index_2_rawid = dict(zip(item_profile['click_article_id'], item_profile_['click_article_id']))

    train_set, test_set = gen_data_set(data, 0, 
                                       user_feature_cols=user_aux_features if user_aux_features else None,
                                       item_feature_cols=item_aux_features if item_aux_features else None)
    train_model_input, train_label = gen_model_input(train_set, user_profile, SEQ_LEN,
                                                     user_feature_cols=user_aux_features if user_aux_features else None,
                                                     item_feature_cols=item_aux_features if item_aux_features else None)
    test_model_input, test_label = gen_model_input(test_set, user_profile, SEQ_LEN,
                                                   user_feature_cols=user_aux_features if user_aux_features else None,
                                                   item_feature_cols=item_aux_features if item_aux_features else None)

    embedding_dim = 32
    user_feature_columns = [SparseFeat('user_id', feature_max_idx['user_id'], embedding_dim)]
    # 改进：降低辅助特征的embedding维度，避免模型参数过多
    for feat in user_aux_features:
        # 使用较小的embedding维度（8维）来减少参数量，让模型更专注于主要特征
        feat_emb_dim = min(embedding_dim, 8)  # 辅助特征使用较小的embedding维度
        user_feature_columns.append(SparseFeat(feat, feature_max_idx[feat], feat_emb_dim))
    user_feature_columns.append(
        VarLenSparseFeat(
            SparseFeat('hist_article_id', feature_max_idx['click_article_id'], embedding_dim,
                       embedding_name="click_article_id"), SEQ_LEN, 'mean', 'hist_len')
    )
    
    item_feature_columns = [SparseFeat('click_article_id', feature_max_idx['click_article_id'], embedding_dim)]
    for feat in item_aux_features:
        item_feature_columns.append(SparseFeat(feat, feature_max_idx[feat], embedding_dim))

    train_counter = Counter(train_model_input['click_article_id'])
    item_count = [train_counter.get(i, 0) for i in range(item_feature_columns[0].vocabulary_size)]
    sampler_config = NegativeSampler('frequency', num_sampled=5, item_name="click_article_id", item_count=item_count)

    import tensorflow as tf
    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()

    # 改进：增加模型容量以支持更多特征
    # 当有辅助特征时，增加DNN的隐藏层维度
    if len(user_aux_features) > 0:
        # 有辅助特征时，增加模型容量
        user_dnn_hidden_units = (128, 64, 32, embedding_dim)  # 增加隐藏层维度
        print(f"使用增强模型容量: {user_dnn_hidden_units} (因为有 {len(user_aux_features)} 个辅助特征)")
    else:
        user_dnn_hidden_units = (64, 16, embedding_dim)  # 基线版本保持原样
    
    model = YoutubeDNN(user_feature_columns, item_feature_columns, user_dnn_hidden_units=user_dnn_hidden_units,
                       sampler_config=sampler_config)
    model.compile(optimizer="adam", loss=sampledsoftmaxloss)
    history = model.fit(train_model_input, train_label, batch_size=256, epochs=20, verbose=1, validation_split=0.0)

    test_user_model_input = test_model_input
    all_item_model_input = {"click_article_id": item_profile['click_article_id'].values}
    for feat in item_aux_features:
        item_feat_values = []
        for item_id in item_profile['click_article_id']:
            item_data = data[data['click_article_id'] == item_id]
            if len(item_data) > 0:
                item_feat_values.append(item_data[feat].iloc[0])
            else:
                item_feat_values.append(0 if feat == 'category_id' else feature_max_idx[feat] - 1)
        all_item_model_input[feat] = np.array(item_feat_values)

    def get_feature_name(input_layer):
        name = input_layer.name
        if ':' in name:
            name = name.split(':')[0]
        return name
    
    def get_feature_data(input_layer, input_dict):
        feature_name = get_feature_name(input_layer)
        if feature_name not in input_dict:
            if 'user_id' in input_dict:
                batch_size = len(input_dict['user_id'])
            elif 'click_article_id' in input_dict:
                batch_size = len(input_dict['click_article_id'])
            else:
                batch_size = 1
            return np.zeros((batch_size, 1), dtype=np.int32)
        
        data = input_dict[feature_name]
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        if feature_name in ['user_id', 'hist_len']:
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            elif len(data.shape) > 2:
                data = data.reshape(data.shape[0], -1)[:, :1]
        elif feature_name == 'hist_article_id':
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            elif len(data.shape) > 2:
                data = data.reshape(data.shape[0], -1)
        # click_article_id 和辅助特征（用户特征、物品特征）
        # 注意：user_aux_features 和 item_aux_features 在外部作用域中定义
        elif feature_name == 'click_article_id' or feature_name in user_aux_features or feature_name in item_aux_features:
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            elif len(data.shape) > 2:
                data = data.reshape(data.shape[0], -1)[:, :1]
        
        return data
    
    test_user_model_input_list = [get_feature_data(inp, test_user_model_input) for inp in model.user_input]
    all_item_model_input_list = [get_feature_data(inp, all_item_model_input) for inp in model.item_input]
    
    user_embedding_func = K.function(model.user_input, [model.user_embedding])
    item_embedding_func = K.function(model.item_input, [model.item_embedding])
    
    batch_size = 2 ** 12
    n_samples = len(test_user_model_input_list[0])
    
    user_embs_list = []
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_inputs = [inp[i:end_idx] for inp in test_user_model_input_list]
        batch_emb = user_embedding_func(batch_inputs)[0]
        user_embs_list.append(batch_emb)
    user_embs = np.vstack(user_embs_list)
    
    n_items = len(all_item_model_input_list[0])
    item_embs_list = []
    for i in range(0, n_items, batch_size):
        end_idx = min(i + batch_size, n_items)
        batch_inputs = [inp[i:end_idx] for inp in all_item_model_input_list]
        batch_emb = item_embedding_func(batch_inputs)[0]
        item_embs_list.append(batch_emb)
    item_embs = np.vstack(item_embs_list)

    user_embs = user_embs / np.linalg.norm(user_embs, axis=1, keepdims=True)
    item_embs = item_embs / np.linalg.norm(item_embs, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(embedding_dim)
    index.add(item_embs)
    sim, idx = index.search(np.ascontiguousarray(user_embs), topk)

    user_recall_items_dict = collections.defaultdict(dict)
    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(test_user_model_input['user_id'], sim, idx)):
        target_raw_id = user_index_2_rawid[target_idx]
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
            if rele_idx in item_index_2_rawid:
                rele_raw_id = item_index_2_rawid[rele_idx]
                user_recall_items_dict[target_raw_id][rele_raw_id] = user_recall_items_dict.get(target_raw_id, {}) \
                                                                         .get(rele_raw_id, 0) + sim_value

    user_recall_items_dict = {k: sorted(v.items(), key=lambda x: x[1], reverse=True) for k, v in
                              user_recall_items_dict.items()}

    return user_recall_items_dict

# ===================== 主测试流程 =====================
if __name__ == '__main__':
    print("=" * 80)
    print("YouTubeDNN 召回效果对比测试")
    print("=" * 80)
    
    # 使用相同的采样设置
    sample_size = 10000  # 使用较小的采样以加快测试
    print(f"\n使用采样大小: {sample_size}")
    print(f"随机种子: 42 (固定以确保结果可复现)\n")
    
    # 加载数据
    print("加载数据...")
    all_click_df = get_all_click_df(offline=False, sample_size=sample_size)
    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    all_click_df['click_timestamp'] = all_click_df[['click_timestamp']].apply(max_min_scaler)
    
    item_info_df = get_item_info_df(data_path)
    
    # 合并训练集和测试集以获取完整的用户特征
    try:
        tst_click = pd.read_csv(data_path + 'testA_click_log.csv')
        all_click_with_features = pd.concat([all_click_df, tst_click], ignore_index=True)
    except:
        all_click_with_features = all_click_df
    
    # 划分训练和测试数据（从 all_click_df 划分，因为训练时只需要训练数据）
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
    
    # 但为了获取用户特征，需要从 all_click_with_features 中获取
    # 因为 all_click_df 可能没有包含用户特征列（如果是从原始数据加载的）
    # 检查 all_click_df 中是否有用户特征列
    user_feature_cols = ['click_environment', 'click_deviceGroup', 'click_os', 
                        'click_country', 'click_region', 'click_referrer_type']
    has_user_features = any(col in all_click_df.columns for col in user_feature_cols)
    
    if not has_user_features:
        # 如果 all_click_df 没有用户特征，需要从 all_click_with_features 中获取
        print("注意: all_click_df 中没有用户特征列，将从 all_click_with_features 中获取")
        # 重新加载数据，这次包含用户特征
        all_click_df = get_all_click_df(offline=False, sample_size=sample_size)
        all_click_df['click_timestamp'] = all_click_df[['click_timestamp']].apply(max_min_scaler)
        # 重新合并
        try:
            tst_click = pd.read_csv(data_path + 'testA_click_log.csv')
            all_click_with_features = pd.concat([all_click_df, tst_click], ignore_index=True)
        except:
            all_click_with_features = all_click_df
        # 重新划分
        trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
    
    print(f"\n数据统计:")
    print(f"  训练历史数据: {trn_hist_click_df.shape}")
    print(f"  测试数据（最后点击）: {trn_last_click_df.shape}")
    print(f"  用户数量: {trn_hist_click_df['user_id'].nunique()}")
    
    # ===================== 测试1: 基线版本（无辅助特征）=====================
    print("\n" + "=" * 80)
    print("测试1: 基线版本（只使用 user_id 和 click_article_id）")
    print("=" * 80)
    
    baseline_recall = youtubednn_u2i_dict_baseline(trn_hist_click_df, topk=20)
    baseline_results = metrics_recall(baseline_recall, trn_last_click_df, topk=50)
    
    # ===================== 测试2: 带辅助特征版本 =====================
    print("\n" + "=" * 80)
    print("测试2: 带辅助特征版本（user_id + 用户特征 + click_article_id + 物品特征）")
    print("改进：")
    print("  1. 只使用分布相对均衡的特征（click_os, click_deviceGroup, click_region）")
    print("  2. 避免使用分布极度不均衡的特征（click_environment, click_country）")
    print("  3. 避免使用噪声特征（click_referrer_type）")
    print("  4. 降低辅助特征的embedding维度（8维）")
    print("  5. 增加模型容量（128-64-32-32）以支持更多特征")
    print("  6. 使用本地修改版 YouTubeDNN，支持多个物品特征（category_id）")
    print("=" * 80)
    
    with_features_recall = youtubednn_u2i_dict_with_features(trn_hist_click_df, 
                                                              item_info_df=item_info_df, 
                                                              all_click_df=all_click_with_features,
                                                              topk=20)
    with_features_results = metrics_recall(with_features_recall, trn_last_click_df, topk=50)
    
    # ===================== 对比结果 =====================
    print("\n" + "=" * 80)
    print("对比结果")
    print("=" * 80)
    print(f"{'TopK':<8} {'基线命中率':<15} {'带特征命中率':<15} {'提升率':<15} {'提升点数':<15}")
    print("-" * 80)
    
    for k in sorted(set(list(baseline_results.keys()) + list(with_features_results.keys()))):
        baseline_hit_rate = baseline_results.get(k, {}).get('hit_rate', 0)
        with_features_hit_rate = with_features_results.get(k, {}).get('hit_rate', 0)
        
        if baseline_hit_rate > 0:
            improvement_rate = ((with_features_hit_rate - baseline_hit_rate) / baseline_hit_rate) * 100
            improvement_points = with_features_hit_rate - baseline_hit_rate
        else:
            improvement_rate = float('inf') if with_features_hit_rate > 0 else 0
            improvement_points = with_features_hit_rate - baseline_hit_rate
        
        print(f"{k:<8} {baseline_hit_rate:>13.5f}    {with_features_hit_rate:>13.5f}    {improvement_rate:>13.2f}%    {improvement_points:>13.5f}")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


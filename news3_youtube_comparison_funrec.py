"""
使用 funRec 的 YouTubeDNN 模型进行召回效果对比测试
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
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 使用 funRec 的 YouTubeDNN 模型
import sys
funrec_path = os.path.join(os.path.dirname(__file__), 'funRec', 'src')
sys.path.insert(0, funrec_path)

# 直接导入需要的模块，避免依赖问题
from funrec.models.youtubednn import build_youtubednn_model
from funrec.features.feature_column import FeatureColumn
from collections import Counter

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
    return all_click

def get_item_info_df(data_path='../data_raw/'):
    item_info_df = pd.read_csv(data_path + 'articles.csv')
    # 为了方便与训练集中的click_article_id拼接，需要把article_id修改成click_article_id
    item_info_df = item_info_df.rename(columns={'article_id': 'click_article_id'})
    return item_info_df

def get_hist_and_last_click(all_click):
    """获取用户历史点击和最后一次点击"""
    all_click = all_click.sort_values(['user_id', 'click_timestamp'])
    click_list = []
    for user_id, hist_click in tqdm(all_click.groupby('user_id')):
        click_list.append({
            'user_id': user_id,
            'click_article_id': hist_click['click_article_id'].tolist(),
            'click_timestamp': hist_click['click_timestamp'].tolist()
        })
    
    hist_click_list = []
    last_click_list = []
    for click in click_list:
        if len(click['click_article_id']) > 1:
            hist_click_list.append({
                'user_id': click['user_id'],
                'click_article_id': click['click_article_id'][:-1],
                'click_timestamp': click['click_timestamp'][:-1]
            })
            last_click_list.append({
                'user_id': click['user_id'],
                'click_article_id': click['click_article_id'][-1],
                'click_timestamp': click['click_timestamp'][-1]
            })
    
    hist_click_df = pd.DataFrame(hist_click_list)
    last_click_df = pd.DataFrame(last_click_list)
    
    # 展开 hist_click_df
    hist_click_expanded = []
    for _, row in hist_click_df.iterrows():
        for article_id, timestamp in zip(row['click_article_id'], row['click_timestamp']):
            hist_click_expanded.append({
                'user_id': row['user_id'],
                'click_article_id': article_id,
                'click_timestamp': timestamp
            })
    hist_click_df = pd.DataFrame(hist_click_expanded)
    
    return hist_click_df, last_click_df

def metrics_recall(user_recall_items_dict, trn_last_click_df, topk=5):
    """评估召回效果"""
    last_click_item_dict = dict(zip(trn_last_click_df['user_id'], trn_last_click_df['click_article_id']))
    
    # 只计算同时在召回字典和最后一次点击数据中的用户数（实际参与评估的用户数）
    valid_users = set(user_recall_items_dict.keys()) & set(last_click_item_dict.keys())
    user_num = len(valid_users)
    
    if user_num == 0:
        print("警告: 召回字典和最后一次点击数据中没有共同的用户！")
        print(f"召回字典用户数: {len(user_recall_items_dict)}")
        print(f"最后一次点击用户数: {len(last_click_item_dict)}")
        print(f"召回字典前5个用户: {list(user_recall_items_dict.keys())[:5]}")
        print(f"最后一次点击前5个用户: {list(last_click_item_dict.keys())[:5]}")
        return

    # 调试：检查前几个用户的数据格式和类型
    if user_num > 0:
        sample_user = list(valid_users)[0]
        print(f"\n调试信息 - 示例用户 {sample_user}:")
        print(f"  召回结果类型: {type(user_recall_items_dict[sample_user])}")
        print(f"  召回结果长度: {len(user_recall_items_dict[sample_user])}")
        if len(user_recall_items_dict[sample_user]) > 0:
            print(f"  召回结果第一个元素: {user_recall_items_dict[sample_user][0]}")
            print(f"  召回结果第一个元素类型: {type(user_recall_items_dict[sample_user][0])}")
        print(f"  最后一次点击的 article_id: {last_click_item_dict[sample_user]}")
        print(f"  最后一次点击的 article_id 类型: {type(last_click_item_dict[sample_user])}")
        
        # 检查前5个召回结果
        item_list = user_recall_items_dict[sample_user]
        if len(item_list) > 0:
            if isinstance(item_list[0], (list, tuple)):
                recall_items = [x[0] for x in item_list[:5]]
            else:
                recall_items = item_list[:5]
            print(f"  前5个召回结果: {recall_items}")
            print(f"  召回结果类型: {[type(x) for x in recall_items]}")
            print(f"  是否匹配: {last_click_item_dict[sample_user] in set(recall_items)}")
        print()
    
    for k in range(10, topk + 1, 10):
        hit_num = 0
        for user in valid_users:
            item_list = user_recall_items_dict[user]
            # 处理不同的数据格式：可能是 [(item, score), ...] 或 [item, ...]
            if len(item_list) > 0:
                if isinstance(item_list[0], (list, tuple)):
                    tmp_recall_items = [x[0] for x in item_list[:k]]
                else:
                    tmp_recall_items = item_list[:k]
            else:
                tmp_recall_items = []
            
            # 统一转换为 int 类型进行比较（避免类型不匹配）
            last_click_item = int(last_click_item_dict[user])
            tmp_recall_items = [int(x) for x in tmp_recall_items]
            
            if last_click_item in set(tmp_recall_items):
                hit_num += 1

        hit_rate = round(hit_num * 1.0 / user_num, 5) if user_num > 0 else 0.0
        print(f"k={k}: hit_num={hit_num}, hit_rate={hit_rate}, user_num={user_num}")

# ===================== 使用 funRec 的 YouTubeDNN =====================
def youtubednn_u2i_dict_funrec(data, item_info_df=None, all_click_df=None, topk=20, use_aux_features=True):
    """使用 funRec 的 YouTubeDNN 模型
    
    Args:
        data: 训练数据
        item_info_df: 物品信息数据
        all_click_df: 所有点击数据
        topk: 召回数量
        use_aux_features: 是否使用辅助特征（用户特征和物品特征）
    """
    data = data.copy()
    SEQ_LEN = 30
    
    # 添加用户特征
    user_aux_features = []
    initial_user_count = data['user_id'].nunique()
    initial_data_count = len(data)
    
    if use_aux_features and all_click_df is not None:
        user_feature_cols = ['click_os', 'click_deviceGroup', 'click_region']
        available_users = set(data['user_id'].unique())
        user_features_df = all_click_df[all_click_df['user_id'].isin(available_users)]
        
        if len(user_features_df) > 0:
            for col in user_feature_cols:
                if col in user_features_df.columns:
                    if col not in data.columns:
                        user_mode = user_features_df.groupby('user_id')[col].agg(
                            lambda x: x.value_counts().index[0] if len(x) > 0 else x.iloc[0]
                        ).reset_index()
                        user_mode.columns = ['user_id', col]
                        data = data.merge(user_mode, on='user_id', how='left')
                        # 检查 merge 后是否有数据丢失
                        if len(data) != initial_data_count:
                            print(f"⚠️ 警告: merge {col} 后数据量从 {initial_data_count} 变为 {len(data)}")
                        data[col] = data[col].fillna(0)
                    user_aux_features.append(col)
    
    # 检查用户数量是否有变化
    final_user_count = data['user_id'].nunique()
    if initial_user_count != final_user_count:
        print(f"⚠️ 警告: 添加特征后用户数从 {initial_user_count} 变为 {final_user_count}")
    
    if use_aux_features:
        print(f"使用辅助特征: 用户特征={user_aux_features}, 初始用户数={initial_user_count}, 最终用户数={final_user_count}")
    
    # 添加物品特征
    item_aux_features = []
    if use_aux_features and item_info_df is not None:
        item_feature_cols = ['category_id']
        for col in item_feature_cols:
            if col in item_info_df.columns:
                data = data.merge(item_info_df[['click_article_id', col]], 
                                on='click_article_id', how='left')
                data[col] = data[col].fillna(-1).astype(int)
                item_aux_features.append(col)
    
    # 编码特征
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
            # 对于 click_article_id 和 user_id，如果提供了 all_click_df，合并所有值
            if feature in ['click_article_id', 'user_id'] and all_click_df is not None and feature in all_click_df.columns:
                unique_values = pd.concat([data[[feature]], all_click_df[[feature]]])[feature].dropna().unique()
            else:
                unique_values = data[feature].unique()
            lbe.fit(unique_values)
        
        data[feature] = lbe.transform(data[feature])
        feature_max_idx[feature] = data[feature].max() + 1
        label_encoders[feature] = lbe
    
    # 准备训练数据（使用滑动窗口）
    train_data = []
    for user_id, user_data in data.groupby('user_id'):
        # 按时间排序用户的所有点击
        user_data_sorted = user_data.sort_values('click_timestamp').reset_index(drop=True)
        
        # 滑动窗口：对于第 i 个点击，历史序列是前 i-1 个点击
        hist = []  # 初始历史序列为空
        for idx, row in user_data_sorted.iterrows():
            # 只有当历史序列不为空时，才创建训练样本
            # 对于第一个点击，历史序列为空，跳过
            if len(hist) > 0:
                train_data.append({
                    'user_id': row['user_id'],
                    'click_article_id': row['click_article_id'],  # 当前点击的物品
                    'hist_article_id': hist.copy(),  # 历史序列（当前点击之前的所有点击）
                    **{feat: row[feat] for feat in user_aux_features + item_aux_features}
                })
            # 将当前点击添加到历史序列中（用于下一次迭代）
            hist.append(row['click_article_id'])
    
    train_df = pd.DataFrame(train_data)
    
    # 构建 funRec 的 FeatureColumn
    embedding_dim = 32
    user_embedding_dim = 64
    feature_columns = []
    
    # 用户特征
    feature_columns.append(FeatureColumn(
        name='user_id',
        vocab_size=feature_max_idx['user_id'],
        emb_dim=user_embedding_dim,
        group=['user_dnn']
    ))
    
    for feat in user_aux_features:
        feature_columns.append(FeatureColumn(
            name=feat,
            vocab_size=feature_max_idx[feat],
            emb_dim=8,  # 辅助特征使用较小的维度
            group=['user_dnn']
        ))
    
    # 历史序列特征
    feature_columns.append(FeatureColumn(
        name='hist_article_id',
        emb_name='click_article_id',  # 共享 embedding
        vocab_size=feature_max_idx['click_article_id'],
        emb_dim=user_embedding_dim,
        max_len=SEQ_LEN,
        type='varlen_sparse',
        combiner='mean',
        group=['raw_hist_seq']
    ))
    
    # 物品特征（目标物品）
    feature_columns.append(FeatureColumn(
        name='click_article_id',
        vocab_size=feature_max_idx['click_article_id'],
        emb_dim=user_embedding_dim,
        group=['target_item']
    ))
    
    # 物品辅助特征（如果有多个，需要特殊处理）
    for feat in item_aux_features:
        feature_columns.append(FeatureColumn(
            name=feat,
            vocab_size=feature_max_idx[feat],
            emb_dim=embedding_dim,
            group=['target_item']
        ))
    
    # 准备模型输入数据
    def prepare_input_data(df):
        inputs = {
            'user_id': df['user_id'].values.reshape(-1, 1),
            'click_article_id': df['click_article_id'].values.reshape(-1, 1),
            'hist_article_id': pad_sequences(
                df['hist_article_id'].apply(lambda x: x if isinstance(x, list) else []).tolist(),
                maxlen=SEQ_LEN,
                padding='post',
                truncating='post',
                value=0
            )
        }
        
        for feat in user_aux_features:
            inputs[feat] = df[feat].values.reshape(-1, 1)
        
        for feat in item_aux_features:
            inputs[feat] = df[feat].values.reshape(-1, 1)
        
        return inputs
    
    train_inputs = prepare_input_data(train_df)
    train_labels = train_df['click_article_id'].values.reshape(-1, 1)
    
    # 构建模型
    model_config = {
        'emb_dim': user_embedding_dim,  # 必须与物品 embedding 维度一致
        'neg_sample': 20,
        'dnn_units': [128, 64, 32] if len(user_aux_features) > 0 else [64, 16],
        'label_name': 'click_article_id'
    }
    
    model, user_model, item_model = build_youtubednn_model(feature_columns, model_config)
    
    # 编译模型
    def sampledsoftmaxloss(y_true, y_pred):
        return tf.reduce_mean(y_pred)
    
    model.compile(optimizer='adam', loss=sampledsoftmaxloss)
    
    # 训练模型
    print("开始训练模型...")
    model.fit(train_inputs, train_labels, batch_size=256, epochs=10, verbose=1)
    
    # 获取所有用户和物品的 embedding
    print("提取 embedding...")
    user_profile = data[["user_id"]].drop_duplicates('user_id')
    
    # ⚠️ 重要：item_profile 应该包含所有可能的物品，而不仅仅是训练数据中的物品
    # 这样才能召回所有物品，包括最后一次点击的物品
    # 优先使用 all_click_df，因为它包含实际出现过的物品，可以正确编码
    if all_click_df is not None:
        # 使用 all_click_df 中的物品（包含训练和测试数据中的所有物品）
        item_profile_raw = all_click_df[["click_article_id"]].drop_duplicates('click_article_id').copy()
        print(f"使用 all_click_df 构建 item_profile，物品数: {len(item_profile_raw)}")
    elif item_info_df is not None and 'click_article_id' in item_info_df.columns:
        # 如果 all_click_df 不可用，使用 item_info_df 获取所有物品
        item_profile_raw = item_info_df[["click_article_id"]].drop_duplicates('click_article_id').copy()
        print(f"使用 item_info_df 构建 item_profile，物品数: {len(item_profile_raw)}")
    else:
        # 只使用训练数据中的物品（最不理想的情况）
        item_profile_raw = data[["click_article_id"]].drop_duplicates('click_article_id').copy()
        print(f"使用训练数据构建 item_profile，物品数: {len(item_profile_raw)}")
    
    # ⚠️ 关键修复：对 item_profile 中的物品 ID 进行编码
    # 保存原始 ID 到编码 ID 的映射
    item_encoder = label_encoders['click_article_id']
    # 对于不在训练数据中的物品，使用 0 作为编码（或者跳过）
    item_profile = item_profile_raw.copy()
    item_profile['click_article_id_encoded'] = item_profile['click_article_id'].apply(
        lambda x: item_encoder.transform([x])[0] if x in item_encoder.classes_ else -1
    )
    # 只保留能够编码的物品（在训练数据中出现过的，或者能够映射的）
    # 对于新物品，我们需要特殊处理，这里先过滤掉无法编码的
    item_profile = item_profile[item_profile['click_article_id_encoded'] >= 0].copy()
    print(f"编码后有效物品数: {len(item_profile)}")
    
    # 创建原始 ID 到编码 ID 的映射（用于返回结果时还原）
    rawid_2_encodedid = dict(zip(item_profile['click_article_id'], item_profile['click_article_id_encoded']))
    encodedid_2_rawid = dict(zip(item_profile['click_article_id_encoded'], item_profile['click_article_id']))
    
    # 构建用户历史序列字典（编码后的物品ID）
    user_hist_dict = {}
    for user_id, user_data in data.groupby('user_id'):
        # 按时间排序用户的所有点击
        user_data_sorted = user_data.sort_values('click_timestamp')
        # 获取历史序列（所有点击，编码后的物品ID）
        hist = user_data_sorted['click_article_id'].tolist()
        user_hist_dict[user_id] = hist
    
    # 准备用户输入
    user_inputs = {
        'user_id': user_profile['user_id'].values.reshape(-1, 1),
        'hist_article_id': np.zeros((len(user_profile), SEQ_LEN), dtype=np.int32)
    }
    for feat in user_aux_features:
        user_feat_values = []
        for user_id in user_profile['user_id']:
            user_data = data[data['user_id'] == user_id]
            if len(user_data) > 0:
                user_feat_values.append(user_data[feat].iloc[0])
            else:
                user_feat_values.append(0)
        user_inputs[feat] = np.array(user_feat_values).reshape(-1, 1)
    
    # 更新历史序列
    for idx, user_id in enumerate(user_profile['user_id']):
        hist = user_hist_dict.get(user_id, [])
        if len(hist) > 0:
            hist_padded = pad_sequences([hist], maxlen=SEQ_LEN, padding='post', truncating='post', value=0)[0]
            user_inputs['hist_article_id'][idx] = hist_padded
    
    # 准备物品输入（使用编码后的 ID）
    item_inputs = {
        'click_article_id': item_profile['click_article_id_encoded'].values.reshape(-1, 1)
    }
    for feat in item_aux_features:
        item_feat_values = []
        for item_id_raw in item_profile['click_article_id']:
            item_id_encoded = rawid_2_encodedid[item_id_raw]
            # 优先从训练数据中获取特征
            item_data = data[data['click_article_id'] == item_id_encoded]
            if len(item_data) > 0:
                item_feat_values.append(item_data[feat].iloc[0])
            elif item_info_df is not None and feat in item_info_df.columns:
                # 从 item_info_df 中获取特征，然后编码
                item_info = item_info_df[item_info_df['click_article_id'] == item_id_raw]
                if len(item_info) > 0:
                    feat_value = item_info[feat].iloc[0]
                    # 如果特征也需要编码，进行编码
                    if feat in label_encoders:
                        if feat_value in label_encoders[feat].classes_:
                            feat_value = label_encoders[feat].transform([feat_value])[0]
                        else:
                            feat_value = 0
                    item_feat_values.append(feat_value)
                else:
                    item_feat_values.append(0)
            else:
                item_feat_values.append(0)
        item_inputs[feat] = np.array(item_feat_values).reshape(-1, 1)
    
    # 获取 embedding
    user_embs = user_model.predict(user_inputs, batch_size=2048, verbose=0)
    item_embs = item_model.predict(item_inputs, batch_size=2048, verbose=0)
    
    # L2 归一化
    user_embs = user_embs / (np.linalg.norm(user_embs, axis=1, keepdims=True) + 1e-8)
    item_embs = item_embs / (np.linalg.norm(item_embs, axis=1, keepdims=True) + 1e-8)
    
    # 构建索引
    # user_id 需要从编码后的 ID 还原为原始 ID
    user_encoder = label_encoders['user_id']
    user_index_2_rawid = {
        idx: user_encoder.inverse_transform([encoded_id])[0] 
        for idx, encoded_id in enumerate(user_profile['user_id'])
    }
    # item_id 已经是原始 ID（从 item_profile['click_article_id'] 获取）
    item_index_2_rawid = dict(zip(range(len(item_profile)), item_profile['click_article_id']))
    
    # 使用 faiss 进行相似度搜索
    index = faiss.IndexFlatIP(user_embedding_dim)  # 必须与 embedding 维度一致
    index.add(item_embs.astype('float32'))
    sim, idx = index.search(user_embs.astype('float32'), topk + 1)  # 多取一个，因为可能第一个是自己
    
    # 构建召回结果
    user_recall_items_dict = collections.defaultdict(dict)
    for user_idx, (sim_list, idx_list) in enumerate(zip(sim, idx)):
        user_id = user_index_2_rawid[user_idx]
        for sim_value, item_idx in zip(sim_list, idx_list):
            if item_idx < len(item_index_2_rawid) and item_idx in item_index_2_rawid:
                item_id = item_index_2_rawid[item_idx]
                user_recall_items_dict[user_id][item_id] = sim_value
    
    user_recall_items_dict = {
        k: sorted(v.items(), key=lambda x: x[1], reverse=True) 
        for k, v in user_recall_items_dict.items()
    }
    
    return user_recall_items_dict

# ===================== 主程序 =====================
if __name__ == '__main__':
    print("=" * 80)
    print("使用 funRec 的 YouTubeDNN 模型进行召回测试")
    print("=" * 80)
    
    # 加载数据
    all_click_df = get_all_click_df(data_path, offline=False, sample_size=20000)
    item_info_df = get_item_info_df(data_path)
    
    # 采样数据（确保一致性）
    # np.random.seed(42)
    sample_users = np.random.choice(all_click_df['user_id'].unique(), size=50000, replace=False)
    all_click_df = all_click_df[all_click_df['user_id'].isin(sample_users)]
    
    # 获取历史点击和最后一次点击
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
    
    print(f"训练历史点击数据: {len(trn_hist_click_df)}")
    print(f"最后一次点击数据: {len(trn_last_click_df)}")
    
    # ========== 实验1: 只使用 ID 特征 ==========
    print("\n" + "=" * 80)
    print("实验1: 只使用 ID 特征（user_id + click_article_id）")
    print("特征: user_id, click_article_id, hist_article_id")
    print("=" * 80)
    recall_dict_id_only = youtubednn_u2i_dict_funrec(
        trn_hist_click_df,
        item_info_df=item_info_df,
        all_click_df=all_click_df,
        topk=50,
        use_aux_features=False
    )
    
    # 评估召回效果
    print("\n【只使用 ID 特征】召回效果评估:")
    metrics_recall(recall_dict_id_only, trn_last_click_df, topk=50)
    
    # ========== 实验2: 使用辅助特征 ==========
    print("\n" + "=" * 80)
    print("实验2: 使用辅助特征（ID + 用户特征 + 物品特征）")
    print("特征: user_id, click_article_id, hist_article_id")
    print("     + 用户特征: click_os, click_deviceGroup, click_region")
    print("     + 物品特征: category_id")
    print("=" * 80)
    recall_dict_with_aux = youtubednn_u2i_dict_funrec(
        trn_hist_click_df,
        item_info_df=item_info_df,
        all_click_df=all_click_df,
        topk=50,
        use_aux_features=True
    )
    
    # 评估召回效果
    print("\n【使用辅助特征】召回效果评估:")
    metrics_recall(recall_dict_with_aux, trn_last_click_df, topk=50)
    
    # ========== 对比结果 ==========
    print("\n" + "=" * 80)
    print("对比结果汇总")
    print("=" * 80)
    
    def calculate_recall_at_k(recall_dict, last_click_df, k):
        """计算指定 k 的召回率"""
        last_click_item_dict = dict(zip(last_click_df['user_id'], last_click_df['click_article_id']))
        valid_users = set(recall_dict.keys()) & set(last_click_item_dict.keys())
        if len(valid_users) == 0:
            return 0.0, 0
        
        hit_num = 0
        for user in valid_users:
            item_list = recall_dict[user]
            if len(item_list) > 0:
                if isinstance(item_list[0], (list, tuple)):
                    tmp_recall_items = [int(x[0]) for x in item_list[:k]]
                else:
                    tmp_recall_items = [int(x) for x in item_list[:k]]
            else:
                tmp_recall_items = []
            
            last_click_item = int(last_click_item_dict[user])
            if last_click_item in set(tmp_recall_items):
                hit_num += 1
        
        return round(hit_num * 1.0 / len(valid_users), 5), len(valid_users)
    
    # 先检查两个实验的用户集合
    print(f"\n【诊断信息】")
    print(f"仅ID特征 - 召回用户数: {len(recall_dict_id_only)}")
    print(f"使用辅助特征 - 召回用户数: {len(recall_dict_with_aux)}")
    print(f"最后一次点击用户数: {len(trn_last_click_df)}")
    
    # 计算用户集合的交集
    id_only_users = set(recall_dict_id_only.keys())
    aux_users = set(recall_dict_with_aux.keys())
    last_click_users = set(trn_last_click_df['user_id'])
    
    print(f"仅ID特征 & 最后点击用户交集: {len(id_only_users & last_click_users)}")
    print(f"辅助特征 & 最后点击用户交集: {len(aux_users & last_click_users)}")
    print(f"仅ID特征 & 辅助特征交集: {len(id_only_users & aux_users)}")
    
    # 找出只在ID特征中有的用户
    only_in_id = id_only_users - aux_users
    only_in_aux = aux_users - id_only_users
    print(f"只在ID特征中有召回的用户数: {len(only_in_id)}")
    print(f"只在辅助特征中有召回的用户数: {len(only_in_aux)}")
    
    print(f"\n{'K值':<10} {'仅ID特征':<20} {'用户数':<10} {'使用辅助特征':<20} {'用户数':<10} {'提升':<15}")
    print("-" * 90)
    for k in [10, 20, 30, 40, 50]:
        recall_id, user_num_id = calculate_recall_at_k(recall_dict_id_only, trn_last_click_df, k)
        recall_aux, user_num_aux = calculate_recall_at_k(recall_dict_with_aux, trn_last_click_df, k)
        improvement = round((recall_aux - recall_id) / recall_id * 100, 2) if recall_id > 0 else 0.0
        print(f"k={k:<6} {recall_id:<20.5f} {user_num_id:<10} {recall_aux:<20.5f} {user_num_aux:<10} {improvement:>10.2f}%")
    
    print("\n" + "=" * 80)
    print("【分析建议】")
    print("如果用户数不同，可能是因为：")
    print("1. 辅助特征的 merge 操作导致某些用户的数据被过滤")
    print("2. 编码问题导致某些用户无法正确映射")
    print("3. 模型训练时某些用户的特征缺失导致无法生成 embedding")
    print("=" * 80)


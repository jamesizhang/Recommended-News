"""
基于 Transformer 的生成式召回模型
使用 next-token prediction 方式训练，支持用户特征和物品特征
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os
import pickle
import warnings
import random
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math

warnings.filterwarnings('ignore')

# 设置随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

data_path = '../data_raw/'
save_path = '../temp_results/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ===================== 数据加载函数 =====================
def get_all_click_df(data_path='../data_raw/', offline=True, sample_size=None):
    """读取点击数据"""
    if offline:
        all_click = pd.read_csv(data_path + 'train_click_log.csv')
    else:
        trn_click = pd.read_csv(data_path + 'train_click_log.csv')
        tst_click = pd.read_csv(data_path + 'testA_click_log.csv')
        all_click = pd.concat([trn_click, tst_click], ignore_index=True)
    
    all_click = all_click.drop_duplicates(['user_id', 'click_article_id', 'click_timestamp'])
    print(f"all_click.shape: {all_click.shape}")
    print(f"unique users: {all_click['user_id'].nunique()}")
    print(f"unique items: {all_click['click_article_id'].nunique()}")
    
    if sample_size and len(all_click) > sample_size:
        all_click = all_click.sample(n=sample_size, random_state=1)
    return all_click

def get_item_info_df(data_path):
    """读取文章信息"""
    item_info_df = pd.read_csv(data_path + 'articles.csv')
    item_info_df = item_info_df.rename(columns={'article_id': 'click_article_id'})
    return item_info_df

def get_item_emb_dict(data_path):
    """读取文章 embedding"""
    item_emb_df = pd.read_csv(data_path + 'articles_emb.csv')
    item_emb_cols = [x for x in item_emb_df.columns if 'emb' in x]
    item_emb_np = np.ascontiguousarray(item_emb_df[item_emb_cols].values, dtype=np.float32)
    # 归一化
    item_emb_np = item_emb_np / (np.linalg.norm(item_emb_np, axis=1, keepdims=True) + 1e-8)
    
    item_emb_dict = dict(zip(item_emb_df['article_id'], item_emb_np))
    return item_emb_dict

def get_hist_and_last_click(all_click):
    """获取历史点击和最后一次点击"""
    all_click = all_click.sort_values(by=['user_id', 'click_timestamp'])
    click_last_df = all_click.groupby('user_id').tail(1)
    
    def hist_func(user_df):
        if len(user_df) == 1:
            return user_df
        else:
            return user_df[:-1]
    
    click_hist_df = all_click.groupby('user_id').apply(hist_func).reset_index(drop=True)
    return click_hist_df, click_last_df

# ===================== Transformer 模型定义 =====================
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerRecallModel(nn.Module):
    """基于 Transformer 的生成式召回模型"""
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, 
                 dim_feedforward=1024, max_seq_len=100, dropout=0.1,
                 user_vocab_size=None, user_feat_vocab_sizes=None, item_feat_dims=None):
        """
        Args:
            vocab_size: 物品词汇表大小（包含所有 article_id）
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: Transformer 层数
            dim_feedforward: FFN 维度
            max_seq_len: 最大序列长度
            dropout: Dropout 率
            user_vocab_size: 用户词汇表大小（如果为 None，则不使用 user_id embedding）
            user_feat_vocab_sizes: 用户特征词汇表大小字典，如 {'click_os': 10, 'click_deviceGroup': 5, 'click_region': 20}
            item_feat_dims: 物品特征维度字典，如 {'words_count': 1, 'emb': 250}
        """
        super(TransformerRecallModel, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.user_vocab_size = user_vocab_size
        
        # 物品 ID embedding
        self.item_embedding = nn.Embedding(vocab_size, d_model)
        
        # 用户 ID embedding（可选，如果提供了 user_vocab_size）
        if user_vocab_size is not None:
            self.user_embedding = nn.Embedding(user_vocab_size, d_model)
        else:
            self.user_embedding = None
        
        # 用户特征 embedding（可选）
        self.user_feat_embeddings = nn.ModuleDict()
        if user_feat_vocab_sizes:
            user_feat_dim = 0
            for feat_name, feat_vocab_size in user_feat_vocab_sizes.items():  # 修复变量名冲突
                emb_dim = min(32, feat_vocab_size // 2)  # 根据词汇表大小调整 embedding 维度
                self.user_feat_embeddings[feat_name] = nn.Embedding(feat_vocab_size, emb_dim)
                user_feat_dim += emb_dim
            # 将用户特征投影到 d_model
            if user_feat_dim > 0:
                self.user_feat_proj = nn.Linear(user_feat_dim, d_model)
            else:
                self.user_feat_proj = None
        else:
            self.user_feat_proj = None
        
        # 物品特征处理（可选）
        self.item_feat_proj = None
        if item_feat_dims:
            item_feat_dim = sum(item_feat_dims.values())
            if item_feat_dim > 0:
                self.item_feat_proj = nn.Linear(item_feat_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer 编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # (seq_len, batch, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 输出层：预测下一个物品
        self.output_proj = nn.Linear(d_model, vocab_size)
        print(f"模型初始化: output_proj 维度: {d_model} -> {vocab_size}")
        
        self.dropout = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        initrange = 0.1
        self.item_embedding.weight.data.uniform_(-initrange, initrange)
        if self.user_embedding is not None:
            self.user_embedding.weight.data.uniform_(-initrange, initrange)
        if self.user_feat_proj is not None:
            self.user_feat_proj.weight.data.uniform_(-initrange, initrange)
        if self.item_feat_proj is not None:
            self.item_feat_proj.weight.data.uniform_(-initrange, initrange)
        self.output_proj.bias.data.zero_()
        self.output_proj.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, item_ids, user_ids=None, user_feats=None, item_feats=None, src_mask=None):
        """
        Args:
            item_ids: (seq_len, batch_size) 物品 ID 序列（第一个位置可能是 user_id，需要单独处理）
            user_ids: (batch_size,) 用户 ID（可选，如果提供则第一个位置使用 user_embedding）
            user_feats: dict, 用户特征，如 {'click_os': (batch_size,), 'click_deviceGroup': (batch_size,), ...}
            item_feats: dict, 物品特征，如 {'words_count': (seq_len, batch_size, 1), 'emb': (seq_len, batch_size, 250)}
            src_mask: (seq_len, seq_len) 注意力掩码（causal mask）
        Returns:
            output: (seq_len, batch_size, vocab_size) 预测下一个物品的 logits
        """
        seq_len, batch_size = item_ids.shape
        
        # 物品 ID embedding（所有位置）
        item_emb = self.item_embedding(item_ids)  # (seq_len, batch_size, d_model)
        
        # 如果提供了 user_ids 和 user_embedding，第一个位置使用 user_embedding
        if user_ids is not None and self.user_embedding is not None and seq_len > 0:
            user_emb = self.user_embedding(user_ids)  # (batch_size, d_model)
            item_emb[0] = user_emb.unsqueeze(0)  # (1, batch_size, d_model) -> 替换第一个位置
        
        # 添加用户特征（叠加到第一个位置）
        # 特征融合方式：多个特征 concat → Linear 投影 → 与主 embedding 相加
        # 注意：当前使用相加方式，如需对比性能可改为 concat（需要调整 d_model 维度）
        if user_feats is not None and self.user_feat_proj is not None:
            user_feat_embs = []
            for feat_name, feat_values in user_feats.items():
                if feat_name in self.user_feat_embeddings:
                    # feat_values: (batch_size,)
                    feat_emb = self.user_feat_embeddings[feat_name](feat_values)  # (batch_size, emb_dim)
                    user_feat_embs.append(feat_emb)
            if user_feat_embs:
                # 多个用户特征 concat
                user_feat_emb = torch.cat(user_feat_embs, dim=1)  # (batch_size, total_dim)
                # Linear 投影到 d_model
                user_feat_emb = self.user_feat_proj(user_feat_emb)  # (batch_size, d_model)
                # 与主 embedding 相加（当前方式，保留以便后续对比）
                if seq_len > 0:
                    item_emb[0] = item_emb[0] + user_feat_emb.unsqueeze(0)  # 广播到第一个位置
                # 如需改为 concat，可改为：
                # item_emb = torch.cat([item_emb, user_feat_emb.unsqueeze(0).expand(seq_len, -1, -1)], dim=2)
                # 但需要调整后续的 d_model 维度
        
        # 添加物品特征
        # 特征融合方式：多个特征 concat → Linear 投影 → 与主 embedding 相加
        # 注意：当前使用相加方式，如需对比性能可改为 concat（需要调整 d_model 维度）
        if item_feats is not None and self.item_feat_proj is not None:
            item_feat_list = []
            for feat_name, feat_values in item_feats.items():
                # feat_values: (seq_len, batch_size, feat_dim)
                item_feat_list.append(feat_values)
            if item_feat_list:
                # 多个物品特征 concat
                item_feat_emb = torch.cat(item_feat_list, dim=2)  # (seq_len, batch_size, total_dim)
                # Linear 投影到 d_model
                item_feat_emb = self.item_feat_proj(item_feat_emb)  # (seq_len, batch_size, d_model)
                # 与主 embedding 相加（当前方式，保留以便后续对比）
                item_emb = item_emb + item_feat_emb
                # 如需改为 concat，可改为：
                # item_emb = torch.cat([item_emb, item_feat_emb], dim=2)
                # 但需要调整后续的 d_model 维度
        
        # 位置编码
        item_emb = item_emb * math.sqrt(self.d_model)
        item_emb = self.pos_encoder(item_emb)
        item_emb = self.dropout(item_emb)
        
        # Transformer 编码
        output = self.transformer_encoder(item_emb, mask=src_mask)  # (seq_len, batch_size, d_model)
        
        # 预测下一个物品
        # 检查 output_proj 的维度
        if not hasattr(self, '_output_proj_checked'):
            print(f"forward 中检查: output_proj 权重形状: {self.output_proj.weight.shape}")
            print(f"  output.shape 在投影前: {output.shape}")
            self._output_proj_checked = True
        
        output = self.output_proj(output)  # (seq_len, batch_size, vocab_size)
        
        return output

def generate_square_subsequent_mask(sz):
    """生成 causal mask（下三角矩阵）"""
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# ===================== 数据集类 =====================
class RecallDataset(Dataset):
    """召回数据集"""
    def __init__(self, sequences, user_feats_dict=None, item_feats_dict=None, 
                 max_seq_len=100, pad_token=0):
        """
        Args:
            sequences: list of (user_id, item_list), item_list 是按时间排序的物品序列
            user_feats_dict: dict, {user_id: {'click_os': val, 'click_deviceGroup': val, ...}}
            item_feats_dict: dict, {item_id: {'words_count': val, 'emb': array, ...}}
            max_seq_len: 最大序列长度
            pad_token: padding token
        """
        self.sequences = sequences
        self.user_feats_dict = user_feats_dict or {}
        self.item_feats_dict = item_feats_dict or {}
        self.max_seq_len = max_seq_len
        self.pad_token = pad_token
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        user_id, item_list = self.sequences[idx]
        
        # 构建训练样本：对于序列 [item1, item2, item3, ...]
        # 生成多个样本：(user, item1) -> item2, (user, item1, item2) -> item3, ...
        # 这里我们返回整个序列，在 collate_fn 中处理
        
        # 获取用户特征
        user_feats = self.user_feats_dict.get(user_id, {})
        
        # 获取物品特征
        item_feats = {}
        for item_id in item_list:
            if item_id in self.item_feats_dict:
                item_feats[item_id] = self.item_feats_dict[item_id]
        
        return {
            'user_id': user_id,
            'item_list': item_list,
            'user_feats': user_feats,
            'item_feats': item_feats
        }

def collate_fn(batch, max_seq_len=100, pad_token=0, item_feat_dims=None):
    """批处理函数：为每个序列生成完整的 next-token prediction 样本"""
    all_inputs = []
    all_targets = []  # 现在是 (batch_size, seq_len) 形状，每个位置对应下一个 token
    all_user_ids = []  # 单独存储 user_id
    all_user_feats = []
    all_item_feats = []
    
    # 确定物品特征维度
    if item_feat_dims is None:
        item_feat_dims = {}
        # 从第一个样本中推断特征维度
        for sample in batch:
            if sample['item_feats']:
                sample_item_feats = next(iter(sample['item_feats'].values()))
                if 'words_count' in sample_item_feats:
                    item_feat_dims['words_count'] = 1
                if 'emb' in sample_item_feats:
                    item_feat_dims['emb'] = len(sample_item_feats['emb'])
                break
    
    for sample in batch:
        user_id = sample['user_id']
        item_list = sample['item_list']
        user_feats = sample['user_feats']
        
        if len(item_list) < 2:
            continue  # 至少需要2个物品才能生成训练样本
        
        # 截断序列
        if len(item_list) > max_seq_len:
            item_list = item_list[:max_seq_len]
        
        # 输入序列：[userID, item1, item2, ..., item_n-1]（第一个位置是 userID 占位符，后面是 items）
        # 目标序列：[item1, item2, ..., item_n]（每个位置预测下一个 token）
        # 注意：第一个位置（userID）预测 item1，第二个位置（item1）预测 item2，以此类推
        input_seq = item_list[:-1]  # 去掉最后一个 item，作为输入
        target_seq = item_list  # 完整序列作为目标（第一个位置是 item1，对应 userID 位置的预测）
        
        # Padding 输入序列（第一个位置是 userID 占位符，后面是 items，不够长的话后面补 padding）
        # 格式：[userID, item1, item2, ..., 0, 0]
        if len(input_seq) < max_seq_len - 1:
            padded_input_seq = [pad_token] + input_seq + [pad_token] * (max_seq_len - 1 - len(input_seq))
        else:
            padded_input_seq = [pad_token] + input_seq[-(max_seq_len - 1):]
        
        # Padding 目标序列（第一个位置是 item1，后面依次是 item2, item3, ...，不够长的话后面补 padding）
        # 格式：[item1, item2, item3, ..., 0, 0]
        if len(target_seq) < max_seq_len:
            padded_target_seq = target_seq + [pad_token] * (max_seq_len - len(target_seq))
        else:
            padded_target_seq = target_seq[:max_seq_len]
        
        # 确保长度一致（padding 在后面）
        seq_len = len(padded_input_seq)
        if len(padded_target_seq) < seq_len:
            padded_target_seq = padded_target_seq + [pad_token] * (seq_len - len(padded_target_seq))
        elif len(padded_target_seq) > seq_len:
            padded_target_seq = padded_target_seq[:seq_len]
        
        all_inputs.append(padded_input_seq)
        all_targets.append(padded_target_seq)
        all_user_ids.append(user_id)
        all_user_feats.append(user_feats)
        
        # 获取输入序列中每个物品的特征
        seq_item_feats = {}
        for item_id in input_seq:
            if item_id in sample['item_feats']:
                seq_item_feats[item_id] = sample['item_feats'][item_id]
        all_item_feats.append(seq_item_feats)
    
    if len(all_inputs) == 0:
        return None
    
    # 统一序列长度（取最大长度）
    max_len = max(len(seq) for seq in all_inputs)
    padded_inputs = []
    padded_targets = []
    padded_item_feats_list = []
    
    for input_seq, target_seq, item_feats in zip(all_inputs, all_targets, all_item_feats):
        # 确保长度一致（padding 在后面）
        if len(input_seq) < max_len:
            input_seq = input_seq + [pad_token] * (max_len - len(input_seq))
        if len(target_seq) < max_len:
            target_seq = target_seq + [pad_token] * (max_len - len(target_seq))
        
        padded_inputs.append(input_seq)
        padded_targets.append(target_seq)
        
        # Padding 物品特征（第一个位置是 user_id 占位符，使用默认值）
        padded_item_feats = {}
        for feat_name in item_feat_dims.keys():
            feat_list = []
            # 第一个位置是 user_id 占位符，使用默认值
            if feat_name == 'words_count':
                feat_list.append(np.array([0.0], dtype=np.float32))
            elif feat_name == 'emb':
                feat_list.append(np.zeros(item_feat_dims[feat_name], dtype=np.float32))
            # 后面的位置是物品特征
            for item_id in input_seq[1:]:  # 跳过第一个占位符
                if item_id in item_feats and feat_name in item_feats[item_id]:
                    feat_value = item_feats[item_id][feat_name]
                    if isinstance(feat_value, np.ndarray):
                        feat_list.append(feat_value)
                    else:
                        feat_list.append(np.array([feat_value], dtype=np.float32))
                else:
                    # 使用默认值
                    if feat_name == 'words_count':
                        feat_list.append(np.array([0.0], dtype=np.float32))
                    elif feat_name == 'emb':
                        feat_list.append(np.zeros(item_feat_dims[feat_name], dtype=np.float32))
            # 如果特征数量不足，补充 padding（在后面补）
            while len(feat_list) < max_len:
                if feat_name == 'words_count':
                    feat_list.append(np.array([0.0], dtype=np.float32))
                elif feat_name == 'emb':
                    feat_list.append(np.zeros(item_feat_dims[feat_name], dtype=np.float32))
            # Stack: (seq_len, feat_dim)
            padded_item_feats[feat_name] = np.stack(feat_list, axis=0)
        padded_item_feats_list.append(padded_item_feats)
    
    return {
        'input_ids': torch.LongTensor(padded_inputs),  # (batch_size, seq_len)
        'user_ids': torch.LongTensor(all_user_ids),  # (batch_size,) 单独存储 user_id
        'targets': torch.LongTensor(padded_targets),  # (batch_size, seq_len) 每个位置对应下一个 token
        'user_feats': all_user_feats,
        'item_feats': padded_item_feats_list
    }

# ===================== 训练和召回函数 =====================
def prepare_data(all_click_df, item_info_df, item_emb_dict, 
                 use_user_feats=True, use_item_feats=True):
    """准备训练数据"""
    print("准备训练数据...")
    
    # 按用户分组，获取每个用户的点击序列（按时间排序）
    sequences = []
    for user_id, group in tqdm(all_click_df.groupby('user_id')):
        group = group.sort_values('click_timestamp')
        item_list = group['click_article_id'].tolist()
        if len(item_list) >= 2:  # 至少需要2个物品
            sequences.append((user_id, item_list))
    
    print(f"生成了 {len(sequences)} 个用户序列")
    
    # 准备用户特征
    user_feats_dict = {}
    if use_user_feats:
        user_feat_cols = ['click_os', 'click_deviceGroup', 'click_region']
        for user_id, group in tqdm(all_click_df.groupby('user_id')):
            user_feats = {}
            for col in user_feat_cols:
                if col in group.columns:
                    # 使用众数
                    mode_val = group[col].mode()
                    if len(mode_val) > 0:
                        user_feats[col] = mode_val.iloc[0]
            if user_feats:
                user_feats_dict[user_id] = user_feats
    
    # 准备物品特征
    item_feats_dict = {}
    if use_item_feats:
        # 归一化 words_count
        if 'words_count' in item_info_df.columns:
            scaler = MinMaxScaler()
            item_info_df['words_count_norm'] = scaler.fit_transform(
                item_info_df[['words_count']]
            ).flatten()
        
        for _, row in tqdm(item_info_df.iterrows()):
            item_id = row['click_article_id']
            item_feats = {}
            
            if 'words_count_norm' in item_info_df.columns:
                item_feats['words_count'] = float(row['words_count_norm'])
            
            if item_id in item_emb_dict:
                item_feats['emb'] = item_emb_dict[item_id]
            
            if item_feats:
                item_feats_dict[item_id] = item_feats
    
    return sequences, user_feats_dict, item_feats_dict

def train_transformer_recall(all_click_df, item_info_df, item_emb_dict,
                             use_user_feats=True, use_item_feats=True,
                             d_model=32, nhead=8, num_layers=4,
                             dim_feedforward=1024, max_seq_len=100,
                             batch_size=32, epochs=10, lr=0.001,
                             save_model_path=None):
    """训练 Transformer 召回模型"""
    # 获取全局 device 变量
    global device
    
    print("=" * 80)
    print("开始训练 Transformer 召回模型")
    print("=" * 80)
    
    # 准备数据
    sequences, user_feats_dict, item_feats_dict = prepare_data(
        all_click_df, item_info_df, item_emb_dict,
        use_user_feats=use_user_feats, use_item_feats=use_item_feats
    )
    
    # 编码用户和物品 ID
    print("编码用户和物品 ID...")
    all_user_ids = set()
    all_item_ids = set()
    for user_id, item_list in sequences:
        all_user_ids.add(user_id)
        all_item_ids.update(item_list)
    
    # 物品编码器（包含所有物品）
    # 注意：LabelEncoder 编码从 0 开始，所以索引范围是 [0, vocab_size-1]
    # 为了给 padding token (0) 预留位置，我们需要将 padding token 设置为 vocab_size
    # 或者将真实 ID 编码为 [1, vocab_size]，0 作为 padding
    item_encoder = LabelEncoder()
    all_item_ids = sorted(list(all_item_ids))
    item_encoder.fit(all_item_ids)
    vocab_size = len(item_encoder.classes_)
    # 为 padding token 预留位置，所以 vocab_size 需要 +1
    # 真实物品 ID 编码为 [1, vocab_size]，0 作为 padding
    vocab_size_with_pad = vocab_size + 1
    print(f"物品词汇表大小: {vocab_size} (加上 padding: {vocab_size_with_pad})")
    
    # 用户编码器
    user_encoder = LabelEncoder()
    all_user_ids = sorted(list(all_user_ids))
    user_encoder.fit(all_user_ids)
    
    # 编码用户特征
    user_feat_encoders = {}
    user_feat_vocab_sizes = {}
    if use_user_feats and user_feats_dict:
        for feat_name in ['click_os', 'click_deviceGroup', 'click_region']:
            feat_values = []
            for user_feats in user_feats_dict.values():
                if feat_name in user_feats:
                    feat_values.append(user_feats[feat_name])
            if feat_values:
                encoder = LabelEncoder()
                encoder.fit(feat_values)
                user_feat_encoders[feat_name] = encoder
                user_feat_vocab_sizes[feat_name] = len(encoder.classes_)
                print(f"用户特征 {feat_name} 词汇表大小: {user_feat_vocab_sizes[feat_name]}")
    
    # 编码序列
    # 将物品 ID 编码后 +1，这样真实 ID 范围是 [1, vocab_size]，0 作为 padding
    # user_id 保持原始编码，使用独立的 user_embedding
    print("编码序列...")
    encoded_sequences = []
    for user_id, item_list in tqdm(sequences):
        encoded_user_id = user_encoder.transform([user_id])[0]  # 保持原始编码，不映射
        try:
            # item_encoder.transform() 返回 [0, vocab_size-1]，+1 后变成 [1, vocab_size]
            encoded_raw = item_encoder.transform(item_list).tolist()
            encoded_item_list = [x + 1 for x in encoded_raw]  # +1 为 padding 预留位置
            # 验证：编码后的值应该在 [1, vocab_size] 范围内
            max_encoded = max(encoded_item_list) if encoded_item_list else 0
            min_encoded = min(encoded_item_list) if encoded_item_list else 0
            if max_encoded > vocab_size or min_encoded < 1:
                print(f"错误: 编码后的物品 ID 超出范围!")
                print(f"  user_id: {user_id}, item_list 长度: {len(item_list)}")
                print(f"  encoded_raw 范围: [{min(encoded_raw) if encoded_raw else 0}, {max(encoded_raw) if encoded_raw else 0}]")
                print(f"  encoded_item_list 范围: [{min_encoded}, {max_encoded}]")
                print(f"  vocab_size: {vocab_size}, vocab_size_with_pad: {vocab_size_with_pad}")
                raise ValueError(f"编码后的物品 ID 超出范围: [{min_encoded}, {max_encoded}], 期望: [1, {vocab_size}]")
            encoded_sequences.append((encoded_user_id, encoded_item_list))
        except ValueError as e:
            # 如果物品不在编码器中，跳过这个序列
            print(f"警告: 用户 {user_id} 的序列中有物品不在编码器中: {e}")
            continue
    
    # 用户词汇表大小（用于创建 user_embedding）
    user_vocab_size = len(user_encoder.classes_)
    print(f"用户词汇表大小: {user_vocab_size}")
    
    # 编码用户特征（只对训练序列中的用户进行编码）
    # 注意：key 使用原始的 encoded_user_id（不映射）
    encoded_user_feats_dict = {}
    if use_user_feats:
        # 只处理在训练序列中的用户
        train_user_ids = set(all_user_ids)
        for user_id, user_feats in user_feats_dict.items():
            # 只处理在训练序列中的用户
            if user_id not in train_user_ids:
                continue
            try:
                encoded_user_id = user_encoder.transform([user_id])[0]  # 保持原始编码
                encoded_feats = {}
                for feat_name, feat_value in user_feats.items():
                    if feat_name in user_feat_encoders:
                        try:
                            encoded_feats[feat_name] = user_feat_encoders[feat_name].transform([feat_value])[0]
                        except ValueError:
                            # 如果特征值不在编码器中，使用默认值 0
                            encoded_feats[feat_name] = 0
                if encoded_feats:
                    encoded_user_feats_dict[encoded_user_id] = encoded_feats
            except ValueError:
                # 如果用户不在编码器中，跳过
                continue
    
    # 创建数据集
    dataset = RecallDataset(
        encoded_sequences,
        user_feats_dict=encoded_user_feats_dict,
        item_feats_dict=item_feats_dict,  # 物品特征不需要编码（已经是数值）
        max_seq_len=max_seq_len
    )
    
    # 确定物品特征维度
    item_feat_dims = {}
    if use_item_feats and item_feats_dict:
        sample_item_feats = next(iter(item_feats_dict.values()))
        if 'words_count' in sample_item_feats:
            item_feat_dims['words_count'] = 1
        if 'emb' in sample_item_feats:
            item_feat_dims['emb'] = len(sample_item_feats['emb'])
    
    # 创建数据加载器
    def custom_collate_fn(batch):
        result = collate_fn(batch, max_seq_len=max_seq_len, pad_token=0, item_feat_dims=item_feat_dims)
        if result is None:
            return None
        # 检查 targets 的范围（targets 现在是 (batch_size, seq_len)，允许 0 作为 padding）
        targets = result['targets']
        targets_min = targets.min().item()
        targets_max = targets.max().item()
        if targets_max >= vocab_size_with_pad or targets_min < 0:
            print(f"错误: targets 超出有效范围!")
            print(f"  targets 范围: [{targets_min}, {targets_max}]")
            print(f"  期望范围: [0, {vocab_size_with_pad-1}] (0 是 padding)")
            print(f"  vocab_size: {vocab_size}, vocab_size_with_pad: {vocab_size_with_pad}")
            print(f"  targets.shape: {targets.shape}")
            # 找出超出范围的值
            invalid_mask = (targets < 0) | (targets >= vocab_size_with_pad)
            invalid_values = targets[invalid_mask].unique().tolist()
            print(f"  超出范围的值: {invalid_values[:10]} (共 {invalid_mask.sum().item()} 个)")
            raise ValueError(f"targets 超出范围: [{targets_min}, {targets_max}], 期望: [0, {vocab_size_with_pad-1}]")
        return result
    
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=custom_collate_fn, num_workers=0
    )
    
    # 清理 GPU 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 创建模型（使用包含 padding 的 vocab_size）
    print(f"创建模型: vocab_size_with_pad={vocab_size_with_pad}, d_model={d_model}")
    model = TransformerRecallModel(
        vocab_size=vocab_size_with_pad,  # 包含 padding token
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_seq_len=max_seq_len,
        user_vocab_size=user_vocab_size,  # 用户词汇表大小
        user_feat_vocab_sizes=user_feat_vocab_sizes if use_user_feats else None,
        item_feat_dims=item_feat_dims if use_item_feats else None
    )
    # 验证 output_proj 的维度
    print(f"模型创建后验证: output_proj.weight.shape = {model.output_proj.weight.shape}, 期望: ({d_model}, {vocab_size_with_pad})")
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 将模型移到 GPU（如果可用）
    try:
        model = model.to(device)
        print(f"模型已加载到 {device}")
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"GPU 内存不足，尝试使用 CPU...")
            device = torch.device('cpu')
            model = model.to(device)
        else:
            raise
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 padding
    
    # 训练
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            if batch is None:
                continue
            
            input_ids = batch['input_ids'].to(device)  # (batch_size, seq_len)
            user_ids = batch['user_ids'].to(device)  # (batch_size,) 单独的用户 ID
            targets = batch['targets'].to(device)  # (batch_size, seq_len) 每个位置对应下一个 token
            user_feats = batch['user_feats']
            item_feats_list = batch['item_feats']
            
            # 检查索引范围（不应该需要 clamp，如果超出范围说明编码有问题）
            if batch_idx == 0:
                targets_max = targets.max().item()
                targets_min = targets.min().item()
                input_ids_max = input_ids.max().item()
                input_ids_min = input_ids.min().item()
                user_ids_max = user_ids.max().item()
                user_ids_min = user_ids.min().item()
                print(f"第一个 batch 的索引范围:")
                print(f"  input_ids: [{input_ids_min}, {input_ids_max}], vocab_size_with_pad: {vocab_size_with_pad}")
                print(f"  user_ids: [{user_ids_min}, {user_ids_max}], user_vocab_size: {user_vocab_size}")
                print(f"  targets: [{targets_min}, {targets_max}], 期望: [0, {vocab_size_with_pad-1}] (0 是 padding)")
                print(f"  targets.shape: {targets.shape}, input_ids.shape: {input_ids.shape}")
                
                # 如果超出范围，报错（允许 0 作为 padding）
                if targets_max >= vocab_size_with_pad or targets_min < 0:
                    invalid_mask = (targets < 0) | (targets >= vocab_size_with_pad)
                    invalid_values = targets[invalid_mask].unique().tolist()
                    print(f"  错误: targets 超出范围!")
                    print(f"  无效值: {invalid_values}")
                    raise ValueError(f"targets 超出范围: [{targets_min}, {targets_max}], 期望: [0, {vocab_size_with_pad-1}]")
            
            # 转换为 (seq_len, batch_size) 格式
            input_ids = input_ids.transpose(0, 1)  # (seq_len, batch_size)
            targets = targets.transpose(0, 1)  # (seq_len, batch_size) 每个位置对应下一个 token
            
            # 准备用户特征
            batch_user_feats = {}
            if use_user_feats and user_feat_vocab_sizes:
                for feat_name in user_feat_vocab_sizes.keys():
                    feat_values = []
                    for user_feat_dict in user_feats:
                        if feat_name in user_feat_dict:
                            feat_val = user_feat_dict[feat_name]
                            # 确保特征值在有效范围内
                            if feat_val >= 0 and feat_val < user_feat_vocab_sizes[feat_name]:
                                feat_values.append(feat_val)
                            else:
                                feat_values.append(0)  # 默认值
                        else:
                            feat_values.append(0)  # 默认值
                    batch_user_feats[feat_name] = torch.LongTensor(feat_values).to(device)
            
            # 准备物品特征
            batch_item_feats = {}
            if use_item_feats and item_feat_dims:
                for feat_name in item_feat_dims.keys():
                    feat_values = []
                    for item_feat_dict in item_feats_list:
                        if feat_name in item_feat_dict:
                            # item_feat_dict[feat_name] 已经是 (seq_len, feat_dim) 形状
                            feat_values.append(item_feat_dict[feat_name])
                        else:
                            # 默认值
                            seq_len = input_ids.shape[0]
                            if feat_name == 'words_count':
                                feat_values.append(np.zeros((seq_len, 1), dtype=np.float32))
                            elif feat_name == 'emb':
                                feat_values.append(np.zeros((seq_len, item_feat_dims[feat_name]), dtype=np.float32))
                    # Stack: (batch_size, seq_len, feat_dim) -> (seq_len, batch_size, feat_dim)
                    feat_array = np.stack(feat_values, axis=1)  # (seq_len, batch_size, feat_dim)
                    batch_item_feats[feat_name] = torch.FloatTensor(feat_array).to(device)
            
            # 生成 causal mask
            seq_len = input_ids.shape[0]
            src_mask = generate_square_subsequent_mask(seq_len).to(device)
            
            # 前向传播
            optimizer.zero_grad()
            output = model(input_ids, user_ids=user_ids,  # 单独传递 user_ids
                          user_feats=batch_user_feats if batch_user_feats else None,
                          item_feats=batch_item_feats if batch_item_feats else None,
                          src_mask=src_mask)  # (seq_len, batch_size, vocab_size)
            
            # 计算损失（对所有位置进行 next-token prediction）
            # output: (seq_len, batch_size, vocab_size)
            # targets: (seq_len, batch_size) 每个位置对应下一个 token
            # 需要将 output 和 targets 展平，对所有位置计算损失
            
            # 在计算损失前，检查 logits 和 targets 的维度
            if batch_idx == 0:
                print(f"计算损失前的检查:")
                print(f"  output.shape: {output.shape}, 期望: (seq_len, batch_size, {vocab_size_with_pad})")
                print(f"  targets.shape: {targets.shape}, 期望: (seq_len, batch_size)")
                print(f"  targets 所有唯一值范围: [{targets.min().item()}, {targets.max().item()}]")
                # 检查 output 的维度
                if output.shape[2] != vocab_size_with_pad:
                    print(f"  错误: output 的维度不对!")
                    print(f"  output.shape[2]: {output.shape[2]}, vocab_size_with_pad: {vocab_size_with_pad}")
                    raise ValueError(f"output 维度不匹配: {output.shape[2]} != {vocab_size_with_pad}")
                # 检查是否有超出范围的值
                invalid_mask = (targets < 0) | (targets >= vocab_size_with_pad)
                if invalid_mask.any():
                    invalid_values = targets[invalid_mask].unique().tolist()
                    print(f"  错误: targets 中有 {invalid_mask.sum().item()} 个值超出范围!")
                    print(f"  无效值: {invalid_values}")
                    print(f"  期望范围: [0, {vocab_size_with_pad-1}]")
                    raise ValueError(f"targets 中有值超出范围: {invalid_values}")
            
            # 展平：将所有位置的预测和目标展平
            # output: (seq_len, batch_size, vocab_size) -> (seq_len * batch_size, vocab_size)
            # targets: (seq_len, batch_size) -> (seq_len * batch_size,)
            logits_flat = output.reshape(-1, vocab_size_with_pad)  # (seq_len * batch_size, vocab_size)
            targets_flat = targets.reshape(-1)  # (seq_len * batch_size,)
            
            loss = criterion(logits_flat, targets_flat)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 清理GPU缓存（每2个batch清理一次，避免内存累积）
            if batch_idx % 2 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 如果内存不足，尝试清理更多
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    # 保存模型和编码器
    if save_model_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'item_encoder': item_encoder,
            'user_encoder': user_encoder,
            'user_feat_encoders': user_feat_encoders,
            'vocab_size': vocab_size,
            'vocab_size_with_pad': vocab_size_with_pad,  # 保存包含 padding 的 vocab_size
            'user_feat_vocab_sizes': user_feat_vocab_sizes,
            'item_feat_dims': item_feat_dims,
            'model_config': {
                'd_model': d_model,
                'nhead': nhead,
                'num_layers': num_layers,
                'dim_feedforward': dim_feedforward,
                'max_seq_len': max_seq_len
            }
        }, save_model_path)
        print(f"模型已保存到: {save_model_path}")
    
    return model, item_encoder, user_encoder, user_feat_encoders, vocab_size_with_pad

def recall_transformer(model, user_hist_dict, item_encoder, user_encoder, 
                      user_feat_encoders, user_feats_dict, item_feats_dict,
                      all_item_ids, vocab_size_with_pad, topk=50, max_seq_len=100,
                      use_user_feats=True, use_item_feats=True, batch_size=32):
    """使用 Transformer 模型进行批量召回"""
    # 获取全局 device 变量
    global device
    
    print("开始批量召回...")
    model.eval()
    
    user_recall_dict = {}
    
    # 确定物品特征维度（提前确定，避免重复计算）
    item_feat_dims = {}
    if use_item_feats and item_feats_dict:
        sample_item_feats = next(iter(item_feats_dict.values()))
        if 'words_count' in sample_item_feats:
            item_feat_dims['words_count'] = 1
        if 'emb' in sample_item_feats:
            item_feat_dims['emb'] = len(sample_item_feats['emb'])
    
    # 将用户分批处理
    user_items = list(user_hist_dict.items())
    total_users = len(user_items)
    
    with torch.no_grad():
        for batch_start in tqdm(range(0, total_users, batch_size), desc="召回进度"):
            batch_end = min(batch_start + batch_size, total_users)
            batch_users = user_items[batch_start:batch_end]
            
            # 准备批量数据
            batch_input_seqs = []
            batch_user_ids = []
            batch_encoded_user_ids = []
            batch_user_feats_list = []
            batch_item_feats_list = []
            batch_clicked_items = []  # 用于过滤已点击的物品
            
            for user_id, item_list in batch_users:
                # 编码用户和物品
                try:
                    encoded_user_id = user_encoder.transform([user_id])[0]
                except:
                    continue  # 跳过未训练的用户
                
                if len(item_list) == 0:
                    continue
                
                # 编码物品序列（+1 为 padding 预留位置）
                try:
                    encoded_item_list = [x + 1 for x in item_encoder.transform(item_list).tolist()]
                except:
                    # 如果物品不在训练集中，跳过
                    continue
                
                # 构建输入序列：[item1, item2, ...]（不包含 user_id，user_id 单独传递）
                input_seq = encoded_item_list[-max_seq_len+1:]  # 保留最后 max_seq_len-1 个物品
                
                # Padding（使用 0 作为 padding token）
                if len(input_seq) < max_seq_len - 1:  # -1 为 user_id 预留位置
                    input_seq = [0] * (max_seq_len - 1 - len(input_seq)) + input_seq
                
                # 在前面加一个 0 作为 user_id 占位符（实际 user_id 单独传递）
                input_seq = [0] + input_seq
                
                # 确保索引在有效范围内
                input_seq = [min(max(x, 0), vocab_size_with_pad - 1) for x in input_seq]
                
                batch_input_seqs.append(input_seq)
                batch_user_ids.append(user_id)
                batch_encoded_user_ids.append(encoded_user_id)
                batch_clicked_items.append(set(item_list))
                
                # 准备用户特征
                user_feats = {}
                if use_user_feats and user_feat_encoders:
                    encoded_user_feats = user_feats_dict.get(user_id, {})
                    for feat_name in user_feat_encoders.keys():
                        if feat_name in encoded_user_feats:
                            try:
                                feat_value = user_feat_encoders[feat_name].transform([encoded_user_feats[feat_name]])[0]
                            except ValueError:
                                feat_value = 0
                        else:
                            feat_value = 0
                        user_feats[feat_name] = feat_value
                batch_user_feats_list.append(user_feats)
                
                # 准备物品特征
                item_feats = {}
                if use_item_feats and item_feats_dict:
                    for feat_name in item_feat_dims.keys():
                        feat_values = []
                        # 第一个位置是 user_id 占位符，使用默认值
                        if feat_name == 'words_count':
                            feat_values.append(np.array([0.0], dtype=np.float32))
                        elif feat_name == 'emb':
                            feat_values.append(np.zeros(item_feat_dims[feat_name], dtype=np.float32))
                        
                        # 处理后续位置的物品特征
                        for item_id in input_seq[1:]:  # 跳过第一个位置的 user_id 占位符
                            # 解码物品ID（需要先 -1，因为编码时 +1 了）
                            if item_id > 0 and item_id <= len(item_encoder.classes_):
                                try:
                                    raw_item_id = item_encoder.inverse_transform([item_id - 1])[0]  # -1 还原编码
                                    if raw_item_id in item_feats_dict and feat_name in item_feats_dict[raw_item_id]:
                                        feat_value = item_feats_dict[raw_item_id][feat_name]
                                        if isinstance(feat_value, np.ndarray):
                                            feat_values.append(feat_value)
                                        else:
                                            feat_values.append(np.array([feat_value], dtype=np.float32))
                                    else:
                                        # 默认值
                                        if feat_name == 'words_count':
                                            feat_values.append(np.array([0.0], dtype=np.float32))
                                        elif feat_name == 'emb':
                                            feat_values.append(np.zeros(item_feat_dims[feat_name], dtype=np.float32))
                                except:
                                    # 默认值
                                    if feat_name == 'words_count':
                                        feat_values.append(np.array([0.0], dtype=np.float32))
                                    elif feat_name == 'emb':
                                        feat_values.append(np.zeros(item_feat_dims[feat_name], dtype=np.float32))
                            else:
                                # Padding token (0) 或无效的 ID
                                if feat_name == 'words_count':
                                    feat_values.append(np.array([0.0], dtype=np.float32))
                                elif feat_name == 'emb':
                                    feat_values.append(np.zeros(item_feat_dims[feat_name], dtype=np.float32))
                        # Stack: (seq_len, feat_dim)
                        item_feats[feat_name] = np.stack(feat_values, axis=0)  # (seq_len, feat_dim)
                batch_item_feats_list.append(item_feats)
            
            if len(batch_input_seqs) == 0:
                continue
            
            # 统一序列长度（所有序列已经 padding 到 max_seq_len）
            seq_len = max_seq_len
            
            # 构建批量输入
            input_ids = torch.LongTensor(batch_input_seqs).to(device)  # (batch_size, seq_len)
            input_ids = input_ids.transpose(0, 1)  # (seq_len, batch_size)
            user_ids_tensor = torch.LongTensor(batch_encoded_user_ids).to(device)  # (batch_size,)
            
            # 准备批量用户特征
            batch_user_feats = {}
            if use_user_feats and user_feat_encoders and batch_user_feats_list:
                for feat_name in user_feat_encoders.keys():
                    feat_values = [feat_dict.get(feat_name, 0) for feat_dict in batch_user_feats_list]
                    batch_user_feats[feat_name] = torch.LongTensor(feat_values).to(device)  # (batch_size,)
            
            # 准备批量物品特征
            batch_item_feats = {}
            if use_item_feats and item_feat_dims and batch_item_feats_list:
                for feat_name in item_feat_dims.keys():
                    feat_arrays = []
                    for item_feat_dict in batch_item_feats_list:
                        if feat_name in item_feat_dict:
                            feat_arrays.append(item_feat_dict[feat_name])
                        else:
                            # 默认值
                            if feat_name == 'words_count':
                                feat_arrays.append(np.zeros((seq_len, 1), dtype=np.float32))
                            elif feat_name == 'emb':
                                feat_arrays.append(np.zeros((seq_len, item_feat_dims[feat_name]), dtype=np.float32))
                    # Stack: (batch_size, seq_len, feat_dim) -> (seq_len, batch_size, feat_dim)
                    feat_array = np.stack(feat_arrays, axis=1)  # (seq_len, batch_size, feat_dim)
                    batch_item_feats[feat_name] = torch.FloatTensor(feat_array).to(device)
            
            # 生成 causal mask
            src_mask = generate_square_subsequent_mask(seq_len).to(device)
            
            # 批量前向传播
            output = model(input_ids, 
                          user_ids=user_ids_tensor,  # 单独传递 user_ids
                          user_feats=batch_user_feats if batch_user_feats else None,
                          item_feats=batch_item_feats if batch_item_feats else None,
                          src_mask=src_mask)  # (seq_len, batch_size, vocab_size)
            
            # 取最后一个位置的输出
            logits = output[-1]  # (batch_size, vocab_size)
            
            # 批量获取 topk（取一个合理的上限，避免内存过大）
            # 考虑到需要过滤已点击的物品，取 topk * 3 应该足够
            max_topk = min(topk * 3, vocab_size_with_pad)
            scores, indices = torch.topk(logits, max_topk, dim=1)  # (batch_size, max_topk)
            scores = scores.cpu().numpy()
            indices = indices.cpu().numpy()
            
            # 批量解码物品ID并过滤已点击的物品
            for batch_idx, (user_id, clicked_items) in enumerate(zip(batch_user_ids, batch_clicked_items)):
                recall_items = []
                for score, idx in zip(scores[batch_idx], indices[batch_idx]):
                    # 跳过 padding token (0)
                    if idx > 0 and idx <= len(item_encoder.classes_):
                        try:
                            raw_item_id = item_encoder.inverse_transform([idx - 1])[0]  # -1 还原编码
                            if raw_item_id not in clicked_items:
                                recall_items.append((raw_item_id, float(score)))
                        except:
                            continue
                    if len(recall_items) >= topk:
                        break
                user_recall_dict[user_id] = recall_items
    
    return user_recall_dict

def metrics_recall(user_recall_items_dict, trn_last_click_df, topk=50):
    """评估召回效果"""
    last_click_item_dict = dict(zip(trn_last_click_df['user_id'], trn_last_click_df['click_article_id']))
    valid_users = set(user_recall_items_dict.keys()) & set(last_click_item_dict.keys())
    user_num = len(valid_users)
    
    if user_num == 0:
        print("警告: 召回字典和最后一次点击数据中没有共同的用户！")
        return
    
    for k in range(10, topk + 1, 10):
        hit_num = 0
        for user in valid_users:
            item_list = user_recall_items_dict[user]
            if len(item_list) > 0:
                if isinstance(item_list[0], (list, tuple)):
                    tmp_recall_items = [x[0] for x in item_list[:k]]
                else:
                    tmp_recall_items = item_list[:k]
            else:
                tmp_recall_items = []
            
            last_click_item = int(last_click_item_dict[user])
            tmp_recall_items = [int(x) for x in tmp_recall_items]
            
            if last_click_item in set(tmp_recall_items):
                hit_num += 1
        
        hit_rate = round(hit_num * 1.0 / user_num, 5) if user_num > 0 else 0.0
        print(f"k={k}: hit_num={hit_num}, hit_rate={hit_rate}, user_num={user_num}")

# ===================== 主程序 =====================
if __name__ == '__main__':
    print("=" * 80)
    print("基于 Transformer 的生成式召回模型")
    print("=" * 80)
    
    # 加载数据
    print("\n加载数据...")
    all_click_df = get_all_click_df(data_path, offline=False, sample_size=None)  # 先用小数据测试
    sample_users = np.random.choice(all_click_df['user_id'].unique(), size=20000, replace=False)
    all_click_df = all_click_df[all_click_df['user_id'].isin(sample_users)]
    item_info_df = get_item_info_df(data_path)
    item_emb_dict = get_item_emb_dict(data_path)
    
    # 获取历史点击和最后一次点击
    trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
    
    print(f"训练历史点击数据: {len(trn_hist_click_df)}")
    print(f"最后一次点击数据: {len(trn_last_click_df)}")
    
    # 准备用户特征字典（用于召回时）
    user_feats_dict_for_recall = {}
    user_feat_cols = ['click_os', 'click_deviceGroup', 'click_region']
    for user_id, group in all_click_df.groupby('user_id'):
        user_feats = {}
        for col in user_feat_cols:
            if col in group.columns:
                mode_val = group[col].mode()
                if len(mode_val) > 0:
                    user_feats[col] = mode_val.iloc[0]
        if user_feats:
            user_feats_dict_for_recall[user_id] = user_feats
    
    # 准备物品特征字典（用于召回时）
    item_feats_dict_for_recall = {}
    if 'words_count' in item_info_df.columns:
        scaler = MinMaxScaler()
        item_info_df['words_count_norm'] = scaler.fit_transform(
            item_info_df[['words_count']]
        ).flatten()
    
    for _, row in item_info_df.iterrows():
        item_id = row['click_article_id']
        item_feats = {}
        if 'words_count_norm' in item_info_df.columns:
            item_feats['words_count'] = float(row['words_count_norm'])
        if item_id in item_emb_dict:
            item_feats['emb'] = item_emb_dict[item_id]
        if item_feats:
            item_feats_dict_for_recall[item_id] = item_feats
    
    # 训练模型
    max_seq_len = 30
    model_path = save_path + 'transformer_recall_model.pth'
    model, item_encoder, user_encoder, user_feat_encoders, vocab_size_with_pad = train_transformer_recall(
        trn_hist_click_df,
        item_info_df,
        item_emb_dict,
        use_user_feats=True,
        use_item_feats=True,
        d_model=32,  # 进一步减小模型维度以节省内存
        nhead=4,  # 减小注意力头数（d_model 必须能被 nhead 整除）
        num_layers=4,  # 减少层数以节省内存
        dim_feedforward=16,  # 进一步减小 FFN 维度
        max_seq_len=max_seq_len,  # 进一步减小序列长度
        batch_size=10,  # 进一步减小batch_size
        epochs=10,
        lr=0.001,
        save_model_path=model_path
    )
    
    # 准备召回数据：每个用户的历史点击序列
    user_hist_dict = {}
    for user_id, group in trn_hist_click_df.groupby('user_id'):
        group = group.sort_values('click_timestamp')
        item_list = group['click_article_id'].tolist()
        if len(item_list) > 0:
            user_hist_dict[user_id] = item_list
    
    # 获取所有物品ID（用于召回）
    all_item_ids = set(item_info_df['click_article_id'].unique())
    all_item_ids.update(all_click_df['click_article_id'].unique())
    
    # 召回
    recall_dict = recall_transformer(
        model,
        user_hist_dict,
        item_encoder,
        user_encoder,
        user_feat_encoders,
        user_feats_dict_for_recall,
        item_feats_dict_for_recall,
        all_item_ids,
        vocab_size_with_pad,
        topk=50,
        max_seq_len=max_seq_len,  # 与训练时的 max_seq_len 保持一致
        use_user_feats=True,
        use_item_feats=True
    )
    
    # 评估
    print("\n召回效果评估:")
    metrics_recall(recall_dict, trn_last_click_df, topk=50)
    
    # 保存召回结果
    pickle.dump(recall_dict, open(save_path + 'transformer_recall_dict.pkl', 'wb'))
    print(f"\n召回结果已保存到: {save_path}transformer_recall_dict.pkl")


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
from datetime import datetime
import torch.nn.functional as F
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

# ===================== 早停机制 =====================
class EarlyStopping:
    """早停机制，用于防止过拟合"""
    def __init__(self, patience=3, min_delta=0.001, mode='max', verbose=True):
        """
        Args:
            patience: 容忍验证集指标不提升的epoch数
            min_delta: 认为指标提升的最小变化量
            mode: 'max' 表示指标越大越好，'min' 表示越小越好
            verbose: 是否打印信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score, epoch):
        """
        检查是否应该早停
        Returns:
            True: 应该早停
            False: 继续训练
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            if self.verbose:
                print(f"  早停: 初始化最佳指标 = {self.best_score:.5f}")
        elif self.mode == 'max':
            # 指标越大越好
            if score < self.best_score + self.min_delta:
                self.counter += 1
                if self.verbose:
                    print(f"  早停: 验证集指标未提升 (当前: {score:.5f}, 最佳: {self.best_score:.5f}, 等待: {self.counter}/{self.patience})")
                if self.counter >= self.patience:
                    self.early_stop = True
                    if self.verbose:
                        print(f"  早停: 验证集指标连续 {self.patience} 个epoch未提升，停止训练")
                        print(f"  最佳指标: {self.best_score:.5f} (Epoch {self.best_epoch+1})")
            else:
                # 指标提升了
                improvement = score - self.best_score
                self.best_score = score
                self.best_epoch = epoch
                self.counter = 0
                if self.verbose:
                    print(f"  早停: 验证集指标提升 +{improvement:.5f} (当前: {score:.5f}, 最佳: {self.best_score:.5f})")
        else:
            # 指标越小越好
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.verbose:
                    print(f"  早停: 验证集指标未下降 (当前: {score:.5f}, 最佳: {self.best_score:.5f}, 等待: {self.counter}/{self.patience})")
                if self.counter >= self.patience:
                    self.early_stop = True
                    if self.verbose:
                        print(f"  早停: 验证集指标连续 {self.patience} 个epoch未下降，停止训练")
                        print(f"  最佳指标: {self.best_score:.5f} (Epoch {self.best_epoch+1})")
            else:
                improvement = self.best_score - score
                self.best_score = score
                self.best_epoch = epoch
                self.counter = 0
                if self.verbose:
                    print(f"  早停: 验证集指标下降 -{improvement:.5f} (当前: {score:.5f}, 最佳: {self.best_score:.5f})")
        
        return self.early_stop

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
            activation=F.gelu,  # 使用 GELU 激活函数
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
    
    def forward(self, item_ids, user_ids=None, user_feats=None, item_feats=None, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            item_ids: (seq_len, batch_size) 物品 ID 序列（第一个位置可能是 user_id，需要单独处理）
            user_ids: (batch_size,) 用户 ID（可选，如果提供则第一个位置使用 user_embedding）
            user_feats: dict, 用户特征，如 {'click_os': (batch_size,), 'click_deviceGroup': (batch_size,), ...}
            item_feats: dict, 物品特征，如 {'words_count': (seq_len, batch_size, 1), 'emb': (seq_len, batch_size, 250)}
            src_mask: (seq_len, seq_len) 注意力掩码（causal mask）
            src_key_padding_mask: (batch_size, seq_len) padding 掩码，True 表示 padding 位置（需要被 mask）
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
        # src_mask: causal mask (seq_len, seq_len)
        # src_key_padding_mask: padding mask (batch_size, seq_len)，True 表示 padding 位置
        output = self.transformer_encoder(
            item_emb, 
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )  # (seq_len, batch_size, d_model)
        
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
        
        # 允许只有1次点击的用户参与训练
        if len(item_list) < 1:
            continue  # 至少需要1个物品才能生成训练样本
        
        # 截断序列
        if len(item_list) > max_seq_len:
            item_list = item_list[:max_seq_len]
        
        # 输入序列：[userID, item1, item2, ..., item_n-1]（第一个位置是 userID 占位符，后面是 items）
        # 目标序列：[item1, item2, ..., item_n]（每个位置预测下一个 token）
        # 注意：第一个位置（userID）预测 item1，第二个位置（item1）预测 item2，以此类推
        # 对于只有1次点击的用户：input_seq = [] (空)，target_seq = [item1]
        # 输入序列：[userID占位符]，目标序列：[item1, padding, ...]
        input_seq = item_list[:-1] if len(item_list) > 1 else []  # 去掉最后一个 item，作为输入（只有1次点击时为空）
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
    
    # 生成 padding mask: (batch_size, seq_len)，True 表示 padding 位置（需要被 mask）
    # 注意：第一个位置是用户占位符（虽然值是 pad_token，但有效），不应该被 mask
    padding_mask = []
    for input_seq in padded_inputs:
        # True 表示 padding 位置（需要被 mask），False 表示有效位置
        mask = [item_id == pad_token for item_id in input_seq]
        # 第一个位置是用户占位符，有效，不 mask
        mask[0] = False
        padding_mask.append(mask)
    padding_mask = torch.BoolTensor(padding_mask)  # (batch_size, seq_len)
    
    return {
        'input_ids': torch.LongTensor(padded_inputs),  # (batch_size, seq_len)
        'user_ids': torch.LongTensor(all_user_ids),  # (batch_size,) 单独存储 user_id
        'targets': torch.LongTensor(padded_targets),  # (batch_size, seq_len) 每个位置对应下一个 token
        'user_feats': all_user_feats,
        'item_feats': padded_item_feats_list,
        'padding_mask': padding_mask  # (batch_size, seq_len)，True 表示 padding 位置
    }

# ===================== 训练和召回函数 =====================
def prepare_data(all_click_df, item_info_df, item_emb_dict, 
                 use_user_feats=True, use_item_feats=True,
                 val_user_sequences=None, trn_click_df=None):
    """
    准备训练数据
    Args:
        all_click_df: 全量点击数据（train + test）
        item_info_df: 物品信息
        item_emb_dict: 物品embedding字典
        use_user_feats: 是否使用用户特征
        use_item_feats: 是否使用物品特征
        val_user_sequences: 验证集用户的完整序列字典 {user_id: full_item_list}（来自train）
                           对于验证集用户，训练时只使用部分序列（去掉最后一个item）
        trn_click_df: 训练集数据（用于确保验证集用户只使用train中的序列）
    """
    print("准备训练数据...")
    
    # 按用户分组，获取每个用户的点击序列（按时间排序）
    sequences = []
    val_user_set = set(val_user_sequences.keys()) if val_user_sequences else set()
    
    # 对于验证集用户，需要从 train 数据中获取序列（确保只使用 train 的数据）
    if val_user_sequences and trn_click_df is not None:
        print(f"  处理验证集用户（从train数据中获取，只使用部分序列）...")
        for user_id, full_item_list in tqdm(val_user_sequences.items(), desc="验证集用户"):
            # 验证集用户：训练时只使用部分序列（去掉最后一个item）
            if len(full_item_list) >= 2:
                train_item_list = full_item_list[:-1]  # 去掉最后一个
                sequences.append((user_id, train_item_list))
            elif len(full_item_list) == 1:
                # 只有一次点击的验证集用户也保留（虽然无法生成训练样本，但需要保留用户ID用于召回）
                sequences.append((user_id, full_item_list))
    
    # 处理其他用户（非验证集用户）
    print(f"  处理其他用户（使用全量数据的完整序列）...")
    for user_id, group in tqdm(all_click_df.groupby('user_id'), desc="其他用户"):
        # 跳过验证集用户（已经在上面处理过了）
        if user_id in val_user_set:
            continue
        
        group = group.sort_values('click_timestamp')
        item_list = group['click_article_id'].tolist()
        
        # 普通用户，使用完整序列
        # 注意：即使只有一次点击，也要保留（虽然无法生成训练样本，但需要保留用户ID用于召回）
        if len(item_list) >= 1:  # 至少需要1个物品（保留所有用户，包括只有一次点击的）
            sequences.append((user_id, item_list))
    
    print(f"生成了 {len(sequences)} 个用户序列")
    
    # 统计用户点击次数分布
    user_click_counts = {}
    for user_id, item_list in sequences:
        user_click_counts[user_id] = len(item_list)
    
    single_click_users = sum(1 for count in user_click_counts.values() if count == 1)
    multi_click_users = sum(1 for count in user_click_counts.values() if count >= 2)
    
    print(f"  用户点击次数分布:")
    print(f"    只有1次点击的用户: {single_click_users} 个（可以生成1个训练样本：user_id → item1）")
    print(f"    有2次及以上点击的用户: {multi_click_users} 个（可以生成多个训练样本）")
    
    if val_user_set:
        val_seq_count = len([s for s in sequences if s[0] in val_user_set])
        print(f"  其中验证集用户: {val_seq_count} 个（使用部分序列，来自train数据）")
        print(f"  其他用户: {len(sequences) - val_seq_count} 个（使用完整序列，来自全量数据）")
    
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
                             save_model_path=None, val_user_sequences=None, trn_click_df=None,
                             dropout=0.2, weight_decay=1e-4, 
                             early_stopping_patience=3, label_smoothing=0.0):
    """
    训练 Transformer 召回模型
    Args:
        all_click_df: 全量点击数据（train + test）
        val_user_sequences: 验证集用户的完整序列字典 {user_id: full_item_list}（来自train）
                           对于验证集用户，训练时只使用部分序列（去掉最后一个item）
        trn_click_df: 训练集数据（用于确保验证集用户只使用train中的序列）
    """
    # 获取全局 device 变量
    global device
    
    print("=" * 80)
    print("开始训练 Transformer 召回模型")
    print("=" * 80)
    
    # 准备数据
    sequences, user_feats_dict, item_feats_dict = prepare_data(
        all_click_df, item_info_df, item_emb_dict,
        use_user_feats=use_user_feats, use_item_feats=use_item_feats,
        val_user_sequences=val_user_sequences, trn_click_df=trn_click_df
    )
    
    # 编码用户和物品 ID
    print("编码用户和物品 ID...")
    all_user_ids = set()
    all_item_ids = set()
    for user_id, item_list in sequences:
        all_user_ids.add(user_id)  # 所有用户（包括只有一次点击的）都会被添加
        all_item_ids.update(item_list)
    
    # 重要：sequences 中已经包含了所有用户（包括只有一次点击的）
    # 所以 all_user_ids 已经包含了所有用户，不需要再从 all_click_df 中添加
    # 但为了确保完整性，我们还是检查一下
    all_users_in_data = set(all_click_df['user_id'].unique())
    missing_users = all_users_in_data - all_user_ids
    if len(missing_users) > 0:
        print(f"  [Warning] 发现 {len(missing_users)} 个用户不在 sequences 中，添加到 user_encoder")
        all_user_ids.update(missing_users)
    
    print(f"  用户ID统计:")
    print(f"    sequences 中的用户数: {len(sequences)}")
    print(f"    所有用户数（将保存到 user_encoder）: {len(all_user_ids)}")
    print(f"    只有1次点击的用户数: {sum(1 for _, items in sequences if len(items) == 1)}")
    print(f"    有2次及以上点击的用户数: {sum(1 for _, items in sequences if len(items) >= 2)}")
    
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
    
    # 用户编码器（包含所有用户，包括只有一次点击的）
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
        dropout=dropout,  # Dropout 率（增加以防止过拟合）
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
    # 添加权重衰减（L2正则化）以防止过拟合
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # 添加标签平滑（label smoothing）以提升泛化能力
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=label_smoothing)  # 忽略 padding
    
    # 添加学习率衰减（防止过拟合）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6
    )
    
    # 早停机制
    early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=0.001, mode='max', verbose=True)
    best_val_score = None
    best_model_state = None
    
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
            padding_mask = batch['padding_mask'].to(device)  # (batch_size, seq_len)，True 表示 padding 位置
            
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
            
            # 准备 padding mask: (batch_size, seq_len) -> 需要转置为 (seq_len, batch_size) 格式
            # 但 PyTorch 的 src_key_padding_mask 需要 (batch_size, seq_len) 格式
            src_key_padding_mask = padding_mask  # (batch_size, seq_len)，True 表示 padding 位置
            
            # 前向传播
            optimizer.zero_grad()
            output = model(input_ids, user_ids=user_ids,  # 单独传递 user_ids
                          user_feats=batch_user_feats if batch_user_feats else None,
                          item_feats=batch_item_feats if batch_item_feats else None,
                          src_mask=src_mask,  # causal mask
                          src_key_padding_mask=src_key_padding_mask)  # padding mask
            # output: (seq_len, batch_size, vocab_size)
            
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
            
            # CrossEntropyLoss 已经设置了 ignore_index=0，会自动忽略 padding 位置
            # 对所有非padding位置计算 loss（包括所有位置的 next-token prediction）
            # 这是标准的 Transformer 训练方式，让模型在每个位置都能预测下一个 token
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
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 更新学习率（基于 loss）
        scheduler.step(avg_loss)
        
        # 每个 epoch 结束后，评估验证集（如果设置了验证集）
        # 注意：为了早停，需要每个epoch都评估，但可以只打印部分epoch的详细结果
        should_evaluate = (val_user_sequences and len(val_user_sequences) > 0)
        should_print_detail = (epoch in [0, 2, 4, 6, 9] or epoch % 2 == 0)  # 每2个epoch打印一次详细结果
        
        if should_evaluate:
            if should_print_detail:
                print(f"\n评估验证集（Epoch {epoch+1}/{epochs}）...")
            else:
                print(f"\n评估验证集（Epoch {epoch+1}/{epochs}，仅早停检查）...")
            # 准备验证集的输入序列（使用完整序列，但输入时去掉最后一个item）
            val_user_hist_dict = {}
            val_last_click_dict = {}
            for user_id, full_item_list in val_user_sequences.items():
                # 输入序列：去掉最后一个item（用于预测）
                if len(full_item_list) >= 2:
                    input_item_list = full_item_list[:-1]
                    val_user_hist_dict[user_id] = input_item_list
                    val_last_click_dict[user_id] = full_item_list[-1]  # 答案：最后一个item
            
            if len(val_user_hist_dict) > 0:
                # 对验证集进行召回
                val_recall_dict = recall_transformer(
                    model,
                    val_user_hist_dict,
                    item_encoder,
                    user_encoder,
                    user_feat_encoders,
                    user_feats_dict,  # 使用训练时的用户特征字典
                    item_feats_dict,  # 使用训练时的物品特征字典
                    set(item_encoder.classes_),  # 所有物品ID
                    vocab_size_with_pad,
                    topk=50,
                    max_seq_len=max_seq_len,
                    use_user_feats=use_user_feats,
                    use_item_feats=use_item_feats,
                    batch_size=batch_size
                )
                
                # 调试信息：检查召回结果（仅在详细打印时显示）
                if should_print_detail:
                    total_recall_users = len(val_recall_dict)
                    users_with_recall = sum(1 for items in val_recall_dict.values() if len(items) > 0)
                    total_recall_items = sum(len(items) for items in val_recall_dict.values())
                    print(f"  召回统计: 总用户数={total_recall_users}, 有召回结果的用户数={users_with_recall}, 总召回物品数={total_recall_items}")
                
                # 计算验证集的 hit rate
                hit_num = 0
                valid_users = set(val_recall_dict.keys()) & set(val_last_click_dict.keys())
                users_with_empty_recall = 0
                for user_id in valid_users:
                    recall_items = val_recall_dict[user_id]
                    if len(recall_items) == 0:
                        users_with_empty_recall += 1
                        continue
                    # recall_items 是 [(item_id, score), ...] 格式
                    recall_item_ids = [item[0] if isinstance(item, (list, tuple)) else item for item in recall_items[:50]]
                    if val_last_click_dict[user_id] in recall_item_ids:
                        hit_num += 1
                
                hit_rate = round(hit_num * 1.0 / len(valid_users), 5) if len(valid_users) > 0 else 0.0
                if should_print_detail:
                    print(f"  验证集 Hit@50: {hit_num}/{len(valid_users)} = {hit_rate}")
                    if users_with_empty_recall > 0:
                        print(f"  警告: {users_with_empty_recall} 个用户的召回结果为空")
                else:
                    print(f"  验证集 Hit@50: {hit_rate:.5f}")
                
                # 早停检查：使用Hit@50作为验证指标
                if early_stopping(hit_rate, epoch):
                    print(f"\n早停触发！在 Epoch {epoch+1} 停止训练")
                    # 加载最佳模型
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                        print(f"已加载最佳模型（Epoch {early_stopping.best_epoch+1}, Hit@50={early_stopping.best_score:.5f}）")
                    break
                
                # 保存最佳模型
                if best_val_score is None or hit_rate > best_val_score:
                    best_val_score = hit_rate
                    best_model_state = model.state_dict().copy()
                    if should_print_detail:
                        print(f"  保存最佳模型（Hit@50={best_val_score:.5f}）")
                
                # 如果早停，跳出训练循环
                if early_stopping.early_stop:
                    break
                
                # 使用完整的评估指标（可选，用于更详细的评估）
                # 如果需要更详细的评估，可以取消下面的注释
                val_last_click_df = pd.DataFrame({
                    'user_id': list(val_last_click_dict.keys()),
                    'click_article_id': list(val_last_click_dict.values())
                })
                detailed_metrics = evaluate_transformer_recommendation(
                    val_recall_dict, 
                    val_last_click_df, 
                    k_list=[1, 5, 10, 20, 50]
                )
            print()  # 空行分隔
    
    # 保存模型和编码器（如果早停，保存最佳模型；否则保存当前模型）
    if save_model_path:
        # 如果早停，使用最佳模型；否则使用当前模型
        final_model_state = best_model_state if best_model_state is not None else model.state_dict()
        torch.save({
            'model_state_dict': final_model_state,
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
    skipped_users = {'no_user_in_encoder': 0, 'no_items': 0, 'item_not_in_encoder': 0}  # 统计跳过的用户
    
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
                except Exception as e:
                    # 如果用户不在编码器中，记录并跳过（这种情况不应该发生，因为我们已经将所有用户加入了编码器）
                    skipped_users['no_user_in_encoder'] += 1
                    if batch_start == 0:  # 只在第一个batch记录，避免输出过多
                        print(f"  [Warning] 用户 {user_id} 不在 user_encoder 中，跳过: {e}")
                    continue
                
                if len(item_list) == 0:
                    skipped_users['no_items'] += 1
                    if batch_start == 0:  # 只在第一个batch记录
                        print(f"  [Warning] 用户 {user_id} 没有历史点击，跳过")
                    continue
                
                # 编码物品序列（+1 为 padding 预留位置）
                try:
                    encoded_item_list = [x + 1 for x in item_encoder.transform(item_list).tolist()]
                except Exception as e:
                    # 如果物品不在训练集中，记录并跳过
                    skipped_users['item_not_in_encoder'] += 1
                    if batch_start == 0:  # 只在第一个batch记录
                        print(f"  [Warning] 用户 {user_id} 的物品不在 item_encoder 中，跳过: {e}")
                    continue
                
                # 构建输入序列：[item1, item2, ...]（不包含 user_id，user_id 单独传递）
                # 注意：与训练时保持一致，使用右侧padding
                input_seq = encoded_item_list[-max_seq_len+1:]  # 保留最后 max_seq_len-1 个物品
                
                # Padding（使用 0 作为 padding token）
                # 重要：与训练时保持一致，使用右侧padding（不是左侧padding）
                # 训练时格式：[0, item1, item2, ..., padding, padding]
                if len(input_seq) < max_seq_len - 1:  # -1 为 user_id 预留位置
                    input_seq = input_seq + [0] * (max_seq_len - 1 - len(input_seq))  # 右侧padding
                
                # 在前面加一个 0 作为 user_id 占位符（实际 user_id 单独传递）
                # 最终格式：[0, item1, item2, ..., padding, padding]，与训练时一致
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
            
            # 生成 padding mask: (batch_size, seq_len)，True 表示 padding 位置（需要被 mask）
            # 注意：第一个位置是用户占位符（虽然值是 0，但有效），不应该被 mask
            # 序列格式：[0(用户占位符), item1, item2, ..., 0(padding), 0(padding)]
            padding_mask = (input_ids == 0)  # (batch_size, seq_len)
            # 第一个位置是用户占位符，有效，不 mask
            padding_mask[:, 0] = False
            
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
            
            # 准备 padding mask: (batch_size, seq_len)，True 表示 padding 位置
            src_key_padding_mask = padding_mask  # (batch_size, seq_len)
            
            # 批量前向传播
            output = model(input_ids, 
                          user_ids=user_ids_tensor,  # 单独传递 user_ids
                          user_feats=batch_user_feats if batch_user_feats else None,
                          item_feats=batch_item_feats if batch_item_feats else None,
                          src_mask=src_mask,  # causal mask
                          src_key_padding_mask=src_key_padding_mask)  # padding mask
            # output: (seq_len, batch_size, vocab_size)
            
            # 找到每个序列的最后一个非0位置（最后一个真实item的位置）
            # input_ids: (seq_len, batch_size)，需要转回 (batch_size, seq_len) 来查找
            input_ids_batch_first = input_ids.transpose(0, 1)  # (batch_size, seq_len)
            # 找到每个序列中最后一个非0的位置（从后往前找）
            last_nonzero_positions = []
            for batch_idx in range(input_ids_batch_first.shape[0]):
                seq = input_ids_batch_first[batch_idx]  # (seq_len,)
                # 从后往前找第一个非0的位置（跳过padding）
                nonzero_indices = (seq != 0).nonzero(as_tuple=True)[0]
                if len(nonzero_indices) > 0:
                    last_nonzero_pos = nonzero_indices[-1].item()  # 最后一个非0的位置
                else:
                    last_nonzero_pos = seq_len - 1  # 如果没有非0，使用最后一个位置
                last_nonzero_positions.append(last_nonzero_pos)
            
            # 为每个样本取对应最后一个非0位置的输出
            # output: (seq_len, batch_size, vocab_size)
            batch_logits = []
            for batch_idx, pos in enumerate(last_nonzero_positions):
                batch_logits.append(output[pos, batch_idx, :])  # (vocab_size,)
            logits = torch.stack(batch_logits, dim=0)  # (batch_size, vocab_size)
            
            # 批量获取 topk（取一个合理的上限，避免内存过大）
            # 考虑到需要过滤已点击的物品，取 topk * 3 应该足够
            max_topk = min(topk * 3, vocab_size_with_pad)
            scores, indices = torch.topk(logits, max_topk, dim=1)  # (batch_size, max_topk)
            scores = scores.cpu().numpy()
            indices = indices.cpu().numpy()
            
            # 批量解码物品ID并过滤已点击的物品
            decode_fail_count = 0
            filtered_count = 0
            invalid_idx_count = 0
            for batch_idx, (user_id, clicked_items) in enumerate(zip(batch_user_ids, batch_clicked_items)):
                recall_items = []
                for score, idx in zip(scores[batch_idx], indices[batch_idx]):
                    # 跳过 padding token (0)
                    if idx == 0:
                        continue
                    if idx > 0 and idx <= len(item_encoder.classes_):
                        try:
                            raw_item_id = item_encoder.inverse_transform([idx - 1])[0]  # -1 还原编码
                            if raw_item_id not in clicked_items:
                                recall_items.append((raw_item_id, float(score)))
                            else:
                                filtered_count += 1
                        except Exception as e:
                            decode_fail_count += 1
                            continue
                    else:
                        invalid_idx_count += 1
                    if len(recall_items) >= topk:
                        break
                user_recall_dict[user_id] = recall_items
            
            # 每处理10个batch输出一次调试信息（仅第一个batch）
            if batch_start == 0:
                print(f"  调试信息（第一个batch）:")
                print(f"    vocab_size_with_pad={vocab_size_with_pad}, item_encoder.classes_长度={len(item_encoder.classes_)}")
                print(f"    无效索引数={invalid_idx_count}, 解码失败数={decode_fail_count}, 过滤物品数={filtered_count}")
                if len(batch_user_ids) > 0:
                    sample_user = batch_user_ids[0]
                    sample_recall = user_recall_dict.get(sample_user, [])
                    print(f"    示例用户 {sample_user}: 召回物品数={len(sample_recall)}")
                    if len(sample_recall) > 0:
                        print(f"      前3个召回物品: {sample_recall[:3]}")
    
    # 输出召回统计信息
    print(f"\n召回统计:")
    print(f"  总用户数: {total_users}")
    print(f"  成功召回用户数: {len(user_recall_dict)}")
    print(f"  跳过用户数: {sum(skipped_users.values())}")
    if sum(skipped_users.values()) > 0:
        print(f"    其中：")
        print(f"      用户不在编码器中: {skipped_users['no_user_in_encoder']}")
        print(f"      用户没有历史点击: {skipped_users['no_items']}")
        print(f"      物品不在编码器中: {skipped_users['item_not_in_encoder']}")
    
    return user_recall_dict

def metrics_recall(user_recall_items_dict, trn_last_click_df, topk=50):
    """评估召回效果（基础版本，只计算Hit@K）"""
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


def evaluate_transformer_recommendation(user_recall_items_dict, trn_last_click_df, k_list=[1, 5, 10, 20, 50]):
    """
    评估Transformer推荐效果，使用多个指标（Hit@K, NDCG@K, Precision@K, MRR）
    
    Args:
        user_recall_items_dict: {user_id: [(item_id, score), ...]} 或 {user_id: [item_id, ...]}
        trn_last_click_df: DataFrame，包含 'user_id' 和 'click_article_id' 列
        k_list: 要评估的K值列表
    
    Returns:
        dict: 包含各种评估指标的字典
    """
    last_click_item_dict = dict(zip(trn_last_click_df['user_id'], trn_last_click_df['click_article_id']))
    valid_users = set(user_recall_items_dict.keys()) & set(last_click_item_dict.keys())
    user_num = len(valid_users)
    
    if user_num == 0:
        print("警告: 召回字典和最后一次点击数据中没有共同的用户！")
        return {}
    
    # 初始化指标字典
    metrics = {
        'hit_rate': {k: [] for k in k_list},
        'ndcg': {k: [] for k in k_list},
        'precision': {k: [] for k in k_list},
        'mrr': []
    }
    
    # 对每个用户计算指标
    for user in valid_users:
        item_list = user_recall_items_dict[user]
        
        # 处理不同格式的召回结果
        if len(item_list) > 0:
            if isinstance(item_list[0], (list, tuple)):
                recall_item_ids = [int(item[0]) for item in item_list]
            else:
                recall_item_ids = [int(item) for item in item_list]
        else:
            recall_item_ids = []
        
        true_item = int(last_click_item_dict[user])
        
        # 计算每个K的指标
        for k in k_list:
            top_k_items = recall_item_ids[:k]
            
            # Hit@K: 是否命中
            hit = 1 if true_item in top_k_items else 0
            metrics['hit_rate'][k].append(hit)
            
            # Precision@K: 精确率
            precision = hit / k if k > 0 else 0.0
            metrics['precision'][k].append(precision)
            
            # NDCG@K: 归一化折损累积增益
            # DCG@k = Σ(rel_i / log2(i+1))，其中i是位置（从1开始）
            # 对于二元相关性（相关=1，不相关=0），DCG = 1 / log2(rank+1)
            if true_item in top_k_items:
                rank = top_k_items.index(true_item) + 1  # 位置从1开始
                dcg = 1.0 / math.log2(rank + 1)  # DCG = rel / log2(rank+1)
                # 理想情况：相关物品排第一，IDCG = 1 / log2(1+1) = 1 / log2(2) = 1
                idcg = 1.0 / math.log2(2)  # IDCG = 1 / log2(2) = 1.0
                ndcg = dcg / idcg if idcg > 0 else 0.0
            else:
                ndcg = 0.0
            metrics['ndcg'][k].append(ndcg)
        
        # MRR: 平均倒数排名（关注第一个相关物品的位置）
        if true_item in recall_item_ids:
            rank = recall_item_ids.index(true_item) + 1  # 位置从1开始
            mrr = 1.0 / rank
        else:
            mrr = 0.0
        metrics['mrr'].append(mrr)
    
    # 计算平均值
    results = {}
    for k in k_list:
        results[f'Hit@{k}'] = round(np.mean(metrics['hit_rate'][k]), 5) if metrics['hit_rate'][k] else 0.0
        results[f'NDCG@{k}'] = round(np.mean(metrics['ndcg'][k]), 5) if metrics['ndcg'][k] else 0.0
        results[f'Precision@{k}'] = round(np.mean(metrics['precision'][k]), 5) if metrics['precision'][k] else 0.0
    results['MRR'] = round(np.mean(metrics['mrr']), 5) if metrics['mrr'] else 0.0
    
    # 打印结果
    print("\n=== Transformer推荐效果评估 ===")
    print(f"评估用户数: {user_num}")
    print("\n指标结果:")
    for k in k_list:
        print(f"  Hit@{k}: {results[f'Hit@{k}']:.5f}")
        print(f"  NDCG@{k}: {results[f'NDCG@{k}']:.5f}")
        print(f"  Precision@{k}: {results[f'Precision@{k}']:.5f}")
    print(f"  MRR: {results['MRR']:.5f}")
    print("=" * 40)
    
    return results

def recall_dict_2_df(recall_list_dict):
    """
    将召回字典转换为 DataFrame
    Args:
        recall_list_dict: {user_id: [(item_id, score), ...]} 或 {user_id: {item_id: score, ...}}
    Returns:
        DataFrame: columns=['user_id', 'click_article_id', 'pred_score']
    """
    df_row_list = []  # [user, item, score]
    for user, recall_list in tqdm(recall_list_dict.items(), desc="转换召回字典为DataFrame"):
        # 兼容两种结构：
        # 1) list[ (item, score), ... ]
        # 2) dict{ item: score }
        if isinstance(recall_list, dict):
            iterable = recall_list.items()
        else:
            iterable = recall_list

        for it in iterable:
            if isinstance(it, (list, tuple)) and len(it) == 2:
                item, score = it
            else:
                item, score = it, 1.0
            df_row_list.append([int(user), int(item), float(score)])

    col_names = ['user_id', 'click_article_id', 'pred_score']
    recall_list_df = pd.DataFrame(df_row_list, columns=col_names)

    return recall_list_df

def submit(recall_df, topk=5, model_name=None):
    """
    生成提交文件
    Args:
        recall_df: DataFrame, columns=['user_id', 'click_article_id', 'pred_score']
        topk: 每个用户提交的文章数量，默认5
        model_name: 模型名称，用于生成文件名
    """
    recall_df = recall_df.sort_values(by=['user_id', 'pred_score'], ascending=[True, False])
    recall_df['rank'] = recall_df.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')

    # 检查每个用户是否都有足够的文章
    tmp = recall_df.groupby('user_id').apply(lambda x: x['rank'].max())
    if tmp.min() < topk:
        print(f"警告: 部分用户的召回文章数不足 {topk} 个，最少: {tmp.min()}")
        # 对于不足的用户，使用热门文章补全（这里简单处理，实际可以更复杂）
        print("  将使用已有文章补全（可能重复）")

    del recall_df['pred_score']
    submit_df = recall_df[recall_df['rank'] <= topk].set_index(['user_id', 'rank']).unstack(-1).reset_index()

    submit_df.columns = [int(col) if isinstance(col, int) else col for col in submit_df.columns.droplevel(0)]
    # 按照提交格式定义列名
    submit_df = submit_df.rename(columns={'': 'user_id', 1: 'article_1', 2: 'article_2',
                                    3: 'article_3', 4: 'article_4', 5: 'article_5'})

    save_name = save_path + model_name + '_' + datetime.today().strftime('%m-%d') + '.csv'
    submit_df.to_csv(save_name, index=False, header=True)
    print(f"提交文件已保存到: {save_name}")
    print(f"提交文件格式: {submit_df.shape}, columns: {submit_df.columns.tolist()}")

# ===================== 主程序 =====================
if __name__ == '__main__':
    print("=" * 80)
    print("基于 Transformer 的生成式召回模型（全量数据训练，只预测test用户）")
    print("=" * 80)
    
    # ========== 采样开关 ==========
    USE_SAMPLE = False  # 设置为 True 时使用采样数据，False 时使用全量数据
    SAMPLE_USER_NUMS = 10000  # 采样用户数量
    
    if USE_SAMPLE:
        print(f"\n⚠️  采样模式已开启: 将采样 {SAMPLE_USER_NUMS} 个用户进行测试")
    else:
        print(f"\n全量数据模式: 使用所有用户数据")
    
    # 加载数据
    print("\n1. 加载数据...")
    # 加载 train 和 test 数据
    trn_click = pd.read_csv(data_path + 'train_click_log.csv')
    tst_click = pd.read_csv(data_path + 'testA_click_log.csv')
    
    print(f"原始训练集: {len(trn_click)} 条记录, {trn_click['user_id'].nunique()} 个用户")
    print(f"原始测试集: {len(tst_click)} 条记录, {tst_click['user_id'].nunique()} 个用户")
    
    # 如果开启采样，从训练集中采样用户
    if USE_SAMPLE:
        trn_user_ids = trn_click['user_id'].unique()
        if len(trn_user_ids) > SAMPLE_USER_NUMS:
            sample_trn_user_ids = np.random.choice(trn_user_ids, size=SAMPLE_USER_NUMS, replace=False)
            trn_click = trn_click[trn_click['user_id'].isin(sample_trn_user_ids)]
            print(f"采样后训练集: {len(trn_click)} 条记录, {trn_click['user_id'].nunique()} 个用户")
        
        # 测试集也采样（保持一定比例）
        tst_user_ids = tst_click['user_id'].unique()
        sample_tst_size = min(SAMPLE_USER_NUMS // 4, len(tst_user_ids))  # 测试集采样约为训练集的1/4
        if len(tst_user_ids) > sample_tst_size:
            sample_tst_user_ids = np.random.choice(tst_user_ids, size=sample_tst_size, replace=False)
            tst_click = tst_click[tst_click['user_id'].isin(sample_tst_user_ids)]
            print(f"采样后测试集: {len(tst_click)} 条记录, {tst_click['user_id'].nunique()} 个用户")
    
    print(f"训练集: {len(trn_click)} 条记录, {trn_click['user_id'].nunique()} 个用户")
    print(f"测试集: {len(tst_click)} 条记录, {tst_click['user_id'].nunique()} 个用户")
    
    # 获取 test 用户集合
    test_user_ids = set(tst_click['user_id'].unique())
    print(f"测试集用户数: {len(test_user_ids)}")
    
    # 合并 train 和 test 数据用于训练（全量数据）
    all_click_df = pd.concat([trn_click, tst_click], ignore_index=True)
    all_click_df = all_click_df.drop_duplicates(['user_id', 'click_article_id', 'click_timestamp'])
    print(f"全量数据: {len(all_click_df)} 条记录, {all_click_df['user_id'].nunique()} 个用户")
    
    # 加载物品信息和embedding
    item_info_df = get_item_info_df(data_path)
    item_emb_dict = get_item_emb_dict(data_path)
    
    # 2. 从 train 数据中采样一部分作为验证集（用于召回和排序校验）
    print("\n2. 准备验证集...")
    # 根据是否采样调整验证集大小
    if USE_SAMPLE:
        val_sample_user_nums = min(1000, len(trn_click['user_id'].unique()) // 10)  # 采样模式：验证集约为训练集的10%
    else:
        val_sample_user_nums = 10000  # 全量模式：验证集10000个用户
    trn_user_ids = set(trn_click['user_id'].unique())
    # 从训练集中采样用户作为验证集
    sample_val_user_ids = np.random.choice(
        list(trn_user_ids), 
        size=min(val_sample_user_nums, len(trn_user_ids)), 
        replace=False
    )
    sample_val_user_ids = set(sample_val_user_ids)
    print(f"验证集用户数: {len(sample_val_user_ids)}")
    
    # 获取验证集用户的完整序列（用于测试时）
    val_user_sequences = {}  # {user_id: full_item_list}
    for user_id in sample_val_user_ids:
        user_data = trn_click[trn_click['user_id'] == user_id].sort_values('click_timestamp')
        item_list = user_data['click_article_id'].tolist()
        # 注意：即使只有一次点击，也要保留（虽然无法生成训练样本，但需要保留用户ID用于召回）
        if len(item_list) >= 1:  # 至少需要1个物品（保留所有用户，包括只有一次点击的）
            val_user_sequences[user_id] = item_list
    
    single_click_val_users = sum(1 for items in val_user_sequences.values() if len(items) == 1)
    multi_click_val_users = sum(1 for items in val_user_sequences.values() if len(items) >= 2)
    print(f"验证集用户数: {len(val_user_sequences)}")
    print(f"  其中只有1次点击的用户: {single_click_val_users} 个")
    print(f"  有2次及以上点击的用户: {multi_click_val_users} 个")
    
    # 3. 准备用户特征字典（用于召回时）
    print("\n3. 准备用户特征...")
    user_feats_dict_for_recall = {}
    user_feat_cols = ['click_os', 'click_deviceGroup', 'click_region']
    for user_id, group in tqdm(all_click_df.groupby('user_id'), desc="处理用户特征"):
        user_feats = {}
        for col in user_feat_cols:
            if col in group.columns:
                mode_val = group[col].mode()
                if len(mode_val) > 0:
                    user_feats[col] = mode_val.iloc[0]
        if user_feats:
            user_feats_dict_for_recall[user_id] = user_feats
    
    # 4. 准备物品特征字典（用于召回时）
    print("\n4. 准备物品特征...")
    item_feats_dict_for_recall = {}
    if 'words_count' in item_info_df.columns:
        scaler = MinMaxScaler()
        item_info_df['words_count_norm'] = scaler.fit_transform(
            item_info_df[['words_count']]
        ).flatten()
    
    for _, row in tqdm(item_info_df.iterrows(), desc="处理物品特征", total=len(item_info_df)):
        item_id = row['click_article_id']
        item_feats = {}
        if 'words_count_norm' in item_info_df.columns:
            item_feats['words_count'] = float(row['words_count_norm'])
        if item_id in item_emb_dict:
            item_feats['emb'] = item_emb_dict[item_id]
        if item_feats:
            item_feats_dict_for_recall[item_id] = item_feats
    
    # 5. 训练模型（使用全量数据，验证集用户只使用部分序列）
    print("\n5. 训练模型...")
    print("   说明：")
    print("   - 验证集用户：只使用 train 数据中的部分序列（去掉最后一个item）进行训练")
    print("   - 其他用户：使用全量数据（train + test）的完整序列进行训练")
    max_seq_len = 80
    
    # 根据是否采样调整训练参数
    if USE_SAMPLE:
        # 采样模式：使用较小的参数加快训练
        d_model = 32
        nhead = 4
        num_layers = 3
        dim_feedforward = 64  # 修正：应该是 d_model 的 2 倍（32 * 2 = 64）
        batch_size = 32
        epochs = 5  # 采样模式减少epoch数
        print(f"   采样模式: epochs={epochs}, batch_size={batch_size}")
    else:
        # 全量模式：使用原始参数
        d_model = 32
        nhead = 4
        num_layers = 3
        dim_feedforward = 64  # 修正：应该是 d_model 的 2 倍（64 * 2 = 128），避免信息瓶颈
        batch_size = 256
        epochs = 10
        print(f"   全量模式: epochs={epochs}, batch_size={batch_size}")
    
    model_path = save_path + 'transformer_recall_model.pth'
    if USE_SAMPLE:
        model_path = save_path + 'transformer_recall_model_sample.pth'
    
    # 防止过拟合的参数（增强版）
    dropout = 0.3  # 增加 dropout 率（从 0.2 增加到 0.3，进一步防止过拟合）
    weight_decay = 5e-4  # L2 正则化（权重衰减，从 1e-4 增加到 5e-4）
    label_smoothing = 0.1  # 标签平滑（提升泛化能力）
    early_stopping_patience = 3  # 早停耐心值（连续3个epoch不提升则停止）
    
    model, item_encoder, user_encoder, user_feat_encoders, vocab_size_with_pad = train_transformer_recall(
        all_click_df,  # 使用全量数据（train + test）
        item_info_df,
        item_emb_dict,
        use_user_feats=True,
        use_item_feats=True,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        epochs=epochs,
        lr=0.001,
        save_model_path=model_path,
        val_user_sequences=val_user_sequences,  # 验证集用户的完整序列（来自train）
        trn_click_df=trn_click,  # 传递train数据，确保验证集用户只使用train中的序列
        dropout=dropout,  # Dropout 率
        weight_decay=weight_decay,  # 权重衰减
        early_stopping_patience=early_stopping_patience,  # 早停耐心值
        label_smoothing=label_smoothing  # 标签平滑
    )
    
    # 6. 准备召回数据：只预测 test 用户
    print("\n6. 准备召回数据（只预测test用户）...")
    test_user_hist_dict = {}  # test 用户的历史点击序列
    for user_id, group in tst_click.groupby('user_id'):
        group = group.sort_values('click_timestamp')
        item_list = group['click_article_id'].tolist()
        # 注意：test 用户可能只有一次点击，不能过滤
        if len(item_list) > 0:
            test_user_hist_dict[user_id] = item_list
    
    print(f"测试集用户数（用于召回）: {len(test_user_hist_dict)}")
    print(f"  其中只有1次点击的用户: {sum(1 for items in test_user_hist_dict.values() if len(items) == 1)}")
    
    # 获取所有物品ID（用于召回）
    all_item_ids = set(item_info_df['click_article_id'].unique())
    all_item_ids.update(all_click_df['click_article_id'].unique())
    
    # 7. 召回（只预测 test 用户）
    print("\n7. 开始召回（只预测test用户）...")
    recall_dict = recall_transformer(
        model,
        test_user_hist_dict,  # 只预测 test 用户
        item_encoder,
        user_encoder,
        user_feat_encoders,
        user_feats_dict_for_recall,
        item_feats_dict_for_recall,
        all_item_ids,
        vocab_size_with_pad,
        topk=50,
        max_seq_len=max_seq_len,
        use_user_feats=True,
        use_item_feats=True,
        batch_size=32
    )
    
    print(f"召回完成，共召回 {len(recall_dict)} 个用户")
    
    # 8. 保存召回结果（参考 news3.py 的存储方式）
    print("\n8. 保存召回结果...")
    recall_save_path = save_path + 'transformer_recall_dict_test.pkl'
    if USE_SAMPLE:
        recall_save_path = save_path + 'transformer_recall_dict_test_sample.pkl'
    pickle.dump(recall_dict, open(recall_save_path, 'wb'))
    print(f"召回结果已保存到: {recall_save_path}")
    
    # 9. 生成提交文件（参考 news5.py 的方式）
    print("\n9. 生成提交文件...")
    # 将召回字典转换为 DataFrame
    recall_df = recall_dict_2_df(recall_dict)
    print(f"召回 DataFrame 形状: {recall_df.shape}")
    print(f"召回用户数: {recall_df['user_id'].nunique()}")
    print(f"召回文章数: {recall_df['click_article_id'].nunique()}")
    
    # 生成提交文件
    model_name = 'transformer_recall'
    if USE_SAMPLE:
        model_name = 'transformer_recall_sample'
    submit(recall_df, topk=5, model_name=model_name)
    
    # 注意：验证集评估已在训练过程中完成（每个 epoch 结束后），无需重复评估
    
    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)
    if USE_SAMPLE:
        print(f"⚠️  本次运行使用采样模式（{SAMPLE_USER_NUMS} 个用户）")
    print(f"召回结果文件: {recall_save_path}")
    print(f"提交文件: {save_path}{model_name}_MM-DD.csv")


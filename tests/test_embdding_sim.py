import pandas as pd
import numpy as np
import unittest
import os
import tempfile
import shutil
import pickle
from collections import defaultdict
import faiss
from tqdm import tqdm

# 直接复制函数代码，避免导入整个模块时加载数据
def embdding_sim(click_df, item_emb_df, save_path, topk, batch_size=1000):
    """
        基于内容的文章embedding相似性矩阵计算（优化版本：分批处理避免OOM）
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
    
    # 分批处理，避免一次性计算所有相似度导致OOM
    n_items = len(item_emb_np)
    item_sim_dict = defaultdict(dict)
    
    # 检查是否已有部分结果（断点续传）
    checkpoint_file = save_path + 'emb_i2i_sim_checkpoint.pkl'
    start_idx = 0
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            item_sim_dict = checkpoint_data.get('item_sim_dict', defaultdict(dict))
            start_idx = checkpoint_data.get('last_idx', 0)
        except:
            start_idx = 0
    
    # 分批处理
    for batch_start in tqdm(range(start_idx, n_items, batch_size), desc="处理批次", leave=False):
        batch_end = min(batch_start + batch_size, n_items)
        batch_emb = item_emb_np[batch_start:batch_end]
        
        # 对当前批次进行相似度查询
        sim, idx = item_index.search(batch_emb, topk)
        
        # 处理当前批次的结果
        for local_idx, (sim_value_list, rele_idx_list) in enumerate(zip(sim, idx)):
            target_idx = batch_start + local_idx
            target_raw_id = item_idx_2_rawid_dict[target_idx]
            # 从1开始是为了去掉商品本身, 所以最终获得的相似商品只有topk-1
            for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
                rele_raw_id = item_idx_2_rawid_dict[rele_idx]
                item_sim_dict[target_raw_id][rele_raw_id] = item_sim_dict.get(target_raw_id, {}).get(rele_raw_id, 0) + sim_value
        
        # 定期保存检查点（每处理10个批次保存一次）
        if (batch_start // batch_size) % 10 == 0:
            checkpoint_data = {
                'item_sim_dict': item_sim_dict,
                'last_idx': batch_end
            }
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            # 释放内存
            del sim, idx, batch_emb
            import gc
            gc.collect()

    # 保存最终结果
    with open(save_path + 'emb_i2i_sim.pkl', 'wb') as f:
        pickle.dump(item_sim_dict, f)
    
    # 删除检查点文件
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    return item_sim_dict


def u2u_embdding_sim(click_df, user_emb_dict, save_path, topk, batch_size=1000):
    """
        基于用户embedding的相似性矩阵计算（优化版本：分批处理避免OOM）
    """
    user_list = []
    user_emb_list = []
    for user_id, user_emb in user_emb_dict.items():
        user_list.append(user_id)
        user_emb_list.append(user_emb)

    user_index_2_rawid_dict = {k: v for k, v in zip(range(len(user_list)), user_list)}

    user_emb_np = np.array(user_emb_list, dtype=np.float32)
    # 向量进行单位化
    user_emb_np = user_emb_np / np.linalg.norm(user_emb_np, axis=1, keepdims=True)

    # 建立faiss索引
    user_index = faiss.IndexFlatIP(user_emb_np.shape[1])
    user_index.add(user_emb_np)
    
    # 分批处理，避免一次性计算所有相似度导致OOM
    n_users = len(user_emb_np)
    user_sim_dict = defaultdict(dict)
    
    # 检查是否已有部分结果（断点续传）
    checkpoint_file = save_path + 'u2u_emb_sim_checkpoint.pkl'
    start_idx = 0
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            user_sim_dict = checkpoint_data.get('user_sim_dict', defaultdict(dict))
            start_idx = checkpoint_data.get('last_idx', 0)
        except:
            start_idx = 0
    
    # 分批处理
    for batch_start in tqdm(range(start_idx, n_users, batch_size), desc="处理用户批次", leave=False):
        batch_end = min(batch_start + batch_size, n_users)
        batch_emb = user_emb_np[batch_start:batch_end]
        
        # 对当前批次进行相似度查询
        sim, idx = user_index.search(batch_emb, topk)
        
        # 处理当前批次的结果
        for local_idx, (sim_value_list, rele_idx_list) in enumerate(zip(sim, idx)):
            target_idx = batch_start + local_idx
            target_raw_id = user_index_2_rawid_dict[target_idx]
            # 从1开始是为了去掉用户本身, 所以最终获得的相似用户只有topk-1
            try:
                for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
                    rele_raw_id = user_index_2_rawid_dict[rele_idx]
                    user_sim_dict[target_raw_id][rele_raw_id] = user_sim_dict.get(target_raw_id, {}).get(rele_raw_id, 0) + sim_value
            except KeyError as e:
                pass
        
        # 定期保存检查点（每处理10个批次保存一次）
        if (batch_start // batch_size) % 10 == 0:
            checkpoint_data = {
                'user_sim_dict': user_sim_dict,
                'last_idx': batch_end
            }
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            # 释放内存
            del sim, idx, batch_emb
            import gc
            gc.collect()

    # 保存最终结果
    with open(save_path + 'youtube_u2u_sim.pkl', 'wb') as f:
        pickle.dump(user_sim_dict, f)
    
    # 删除检查点文件
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    return user_sim_dict


class TestEmbddingSim(unittest.TestCase):
    """测试 embdding_sim 函数的优化版本"""
    
    def setUp(self):
        """创建测试数据和临时目录"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.save_path = self.temp_dir + '/'
        
        # 创建测试用的 item embedding DataFrame（使用小数据集）
        np.random.seed(42)
        n_items = 30  # 使用很小的数量便于快速测试
        emb_dim = 8
        
        # 生成随机 embedding
        embeddings = np.random.rand(n_items, emb_dim).astype(np.float32)
        # 归一化
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # 创建 DataFrame
        emb_cols = [f'emb_{i}' for i in range(emb_dim)]
        self.item_emb_df = pd.DataFrame(embeddings, columns=emb_cols)
        self.item_emb_df['article_id'] = range(100, 100 + n_items)  # article_id 从 100 开始
        
        # 创建测试用的 click_df（虽然函数中没用到，但为了接口一致）
        self.click_df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'click_article_id': [100, 101, 102],
            'click_timestamp': [1, 2, 3]
        })
    
    def tearDown(self):
        """清理临时文件"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_basic_functionality(self):
        """测试基本功能：能否正确计算相似度矩阵"""
        topk = 10
        batch_size = 20  # 使用较小的 batch_size 便于测试
        
        result = embdding_sim(
            self.click_df, 
            self.item_emb_df, 
            self.save_path, 
            topk=topk,
            batch_size=batch_size
        )
        
        # 验证返回结果类型
        self.assertIsInstance(result, dict)
        
        # 验证每个 item 都有相似度结果
        self.assertEqual(len(result), len(self.item_emb_df))
        
        # 验证每个 item 的相似度结果数量（应该是 topk-1，因为去掉了自己）
        for item_id, sim_items in result.items():
            self.assertLessEqual(len(sim_items), topk - 1)
            # 验证相似度值在合理范围内（内积相似度应该在 [-1, 1] 之间，但归一化后应该在 [0, 1] 之间）
            for sim_item_id, sim_value in sim_items.items():
                self.assertGreaterEqual(sim_value, 0)
                self.assertLessEqual(sim_value, 1)
    
    def test_batch_processing(self):
        """测试分批处理功能"""
        topk = 5
        batch_size = 15  # 使用较小的 batch_size，确保会分批处理
        
        result = embdding_sim(
            self.click_df,
            self.item_emb_df,
            self.save_path,
            topk=topk,
            batch_size=batch_size
        )
        
        # 验证结果完整性
        self.assertEqual(len(result), len(self.item_emb_df))
        
        # 验证所有 item 都被处理了
        expected_item_ids = set(self.item_emb_df['article_id'].values)
        actual_item_ids = set(result.keys())
        self.assertEqual(expected_item_ids, actual_item_ids)
    
    def test_checkpoint_resume(self):
        """测试断点续传功能"""
        topk = 5
        batch_size = 10  # 使用较小的 batch_size，确保会分批处理
        
        # 第一次运行，处理一部分后手动创建检查点
        checkpoint_file = self.save_path + 'emb_i2i_sim_checkpoint.pkl'
        
        # 创建部分结果（模拟已经处理了前 10 个 item）
        partial_result = defaultdict(dict)
        # 为前 10 个 item 创建一些相似度结果
        for i in range(10):
            item_id = 100 + i
            for j in range(1, topk):
                if 100 + i + j < 100 + len(self.item_emb_df):
                    partial_result[item_id][100 + i + j] = 0.9 - j * 0.1
        
        # 保存检查点（假设处理到第 10 个 item）
        checkpoint_data = {
            'item_sim_dict': partial_result,
            'last_idx': 10  # 假设处理到第 10 个 item（索引从 0 开始）
        }
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        # 第二次运行，应该从检查点恢复并继续处理剩余数据
        result = embdding_sim(
            self.click_df,
            self.item_emb_df,
            self.save_path,
            topk=topk,
            batch_size=batch_size
        )
        
        # 验证检查点文件已被删除
        self.assertFalse(os.path.exists(checkpoint_file))
        
        # 验证结果完整性（应该包含所有 item）
        self.assertEqual(len(result), len(self.item_emb_df))
        
        # 验证前 10 个 item 的结果被保留了（可能被更新，但应该存在）
        for i in range(10):
            item_id = 100 + i
            self.assertIn(item_id, result)
    
    def test_output_file_creation(self):
        """测试输出文件是否正确创建"""
        topk = 5
        batch_size = 20
        
        result = embdding_sim(
            self.click_df,
            self.item_emb_df,
            self.save_path,
            topk=topk,
            batch_size=batch_size
        )
        
        # 验证输出文件是否存在
        output_file = self.save_path + 'emb_i2i_sim.pkl'
        self.assertTrue(os.path.exists(output_file))
        
        # 验证可以加载文件
        with open(output_file, 'rb') as f:
            loaded_result = pickle.load(f)
        self.assertEqual(len(loaded_result), len(result))
    
    def test_small_batch_size(self):
        """测试小 batch_size 的情况（模拟内存受限场景）"""
        topk = 5
        batch_size = 5  # 非常小的 batch_size
        
        result = embdding_sim(
            self.click_df,
            self.item_emb_df,
            self.save_path,
            topk=topk,
            batch_size=batch_size
        )
        
        # 验证结果完整性
        self.assertEqual(len(result), len(self.item_emb_df))


class TestU2UEmbddingSim(unittest.TestCase):
    """测试 u2u_embdding_sim 函数的优化版本"""
    
    def setUp(self):
        """创建测试数据和临时目录"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.save_path = self.temp_dir + '/'
        
        # 创建测试用的 user embedding 字典（使用小数据集）
        np.random.seed(42)
        n_users = 20  # 使用很小的数量便于快速测试
        emb_dim = 8
        
        self.user_emb_dict = {}
        for user_id in range(1, n_users + 1):
            emb = np.random.rand(emb_dim).astype(np.float32)
            # 归一化
            emb = emb / np.linalg.norm(emb)
            self.user_emb_dict[user_id] = emb
        
        # 创建测试用的 click_df
        self.click_df = pd.DataFrame({
            'user_id': list(range(1, n_users + 1)),
            'click_article_id': list(range(100, 100 + n_users)),
            'click_timestamp': list(range(n_users))
        })
    
    def tearDown(self):
        """清理临时文件"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_basic_functionality(self):
        """测试基本功能：能否正确计算用户相似度矩阵"""
        topk = 10
        batch_size = 10
        
        result = u2u_embdding_sim(
            self.click_df,
            self.user_emb_dict,
            self.save_path,
            topk=topk,
            batch_size=batch_size
        )
        
        # 验证返回结果类型
        self.assertIsInstance(result, dict)
        
        # 验证每个用户都有相似度结果
        self.assertEqual(len(result), len(self.user_emb_dict))
        
        # 验证每个用户的相似度结果数量（应该是 topk-1，因为去掉了自己）
        for user_id, sim_users in result.items():
            self.assertLessEqual(len(sim_users), topk - 1)
            # 验证相似度值在合理范围内
            for sim_user_id, sim_value in sim_users.items():
                self.assertGreaterEqual(sim_value, 0)
                self.assertLessEqual(sim_value, 1)
    
    def test_batch_processing(self):
        """测试分批处理功能"""
        topk = 5
        batch_size = 10  # 使用较小的 batch_size，确保会分批处理
        
        result = u2u_embdding_sim(
            self.click_df,
            self.user_emb_dict,
            self.save_path,
            topk=topk,
            batch_size=batch_size
        )
        
        # 验证结果完整性
        self.assertEqual(len(result), len(self.user_emb_dict))
        
        # 验证所有用户都被处理了
        expected_user_ids = set(self.user_emb_dict.keys())
        actual_user_ids = set(result.keys())
        self.assertEqual(expected_user_ids, actual_user_ids)
    
    def test_checkpoint_resume(self):
        """测试断点续传功能"""
        topk = 5
        batch_size = 5  # 使用较小的 batch_size，确保会分批处理
        
        # 创建检查点
        checkpoint_file = self.save_path + 'u2u_emb_sim_checkpoint.pkl'
        
        # 创建部分结果（模拟已经处理了前 5 个用户）
        partial_result = defaultdict(dict)
        # 为前 5 个用户创建一些相似度结果
        for i in range(1, 6):
            for j in range(1, topk):
                if i + j <= len(self.user_emb_dict):
                    partial_result[i][i + j] = 0.9 - j * 0.1
        
        # 保存检查点（假设处理到第 5 个用户）
        checkpoint_data = {
            'user_sim_dict': partial_result,
            'last_idx': 5  # 假设处理到第 5 个用户（索引从 0 开始）
        }
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        # 第二次运行，应该从检查点恢复并继续处理剩余数据
        result = u2u_embdding_sim(
            self.click_df,
            self.user_emb_dict,
            self.save_path,
            topk=topk,
            batch_size=batch_size
        )
        
        # 验证检查点文件已被删除
        self.assertFalse(os.path.exists(checkpoint_file))
        
        # 验证结果完整性（应该包含所有用户）
        self.assertEqual(len(result), len(self.user_emb_dict))
        
        # 验证前 5 个用户的结果被保留了（可能被更新，但应该存在）
        for i in range(1, 6):
            self.assertIn(i, result)
    
    def test_output_file_creation(self):
        """测试输出文件是否正确创建"""
        topk = 5
        batch_size = 10
        
        result = u2u_embdding_sim(
            self.click_df,
            self.user_emb_dict,
            self.save_path,
            topk=topk,
            batch_size=batch_size
        )
        
        # 验证输出文件是否存在
        output_file = self.save_path + 'youtube_u2u_sim.pkl'
        self.assertTrue(os.path.exists(output_file))
        
        # 验证可以加载文件
        with open(output_file, 'rb') as f:
            loaded_result = pickle.load(f)
        self.assertEqual(len(loaded_result), len(result))


if __name__ == '__main__':
    unittest.main()


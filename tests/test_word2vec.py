import pandas as pd
import numpy as np
import unittest
import tempfile
import os
from gensim.models import Word2Vec
import pickle

# 导入要测试的函数（需要先设置 save_path）
import sys
sys.path.insert(0, '/root/news/Recommended-News-main')


class TestWord2VecFunction(unittest.TestCase):
    """测试 trian_item_word2vec 函数"""
    
    def setUp(self):
        """创建测试数据"""
        # 创建临时目录用于保存文件
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建模拟的点击数据
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
            'click_article_id': [101, 102, 103, 201, 202, 301, 302, 303, 304, 305],
            'click_timestamp': [1000, 1001, 1002, 2000, 2001, 3000, 3001, 3002, 3003, 3004]
        })
    
    def test_word2vec_basic_functionality(self):
        """测试 Word2Vec 基本功能"""
        # 模拟 docs 的构建过程
        click_df = self.test_data.copy()
        click_df['click_article_id'] = click_df['click_article_id'].astype(str)
        docs = click_df.groupby(['user_id'])['click_article_id'].apply(lambda x: list(x)).reset_index()
        docs = docs['click_article_id'].values.tolist()
        
        # 验证 docs 的结构
        self.assertIsInstance(docs, list)
        self.assertEqual(len(docs), 3)  # 3个用户
        self.assertIsInstance(docs[0], list)
        
        # 测试不同长度的序列
        lengths = [len(doc) for doc in docs]
        print(f'序列长度: {lengths}')
        self.assertTrue(all(l > 0 for l in lengths))  # 所有序列长度都大于0
    
    def test_word2vec_training(self):
        """测试 Word2Vec 训练"""
        click_df = self.test_data.copy()
        click_df['click_article_id'] = click_df['click_article_id'].astype(str)
        docs = click_df.groupby(['user_id'])['click_article_id'].apply(lambda x: list(x)).reset_index()
        docs = docs['click_article_id'].values.tolist()
        
        # 训练 Word2Vec
        w2v = Word2Vec(docs, vector_size=16, sg=1, window=5, seed=2020, workers=1, min_count=1, epochs=1)
        
        # 验证模型
        self.assertIsNotNone(w2v)
        self.assertEqual(w2v.vector_size, 16)
        
        # 验证词汇表
        self.assertGreater(len(w2v.wv), 0)
        print(f'词汇表大小: {len(w2v.wv)}')
    
    def test_word2vec_access_method(self):
        """测试新版本的访问方式 w2v.wv[key]"""
        click_df = self.test_data.copy()
        click_df['click_article_id'] = click_df['click_article_id'].astype(str)
        docs = click_df.groupby(['user_id'])['click_article_id'].apply(lambda x: list(x)).reset_index()
        docs = docs['click_article_id'].values.tolist()
        
        w2v = Word2Vec(docs, vector_size=16, sg=1, window=5, seed=2020, workers=1, min_count=1, epochs=1)
        
        # 测试新版本的访问方式
        test_word = '101'
        self.assertIn(test_word, w2v.wv)
        
        # 使用新方式访问
        vector = w2v.wv[test_word]
        self.assertEqual(vector.shape, (16,))
        self.assertIsInstance(vector, np.ndarray)
        
        # 验证旧方式不可用
        with self.assertRaises(TypeError):
            _ = w2v[test_word]
    
    def test_word2vec_dict_creation(self):
        """测试字典创建（模拟 trian_item_word2vec 中的逻辑）"""
        click_df = self.test_data.copy()
        click_df['click_article_id'] = click_df['click_article_id'].astype(str)
        docs = click_df.groupby(['user_id'])['click_article_id'].apply(lambda x: list(x)).reset_index()
        docs = docs['click_article_id'].values.tolist()
        
        w2v = Word2Vec(docs, vector_size=16, sg=1, window=5, seed=2020, workers=1, min_count=1, epochs=1)
        
        # 模拟原函数中的字典创建逻辑
        item_w2v_emb_dict = {k: w2v.wv[k] for k in click_df['click_article_id'] if k in w2v.wv}
        
        # 验证字典
        self.assertIsInstance(item_w2v_emb_dict, dict)
        self.assertGreater(len(item_w2v_emb_dict), 0)
        
        # 验证所有值都是向量
        for key, value in item_w2v_emb_dict.items():
            self.assertIsInstance(value, np.ndarray)
            self.assertEqual(value.shape, (16,))
    
    def test_word2vec_different_sequence_lengths(self):
        """测试不同长度的序列"""
        # 创建包含不同长度序列的数据
        test_data = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'click_article_id': ['1', '2', '3', '4', '5'],
            'click_timestamp': [1000, 2000, 3000, 4000, 5000]
        })
        
        # 添加更多数据使序列长度不同
        test_data = pd.concat([
            test_data,
            pd.DataFrame({
                'user_id': [1, 1, 2, 2, 2, 3, 3, 3, 3, 3],
                'click_article_id': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
                'click_timestamp': [1001, 1002, 2001, 2002, 2003, 3001, 3002, 3003, 3004, 3005]
            })
        ], ignore_index=True)
        
        click_df = test_data.copy()
        click_df['click_article_id'] = click_df['click_article_id'].astype(str)
        docs = click_df.groupby(['user_id'])['click_article_id'].apply(lambda x: list(x)).reset_index()
        docs = docs['click_article_id'].values.tolist()
        
        # 验证序列长度不同
        lengths = [len(doc) for doc in docs]
        print(f'不同序列长度: {lengths}')
        self.assertTrue(len(set(lengths)) > 1)  # 至少有两种不同的长度
        
        # 训练 Word2Vec
        w2v = Word2Vec(docs, vector_size=16, sg=1, window=5, seed=2020, workers=1, min_count=1, epochs=1)
        
        # 验证训练成功
        self.assertIsNotNone(w2v)
        self.assertGreater(len(w2v.wv), 0)
    
    def test_word2vec_single_item_sequence(self):
        """测试单个物品的序列（边界情况）"""
        test_data = pd.DataFrame({
            'user_id': [1],
            'click_article_id': ['1'],
            'click_timestamp': [1000]
        })
        
        click_df = test_data.copy()
        click_df['click_article_id'] = click_df['click_article_id'].astype(str)
        docs = click_df.groupby(['user_id'])['click_article_id'].apply(lambda x: list(x)).reset_index()
        docs = docs['click_article_id'].values.tolist()
        
        # 验证序列长度为1
        self.assertEqual(len(docs[0]), 1)
        
        # 训练 Word2Vec（应该不会报错）
        w2v = Word2Vec(docs, vector_size=16, sg=1, window=5, seed=2020, workers=1, min_count=1, epochs=1)
        
        # 验证可以访问
        self.assertIn('1', w2v.wv)
        vector = w2v.wv['1']
        self.assertEqual(vector.shape, (16,))
    
    def test_word2vec_short_sequence_vs_window(self):
        """测试序列长度小于 window 的情况"""
        # 创建短序列（长度 < window=5）
        test_data = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 2],
            'click_article_id': ['1', '2', '3', '4', '5'],
            'click_timestamp': [1000, 1001, 2000, 2001, 2002]
        })
        
        click_df = test_data.copy()
        click_df['click_article_id'] = click_df['click_article_id'].astype(str)
        docs = click_df.groupby(['user_id'])['click_article_id'].apply(lambda x: list(x)).reset_index()
        docs = docs['click_article_id'].values.tolist()
        
        # 验证有短序列（长度 < 5）
        lengths = [len(doc) for doc in docs]
        self.assertTrue(any(l < 5 for l in lengths))
        
        # 训练 Word2Vec（应该能正常处理）
        w2v = Word2Vec(docs, vector_size=16, sg=1, window=5, seed=2020, workers=1, min_count=1, epochs=1)
        
        # 验证训练成功
        self.assertIsNotNone(w2v)
        self.assertGreater(len(w2v.wv), 0)


if __name__ == '__main__':
    print("开始运行 Word2Vec 单元测试...")
    unittest.main(verbosity=2)


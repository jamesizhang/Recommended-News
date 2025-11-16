"""
测试Transformer推荐模型的评估指标
包括Hit@K, NDCG@K, Precision@K, MRR等指标
"""
import unittest
import pandas as pd
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from news3_transformer import evaluate_transformer_recommendation


class TestTransformerEvaluation(unittest.TestCase):
    """测试Transformer推荐评估指标"""
    
    def setUp(self):
        """创建测试数据"""
        # 创建模拟的召回结果：{user_id: [(item_id, score), ...]}
        self.user_recall_items_dict = {
            1: [(101, 0.95), (102, 0.85), (103, 0.75), (104, 0.65), (105, 0.55)],  # 真实物品101在第一位
            2: [(201, 0.90), (202, 0.80), (203, 0.70), (204, 0.60), (205, 0.50)],  # 真实物品203在第三位
            3: [(301, 0.88), (302, 0.78), (303, 0.68), (304, 0.58), (305, 0.48)],  # 真实物品305不在前5
            4: [(401, 0.92), (402, 0.82), (403, 0.72), (404, 0.62), (405, 0.52)],  # 真实物品401在第一位
            5: [(501, 0.87), (502, 0.77), (503, 0.67), (504, 0.57), (505, 0.47)],  # 真实物品不在前5
        }
        
        # 创建真实标签数据
        self.trn_last_click_df = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'click_article_id': [101, 203, 305, 401, 999]  # 用户5的真实物品不在召回列表中
        })
    
    def test_hit_rate(self):
        """测试Hit@K指标"""
        results = evaluate_transformer_recommendation(
            self.user_recall_items_dict, 
            self.trn_last_click_df, 
            k_list=[1, 3, 5]
        )
        
        # 验证Hit@1: 用户1和4的真实物品在第一位，用户2、3、5不在
        # 期望: Hit@1 = 2/5 = 0.4
        self.assertAlmostEqual(results['Hit@1'], 0.4, places=4)
        
        # 验证Hit@3: 用户1、2、4的真实物品在前3位，用户3、5不在
        # 期望: Hit@3 = 3/5 = 0.6
        self.assertAlmostEqual(results['Hit@3'], 0.6, places=4)
        
        # 验证Hit@5: 用户1、2、3、4的真实物品在前5位，用户5不在
        # 期望: Hit@5 = 4/5 = 0.8
        self.assertAlmostEqual(results['Hit@5'], 0.8, places=4)
    
    def test_ndcg(self):
        """测试NDCG@K指标"""
        results = evaluate_transformer_recommendation(
            self.user_recall_items_dict, 
            self.trn_last_click_df, 
            k_list=[1, 3, 5]
        )
        
        # NDCG@1: 用户1和4的真实物品在第一位，NDCG=1.0；其他用户NDCG=0.0
        # 期望: NDCG@1 = (1.0 + 0.0 + 0.0 + 1.0 + 0.0) / 5 = 0.4
        self.assertAlmostEqual(results['NDCG@1'], 0.4, places=4)
        
        # NDCG@3: 
        # 用户1: 真实物品在位置1, DCG = 1/log2(2) = 1.0, IDCG = 1.0, NDCG = 1.0
        # 用户2: 真实物品在位置3, DCG = 1/log2(4) = 0.5, IDCG = 1.0, NDCG = 0.5
        # 用户3: 真实物品不在前3, NDCG = 0.0
        # 用户4: 真实物品在位置1, NDCG = 1.0
        # 用户5: 真实物品不在前3, NDCG = 0.0
        # 期望: NDCG@3 = (1.0 + 0.5 + 0.0 + 1.0 + 0.0) / 5 = 0.5
        self.assertAlmostEqual(results['NDCG@3'], 0.5, places=4)
        
        # NDCG应该小于等于Hit Rate（因为考虑了位置）
        self.assertLessEqual(results['NDCG@1'], results['Hit@1'])
        self.assertLessEqual(results['NDCG@3'], results['Hit@3'])
    
    def test_precision(self):
        """测试Precision@K指标"""
        results = evaluate_transformer_recommendation(
            self.user_recall_items_dict, 
            self.trn_last_click_df, 
            k_list=[1, 3, 5]
        )
        
        # Precision@1: 用户1和4命中，期望 = 2/5 = 0.4
        self.assertAlmostEqual(results['Precision@1'], 0.4, places=4)
        
        # Precision@3: 用户1、2、4命中，期望 = 3/15 = 0.2 (3个命中 / 5个用户 * 3个推荐)
        # 实际上每个用户的Precision@3 = hit/3，然后取平均
        # 用户1: 1/3 = 0.333, 用户2: 1/3 = 0.333, 用户3: 0/3 = 0.0, 用户4: 1/3 = 0.333, 用户5: 0/3 = 0.0
        # 平均 = (0.333 + 0.333 + 0.0 + 0.333 + 0.0) / 5 = 0.2
        self.assertAlmostEqual(results['Precision@3'], 0.2, places=4)
        
        # Precision应该小于等于Hit Rate（因为Precision = Hit/K）
        self.assertLessEqual(results['Precision@1'], results['Hit@1'])
    
    def test_mrr(self):
        """测试MRR指标"""
        results = evaluate_transformer_recommendation(
            self.user_recall_items_dict, 
            self.trn_last_click_df, 
            k_list=[5]
        )
        
        # MRR计算：
        # 用户1: 真实物品在位置1, MRR = 1/1 = 1.0
        # 用户2: 真实物品在位置3, MRR = 1/3 = 0.333
        # 用户3: 真实物品在位置5, MRR = 1/5 = 0.2
        # 用户4: 真实物品在位置1, MRR = 1/1 = 1.0
        # 用户5: 真实物品不在召回列表中, MRR = 0.0
        # 期望: MRR = (1.0 + 0.333 + 0.2 + 1.0 + 0.0) / 5 = 0.5066
        self.assertAlmostEqual(results['MRR'], 0.5066, places=3)
    
    def test_empty_recall(self):
        """测试空召回列表的情况"""
        empty_recall_dict = {
            1: [],
            2: [(201, 0.9)]
        }
        empty_last_click_df = pd.DataFrame({
            'user_id': [1, 2],
            'click_article_id': [101, 201]
        })
        
        results = evaluate_transformer_recommendation(
            empty_recall_dict, 
            empty_last_click_df, 
            k_list=[1, 5]
        )
        
        # 用户1没有召回，所有指标应该为0
        # 用户2命中，Hit@1 = 1.0
        self.assertAlmostEqual(results['Hit@1'], 0.5, places=4)  # 1/2 = 0.5
        self.assertAlmostEqual(results['NDCG@1'], 0.5, places=4)
    
    def test_different_formats(self):
        """测试不同格式的召回结果"""
        # 格式1: [(item_id, score), ...]
        format1_dict = {
            1: [(101, 0.9), (102, 0.8)]
        }
        
        # 格式2: [item_id, ...]
        format2_dict = {
            1: [101, 102]
        }
        
        test_df = pd.DataFrame({
            'user_id': [1],
            'click_article_id': [101]
        })
        
        results1 = evaluate_transformer_recommendation(format1_dict, test_df, k_list=[1])
        results2 = evaluate_transformer_recommendation(format2_dict, test_df, k_list=[1])
        
        # 两种格式应该得到相同的结果
        self.assertEqual(results1['Hit@1'], results2['Hit@1'])
        self.assertEqual(results1['NDCG@1'], results2['NDCG@1'])
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 测试K值大于召回列表长度的情况
        short_recall_dict = {
            1: [(101, 0.9), (102, 0.8)]  # 只有2个物品
        }
        test_df = pd.DataFrame({
            'user_id': [1],
            'click_article_id': [101]
        })
        
        results = evaluate_transformer_recommendation(
            short_recall_dict, 
            test_df, 
            k_list=[1, 5, 10]  # K值大于召回列表长度
        )
        
        # 应该能正常计算，不会报错
        self.assertIn('Hit@1', results)
        self.assertIn('Hit@5', results)
        self.assertIn('Hit@10', results)
        
        # Hit@1应该等于Hit@5和Hit@10（因为只有2个物品）
        self.assertEqual(results['Hit@1'], results['Hit@5'])
        self.assertEqual(results['Hit@5'], results['Hit@10'])


if __name__ == '__main__':
    unittest.main()


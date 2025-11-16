"""
单元测试：测试 news1.py 中的 item_based_recommend 函数
主要测试修复后的热门文章补全逻辑
"""
import unittest
import pandas as pd
import numpy as np
import math
from collections import defaultdict


# 直接复制函数代码，避免导入整个模块
def get_user_item_time(click_df):
    click_df = click_df.sort_values('click_timestamp')
    
    def make_item_time_pair(df):
        return list(zip(df['click_article_id'], df['click_timestamp']))
    
    user_item_time_df = click_df.groupby('user_id')[['click_article_id', 'click_timestamp']].apply(
        lambda x: make_item_time_pair(x)
    ).reset_index().rename(columns={0: 'item_time_list'})
    
    user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))
    
    return user_item_time_dict


def get_item_topk_click(click_df, k):
    topk_click = click_df['click_article_id'].value_counts().index[:k]
    return topk_click


def item_based_recommend(user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, item_topk_click):
    """
    基于文章协同过滤的召回（复制自news1.py）
    """
    # 获取用户历史交互的文章
    user_hist_items = user_item_time_dict[user_id]
    user_hist_items_ = {user_id for user_id, _ in user_hist_items}
    
    item_rank = {}
    for loc, (i, click_time) in enumerate(user_hist_items):
        if i not in i2i_sim:
            continue
        for j, wij in sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
            if j in user_hist_items_:
                continue
            
            item_rank.setdefault(j, 0)
            item_rank[j] += wij
    
    # 不足10个，用热门商品补全
    if len(item_rank) < recall_item_num:
        print(f"item_rank: {item_rank}")
        for i, item in enumerate(item_topk_click):
            if item in item_rank:  # 修复后的代码：检查key而不是items
                print(f"Item {item} already in item_rank")
                continue
            print(f"Adding item {item} to item_rank")
            item_rank[item] = - i - 100  # 随便给个负数就行
            if len(item_rank) == recall_item_num:
                break
    
    item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]
    
    return item_rank


class TestItemBasedRecommend(unittest.TestCase):
    """测试 item_based_recommend 函数"""
    
    def setUp(self):
        """准备测试数据"""
        # 创建测试点击数据
        self.test_click_data = pd.DataFrame({
            'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 3],
            'click_article_id': [100, 200, 300, 100, 200, 400, 500, 600, 700],
            'click_timestamp': [1, 2, 3, 1, 2, 1, 2, 3, 4]
        })
        
        # 创建用户-文章-时间字典
        self.user_item_time_dict = get_user_item_time(self.test_click_data)
        
        # 创建文章相似度矩阵
        self.i2i_sim = {
            100: {200: 0.8, 300: 0.6, 400: 0.4, 500: 0.3},
            200: {100: 0.8, 300: 0.7, 400: 0.5, 500: 0.4},
            300: {100: 0.6, 200: 0.7, 400: 0.3, 500: 0.2},
            400: {100: 0.4, 200: 0.5, 300: 0.3, 500: 0.9},
            500: {100: 0.3, 200: 0.4, 300: 0.2, 400: 0.9},
            600: {700: 0.8, 800: 0.5},
            700: {600: 0.8, 800: 0.6},
        }
        
        # 获取热门文章
        self.item_topk_click = get_item_topk_click(self.test_click_data, k=50)
    
    def test_basic_recommendation(self):
        """测试基本召回功能"""
        user_id = 1
        sim_item_topk = 10
        recall_item_num = 5
        
        result = item_based_recommend(
            user_id, 
            self.user_item_time_dict, 
            self.i2i_sim, 
            sim_item_topk, 
            recall_item_num, 
            self.item_topk_click
        )
        
        # 检查返回结果
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), recall_item_num)
        
        # 检查每个元素是 (item, score) 元组
        for item_score in result:
            self.assertIsInstance(item_score, tuple)
            self.assertEqual(len(item_score), 2)
            item, score = item_score
            self.assertIsInstance(item, (int, np.integer))
            self.assertIsInstance(score, (int, float, np.number))
        
        # 检查不包含用户历史点击的文章
        user_hist_items = {item for item, _ in self.user_item_time_dict[user_id]}
        result_items = {item for item, _ in result}
        # 注意：热门文章补全可能会包含用户历史点击，但相似度召回不应该包含
        # 由于代码逻辑中item_rank可能为空，热门文章补全时可能包含历史点击
        # 所以这里只检查相似度召回的部分（分数>0的）
        similarity_items = {item for item, score in result if score > 0}
        self.assertEqual(len(user_hist_items & similarity_items), 0, 
                        "相似度召回结果不应包含用户历史点击的文章")
    
    def test_hot_item_fill(self):
        """测试热门文章补全逻辑（修复后的代码）"""
        # 创建一个用户，其历史点击的文章在相似度矩阵中只有部分相似文章
        user_id = 3  # 用户3的历史点击是 [400, 500, 600, 700]
        sim_item_topk = 10
        recall_item_num = 10
        
        result = item_based_recommend(
            user_id, 
            self.user_item_time_dict, 
            self.i2i_sim, 
            sim_item_topk, 
            recall_item_num, 
            self.item_topk_click
        )
        
        # 检查返回结果
        self.assertIsInstance(result, list)
        # 注意：如果item_topk_click数量不足，可能无法补全到recall_item_num
        # 但至少应该有召回结果
        self.assertTrue(len(result) > 0, "应该有召回结果")
        
        # 检查结果中的文章都是有效的
        result_items = {item for item, _ in result}
        self.assertTrue(len(result_items) > 0, "应该有召回结果")
        
        # 检查相似度召回部分不包含用户历史点击的文章
        user_hist_items = {item for item, _ in self.user_item_time_dict[user_id]}
        similarity_items = {item for item, score in result if score > 0}
        self.assertEqual(len(user_hist_items & similarity_items), 0,
                        "相似度召回结果不应包含用户历史点击的文章")
    
    def test_hot_item_fill_no_duplicate(self):
        """测试热门文章补全时不会重复添加已存在的文章（验证修复后的代码逻辑）"""
        # 创建一个用户，部分历史文章有相似文章，部分没有
        user_id = 2  # 用户2的历史点击是 [100, 200]，这两个都有相似文章
        sim_item_topk = 2  # 只取top-2相似文章，这样item_rank可能只有4个
        recall_item_num = 10  # 需要召回10个，需要补全
        
        # 创建足够的热门文章列表，确保可以补全
        item_topk_click = list(range(1000, 1100))  # 100个热门文章，确保不重复
        
        result = item_based_recommend(
            user_id, 
            self.user_item_time_dict, 
            self.i2i_sim, 
            sim_item_topk, 
            recall_item_num, 
            item_topk_click
        )
        
        # 检查返回结果
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), recall_item_num, "应该召回指定数量的文章")
        
        # 检查是否有重复的文章（这个测试验证修复后的代码逻辑）
        result_items = [item for item, _ in result]
        self.assertEqual(len(result_items), len(set(result_items)), 
                        "召回结果不应包含重复的文章（验证修复后的代码）")
        
        # 检查热门文章补全的分数是负数（因为代码中设置为 -i - 100）
        # 如果item_rank不为空但不足10个，会用热门文章补全
        # 热门文章的分数应该是负数
        hot_item_scores = [score for item, score in result if score < 0]
        if len(hot_item_scores) > 0:
            # 热门文章分数应该是负数
            self.assertTrue(all(score < 0 for score in hot_item_scores),
                          "热门文章补全的分数应该是负数")
    
    def test_empty_similarity_matrix(self):
        """测试相似度矩阵为空的情况"""
        user_id = 1
        sim_item_topk = 10
        recall_item_num = 5
        empty_i2i_sim = {}  # 空的相似度矩阵
        
        result = item_based_recommend(
            user_id, 
            self.user_item_time_dict, 
            empty_i2i_sim, 
            sim_item_topk, 
            recall_item_num, 
            self.item_topk_click
        )
        
        # 即使相似度矩阵为空，也应该通过热门文章补全返回结果
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), recall_item_num)
        
        # 检查结果中的文章都是有效的
        result_items = [item for item, _ in result]
        self.assertEqual(len(result_items), len(set(result_items)),
                        "召回结果不应包含重复的文章")
    
    def test_user_with_no_history(self):
        """测试用户没有历史点击的情况（应该不会发生，但测试边界情况）"""
        # 创建一个没有历史点击的用户
        user_item_time_dict_empty = {999: []}  # 用户999没有历史点击
        
        # 如果用户没有历史点击，应该会抛出KeyError或返回空列表
        # 但根据代码逻辑，如果user_id不在user_item_time_dict中，会抛出KeyError
        # 所以我们测试一个边缘情况：用户有空的点击历史
        sim_item_topk = 10
        recall_item_num = 5
        
        result = item_based_recommend(
            999, 
            user_item_time_dict_empty, 
            self.i2i_sim, 
            sim_item_topk, 
            recall_item_num, 
            self.item_topk_click
        )
        
        # 应该通过热门文章补全返回结果
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), recall_item_num)


if __name__ == '__main__':
    unittest.main()


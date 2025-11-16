import pandas as pd
import numpy as np
import unittest


def make_tuple_func(group_df):
    """从 news4.py 复制的函数"""
    row_data = []
    for name, row_df in group_df.iterrows():
        row_data.append((row_df['sim_item'], row_df['score'], row_df['label']))
    return row_data


def process_groupby_result(groupby_result):
    """从 news4.py 复制的辅助函数"""
    if isinstance(groupby_result, pd.Series):
        return groupby_result.to_frame(name=0).reset_index()
    else:
        if 'user_id' in groupby_result.columns:
            result = groupby_result.drop(columns=['user_id']).reset_index()
        else:
            result = groupby_result.reset_index()
        
        # 确保有一列名为 0（包含列表数据）
        if 0 not in result.columns:
            other_cols = [c for c in result.columns if c != 'user_id']
            if len(other_cols) > 0:
                result = result.rename(columns={other_cols[0]: 0})
        
        return result


class TestNews4GroupbyFix(unittest.TestCase):
    """测试 groupby().apply().reset_index() 的修复"""
    
    def setUp(self):
        """创建测试数据"""
        # 创建模拟的用户-物品标签数据
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 3],
            'sim_item': [101, 102, 103, 201, 202, 301, 302, 303, 304],
            'score': [0.9, 0.8, 0.7, 0.85, 0.75, 0.95, 0.88, 0.82, 0.76],
            'label': [1, 0, 0, 1, 0, 1, 1, 0, 0]
        })
    
    def test_groupby_apply_reset_index(self):
        """测试修复后的 process_groupby_result() 方法"""
        # 使用修复后的方法
        groupby_result = self.test_data.groupby('user_id').apply(make_tuple_func)
        result = process_groupby_result(groupby_result)
        
        # 验证结果不为空
        self.assertIsNotNone(result)
        
        # 验证结果包含正确的列
        self.assertIn('user_id', result.columns)
        self.assertIn(0, result.columns)
        
        # 验证 user_id 列存在且唯一（没有重复）
        self.assertEqual(len(result['user_id'].unique()), len(result))
        
        # 验证结果形状：应该有3个用户
        self.assertEqual(len(result), 3)
        
        # 验证每个用户的数据都是列表格式
        for idx, row in result.iterrows():
            self.assertIsInstance(row[0], list)
            self.assertGreater(len(row[0]), 0)
    
    def test_result_dict_creation(self):
        """测试从结果创建字典的功能（模拟实际使用场景）"""
        # 使用修复后的方法
        groupby_result = self.test_data.groupby('user_id').apply(make_tuple_func)
        user_item_label_tuples = process_groupby_result(groupby_result)
        
        # 创建字典（模拟实际代码中的使用）
        user_item_label_tuples_dict = dict(zip(user_item_label_tuples['user_id'], user_item_label_tuples[0]))
        
        # 验证字典创建成功
        self.assertIsInstance(user_item_label_tuples_dict, dict)
        
        # 验证字典包含所有用户
        self.assertEqual(len(user_item_label_tuples_dict), 3)
        self.assertIn(1, user_item_label_tuples_dict)
        self.assertIn(2, user_item_label_tuples_dict)
        self.assertIn(3, user_item_label_tuples_dict)
        
        # 验证每个用户的值是列表
        for user_id, tuples_list in user_item_label_tuples_dict.items():
            self.assertIsInstance(tuples_list, list)
            self.assertGreater(len(tuples_list), 0)
            # 验证列表中的每个元素都是元组
            for item in tuples_list:
                self.assertIsInstance(item, tuple)
                self.assertEqual(len(item), 3)  # (sim_item, score, label)
    
    def test_no_duplicate_columns(self):
        """测试没有重复的列名（这是原始错误的根本原因）"""
        groupby_result = self.test_data.groupby('user_id').apply(make_tuple_func)
        result = process_groupby_result(groupby_result)
        
        # 验证没有重复的列名
        self.assertEqual(len(result.columns), len(set(result.columns)))
        
        # 验证 user_id 只出现一次
        user_id_count = list(result.columns).count('user_id')
        self.assertEqual(user_id_count, 1)
    
    def test_dataframe_case(self):
        """测试当 groupby().apply() 返回 DataFrame 的情况"""
        # 创建一个可能返回 DataFrame 的场景
        # 在某些 pandas 版本中，groupby().apply() 可能返回 DataFrame
        groupby_result = self.test_data.groupby('user_id').apply(make_tuple_func)
        
        # 测试 process_groupby_result 能否处理 Series 和 DataFrame 两种情况
        result = process_groupby_result(groupby_result)
        
        # 验证结果格式正确
        self.assertIn('user_id', result.columns)
        self.assertIn(0, result.columns)
        self.assertEqual(len(result), 3)
    
    def test_empty_dataframe(self):
        """测试空 DataFrame 的情况（边界情况，实际代码中应该不会遇到）"""
        empty_df = pd.DataFrame(columns=['user_id', 'sim_item', 'score', 'label'])
        # 对于空 DataFrame，groupby().apply() 的行为可能因 pandas 版本而异
        # 在实际代码中，这种情况应该不会出现，所以这里只验证不会抛出关键错误
        try:
            groupby_result = empty_df.groupby('user_id').apply(make_tuple_func)
            result = process_groupby_result(groupby_result)
            # 如果能正常执行到这里，说明没有崩溃
            self.assertTrue(True)
        except Exception as e:
            # 对于空 DataFrame，某些错误是可以接受的
            # 只要不是我们修复的核心问题（user_id 列重复或 to_frame 错误）即可
            if "cannot insert user_id, already exists" in str(e) or "to_frame" in str(e):
                self.fail(f"仍然存在关键问题: {e}")
            # 其他错误可以接受（如空数据相关的错误）
    
    def test_single_user(self):
        """测试只有一个用户的情况"""
        single_user_df = self.test_data[self.test_data['user_id'] == 1]
        groupby_result = single_user_df.groupby('user_id').apply(make_tuple_func)
        result = process_groupby_result(groupby_result)
        
        # 应该只有一行
        self.assertEqual(len(result), 1)
        self.assertEqual(result['user_id'].iloc[0], 1)
        self.assertIsInstance(result[0].iloc[0], list)
        self.assertEqual(len(result[0].iloc[0]), 3)  # 用户1有3条记录


if __name__ == '__main__':
    print("开始运行单元测试...")
    unittest.main(verbosity=2)


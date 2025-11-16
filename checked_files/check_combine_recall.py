"""
检查多路召回合并的效果
"""
import pickle
import os
import pandas as pd
import numpy as np
from collections import defaultdict

save_path = '../temp_results/'

print("=" * 80)
print("多路召回合并效果检查")
print("=" * 80)

# 1. 检查各路召回结果是否存在
print("\n1. 检查各路召回结果文件:")
recall_files = {
    'itemcf_sim_itemcf_recall': 'itemcf_u2i_dict.pkl',
    'embedding_sim_item_recall': 'embedding_u2i_dict.pkl',
    'youtubednn_recall': 'youtube_u2i_dict.pkl',
    'youtubednn_usercf_recall': 'youtube_u2u_sim.pkl',
    'cold_start_recall': 'cold_start_user_items_dict.pkl'
}

user_multi_recall_dict = {}
for method, filename in recall_files.items():
    filepath = os.path.join(save_path, filename)
    if os.path.exists(filepath):
        try:
            user_multi_recall_dict[method] = pickle.load(open(filepath, 'rb'))
            print(f"  ✓ {method}: {len(user_multi_recall_dict[method])} 用户")
        except Exception as e:
            print(f"  ✗ {method}: 加载失败 - {e}")
    else:
        print(f"  ✗ {method}: 文件不存在 - {filepath}")

# 2. 分析各路召回的统计信息
print("\n2. 各路召回统计信息:")
for method, recall_dict in user_multi_recall_dict.items():
    if len(recall_dict) == 0:
        print(f"  {method}: 空字典")
        continue
    
    # 统计每个用户的召回数量
    recall_counts = [len(items) for items in recall_dict.values()]
    total_items = sum(recall_counts)
    unique_items = set()
    for items in recall_dict.values():
        unique_items.update([item[0] if isinstance(item, tuple) else item for item in items])
    
    print(f"  {method}:")
    print(f"    用户数: {len(recall_dict)}")
    print(f"    平均召回数/用户: {np.mean(recall_counts):.2f}")
    print(f"    召回总数: {total_items}")
    print(f"    唯一物品数: {len(unique_items)}")
    print(f"    覆盖率: {len(unique_items) / total_items * 100:.2f}%")

# 3. 检查归一化函数
print("\n3. 归一化函数测试:")
def norm_user_recall_items_sim(sorted_item_list):
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

# 测试归一化
test_cases = [
    [(1, 10), (2, 8), (3, 5), (4, 2)],  # 正常情况
    [(1, 5), (2, 5), (3, 5)],  # 所有分数相同
    [(1, 10)],  # 只有一个物品
    [(1, 0), (2, 0), (3, 0)],  # 所有分数为0
]

for i, test_case in enumerate(test_cases):
    result = norm_user_recall_items_sim(test_case)
    print(f"  测试 {i+1}: {test_case}")
    print(f"    归一化后: {result}")

# 4. 检查合并逻辑
print("\n4. 合并逻辑检查:")
weight_dict = {
    'itemcf_sim_itemcf_recall': 0.5,
    'embedding_sim_item_recall': 1.0,
    'youtubednn_recall': 2.0,
    'youtubednn_usercf_recall': 0.5,
    'cold_start_recall': 1.0
}

# 检查是否有用户在所有召回方法中都出现
if len(user_multi_recall_dict) > 0:
    all_users = set()
    for recall_dict in user_multi_recall_dict.values():
        all_users.update(recall_dict.keys())
    
    print(f"  总用户数: {len(all_users)}")
    
    # 统计每个用户出现在几个召回方法中
    user_method_count = defaultdict(int)
    for method, recall_dict in user_multi_recall_dict.items():
        for user in recall_dict.keys():
            user_method_count[user] += 1
    
    method_counts = list(user_method_count.values())
    print(f"  平均每个用户出现在 {np.mean(method_counts):.2f} 个召回方法中")
    print(f"  最多出现在 {max(method_counts) if method_counts else 0} 个召回方法中")
    print(f"  最少出现在 {min(method_counts) if method_counts else 0} 个召回方法中")

# 5. 检查最终合并结果
print("\n5. 最终合并结果检查:")
final_recall_file = os.path.join(save_path, 'final_recall_items_dict.pkl')
if os.path.exists(final_recall_file):
    try:
        final_recall_dict = pickle.load(open(final_recall_file, 'rb'))
        print(f"  ✓ 最终召回结果: {len(final_recall_dict)} 用户")
        
        # 统计最终召回的物品数量
        recall_counts = [len(items) for items in final_recall_dict.values()]
        print(f"    平均召回数/用户: {np.mean(recall_counts):.2f}")
        print(f"    最大召回数: {max(recall_counts) if recall_counts else 0}")
        print(f"    最小召回数: {min(recall_counts) if recall_counts else 0}")
        
        # 检查分数分布
        all_scores = []
        for items in final_recall_dict.values():
            # 处理不同的数据格式
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, tuple) and len(item) == 2:
                        all_scores.append(item[1])
                    elif isinstance(item, (int, float)):
                        all_scores.append(item)
            elif isinstance(items, dict):
                all_scores.extend(items.values())
        
        if all_scores:
            print(f"    分数统计:")
            print(f"      平均分数: {np.mean(all_scores):.4f}")
            print(f"      最大分数: {np.max(all_scores):.4f}")
            print(f"      最小分数: {np.min(all_scores):.4f}")
            print(f"      分数标准差: {np.std(all_scores):.4f}")
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
else:
    print(f"  ✗ 文件不存在: {final_recall_file}")

# 6. 潜在问题分析
print("\n6. 潜在问题分析:")
print("  a) 归一化问题:")
print("     - 如果某个召回方法只有一个物品，不会归一化")
print("     - 如果所有分数相同，归一化后都是 0 或 1")
print("     - 建议：使用更稳定的归一化方法（如 Z-score 或 sigmoid）")
print("  b) 权重设置:")
print("     - 当前权重基于单路召回效果，但合并后效果可能不同")
print("     - 建议：基于合并后的效果调整权重")
print("  c) 物品重叠:")
print("     - 不同召回方法可能召回相同物品，合并时会累加分数")
print("     - 这是合理的，但需要确保权重设置合理")

print("\n" + "=" * 80)


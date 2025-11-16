"""
改进的多路召回合并函数
修复归一化问题和合并逻辑
"""
import pickle
import os
from collections import defaultdict
import numpy as np
from tqdm import tqdm

def improved_norm_user_recall_items_sim(sorted_item_list):
    """
    改进的归一化函数
    处理边界情况：
    1. 只有一个物品：保持原分数或归一化到固定值
    2. 所有分数相同：使用固定分数或基于排名
    3. 分数为0：使用排名作为分数
    """
    if len(sorted_item_list) == 0:
        return []
    
    if len(sorted_item_list) == 1:
        # 只有一个物品，归一化到0.5（中等分数）
        return [(sorted_item_list[0][0], 0.5)]
    
    scores = [item[1] for item in sorted_item_list]
    min_sim = min(scores)
    max_sim = max(scores)
    
    # 如果所有分数相同，使用排名作为分数
    if max_sim == min_sim:
        norm_sorted_item_list = []
        for i, (item, score) in enumerate(sorted_item_list):
            # 使用排名归一化：第一个物品分数最高
            norm_score = 1.0 - (i / len(sorted_item_list)) * 0.5  # 范围 [0.5, 1.0]
            norm_sorted_item_list.append((item, norm_score))
        return norm_sorted_item_list
    
    # 正常归一化
    norm_sorted_item_list = []
    for item, score in sorted_item_list:
        if max_sim > min_sim:
            norm_score = (score - min_sim) / (max_sim - min_sim)
        else:
            norm_score = 0.5  # 默认中等分数
        norm_sorted_item_list.append((item, norm_score))
    
    return norm_sorted_item_list


def improved_combine_recall_results(user_multi_recall_dict, weight_dict=None, topk=25):
    """
    改进的多路召回合并函数
    """
    final_recall_items_dict = {}
    
    print('多路召回合并...')
    for method, user_recall_items in tqdm(user_multi_recall_dict.items()):
        print(f'{method}...')
        
        # 获取权重
        if weight_dict is None:
            recall_method_weight = 1.0
        else:
            recall_method_weight = weight_dict.get(method, 1.0)
        
        # 对每个用户的召回结果进行归一化
        for user_id, sorted_item_list in user_recall_items.items():
            # 确保是列表格式
            if not isinstance(sorted_item_list, list):
                continue
            
            # 归一化
            norm_item_list = improved_norm_user_recall_items_sim(sorted_item_list)
            
            # 合并到最终结果
            final_recall_items_dict.setdefault(user_id, {})
            for item, score in norm_item_list:
                final_recall_items_dict[user_id].setdefault(item, 0.0)
                final_recall_items_dict[user_id][item] += recall_method_weight * score
    
    # 排序并取topk
    final_recall_items_dict_rank = {}
    for user, recall_item_dict in final_recall_items_dict.items():
        final_recall_items_dict_rank[user] = sorted(
            recall_item_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:topk]
    
    return final_recall_items_dict_rank


# 测试改进的合并函数
if __name__ == '__main__':
    save_path = '../temp_results/'
    
    # 加载召回结果
    user_multi_recall_dict = {}
    recall_files = {
        'youtubednn_recall': 'youtube_u2i_dict.pkl',
        'youtubednn_usercf_recall': 'youtube_u2u_sim.pkl',
        'cold_start_recall': 'cold_start_user_items_dict.pkl'
    }
    
    for method, filename in recall_files.items():
        filepath = os.path.join(save_path, filename)
        if os.path.exists(filepath):
            try:
                user_multi_recall_dict[method] = pickle.load(open(filepath, 'rb'))
                print(f"加载 {method}: {len(user_multi_recall_dict[method])} 用户")
            except Exception as e:
                print(f"加载 {method} 失败: {e}")
    
    # 权重设置
    weight_dict = {
        'youtubednn_recall': 2.0,
        'youtubednn_usercf_recall': 0.5,
        'cold_start_recall': 1.0
    }
    
    # 合并召回结果
    final_recall = improved_combine_recall_results(
        user_multi_recall_dict, 
        weight_dict=weight_dict, 
        topk=150
    )
    
    # 保存结果
    output_path = os.path.join(save_path, 'improved_final_recall_items_dict.pkl')
    pickle.dump(final_recall, open(output_path, 'wb'))
    print(f"\n改进的合并结果已保存到: {output_path}")
    
    # 统计信息
    recall_counts = [len(items) for items in final_recall.values()]
    print(f"\n统计信息:")
    print(f"  用户数: {len(final_recall)}")
    print(f"  平均召回数/用户: {np.mean(recall_counts):.2f}")
    print(f"  最大召回数: {max(recall_counts)}")
    print(f"  最小召回数: {min(recall_counts)}")


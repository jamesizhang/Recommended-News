"""
合并 Transformer 召回结果到多路召回中

功能：
1. 加载现有的各路召回结果（ItemCF, Embedding, YouTubeDNN, Cold Start）
2. 加载 Transformer 召回结果
3. 将 Transformer 召回结果合并到多路召回字典中
4. 调用合并函数进行加权融合
5. 可选：评估合并后的召回效果
"""

import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 路径配置
data_path = '../data_raw/'
save_path = '../temp_results/'


def get_hist_and_last_click(all_click):
    """
    获取历史点击和最后一次点击
    
    Args:
        all_click: 包含所有点击数据的 DataFrame
    
    Returns:
        click_hist_df: 历史点击数据（每个用户除最后一次点击外的所有点击）
        click_last_df: 最后一次点击数据
    """
    all_click = all_click.sort_values(by=['user_id', 'click_timestamp'])
    click_last_df = all_click.groupby('user_id').tail(1)
    
    # 如果用户只有一个点击，hist为空了，会导致训练的时候这个用户不可见，此时默认泄露一下
    def hist_func(user_df):
        if len(user_df) == 1:
            return user_df
        else:
            return user_df[:-1]
    
    click_hist_df = all_click.groupby('user_id').apply(hist_func).reset_index(drop=True)
    
    return click_hist_df, click_last_df


def metrics_recall(user_recall_items_dict, trn_last_click_df, topk=50):
    """
    评估召回效果
    
    Args:
        user_recall_items_dict: 用户召回字典，格式为 {user_id: [(item_id, score), ...]}
        trn_last_click_df: 训练集中每个用户的最后一次点击
        topk: 评估的topk值
    
    Returns:
        results: 评估结果字典
    """
    last_click_item_dict = dict(zip(trn_last_click_df['user_id'], trn_last_click_df['click_article_id']))
    user_num = len(user_recall_items_dict)
    
    results = {}
    for k in range(10, topk + 1, 10):
        hit_num = 0
        for user, item_list in user_recall_items_dict.items():
            if user not in last_click_item_dict:
                continue
            # 获取前k个召回的结果
            tmp_recall_items = [x[0] for x in user_recall_items_dict[user][:k]]
            if last_click_item_dict[user] in set(tmp_recall_items):
                hit_num += 1
        
        hit_rate = round(hit_num * 1.0 / user_num, 5) if user_num > 0 else 0.0
        results[k] = {'hit_num': hit_num, 'hit_rate': hit_rate, 'user_num': user_num}
        print(' topk: ', k, ' : ', 'hit_num: ', hit_num, 'hit_rate: ', hit_rate, 'user_num : ', user_num)
    
    return results


def combine_recall_results(user_multi_recall_dict, weight_dict=None, topk=25):
    """
    合并多路召回结果
    
    Args:
        user_multi_recall_dict: 多路召回字典，格式为 {method_name: {user_id: [(item_id, score), ...]}}
        weight_dict: 各召回方法的权重字典，格式为 {method_name: weight}
        topk: 最终合并后每个用户召回的物品数量
    
    Returns:
        final_recall_items_dict_rank: 合并后的召回结果，格式为 {user_id: [(item_id, score), ...]}
    """
    final_recall_items_dict = {}
    
    # 对每一种召回结果按照用户进行归一化，方便后面多种召回结果，相同用户的物品之间权重相加
    def norm_user_recall_items_sim(sorted_item_list):
        """
        归一化用户召回结果
        
        如果冷启动中没有文章或者只有一篇文章，直接返回，出现这种情况的原因可能是冷启动召回的文章数量太少了，
        基于规则筛选之后就没有文章了, 这里还可以做一些其他的策略性的筛选
        """
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
    
    print('多路召回合并...')
    for method, user_recall_items in tqdm(user_multi_recall_dict.items()):
        print(method + '...')
        
        # 在计算最终召回结果的时候，也可以为每一种召回结果设置一个权重
        if weight_dict is None:
            recall_method_weight = 1.0
        else:
            recall_method_weight = weight_dict.get(method, 1.0)
        
        # 进行归一化
        for user_id, sorted_item_list in user_recall_items.items():
            user_recall_items[user_id] = norm_user_recall_items_sim(sorted_item_list)
        
        # 合并到最终结果
        for user_id, sorted_item_list in user_recall_items.items():
            final_recall_items_dict.setdefault(user_id, {})
            for item, score in sorted_item_list:
                final_recall_items_dict[user_id].setdefault(item, 0.0)
                final_recall_items_dict[user_id][item] += recall_method_weight * score
    
    # 多路召回时也可以控制最终的召回数量
    final_recall_items_dict_rank = {}
    for user, recall_item_dict in final_recall_items_dict.items():
        final_recall_items_dict_rank[user] = sorted(
            recall_item_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:topk]
    
    # 将多路召回后的最终结果字典保存到本地
    pickle.dump(
        final_recall_items_dict, 
        open(os.path.join(save_path, 'final_recall_items_dict.pkl'), 'wb')
    )
    print(f"合并后的召回结果已保存到: {save_path}final_recall_items_dict.pkl")
    
    return final_recall_items_dict_rank


def load_recall_results(use_sample=False):
    """
    加载现有的各路召回结果
    
    Args:
        use_sample: 是否使用采样模式的结果
    
    Returns:
        user_multi_recall_dict: 多路召回字典
    """
    user_multi_recall_dict = {}
    
    # 定义召回结果文件映射
    recall_files = {
        'itemcf_sim_itemcf_recall': 'itemcf_u2i_dict.pkl',
        'embedding_sim_item_recall': 'embedding_u2i_dict.pkl',
        'youtubednn_recall': 'youtube_u2i_dict.pkl',
        'youtubednn_usercf_recall': 'youtube_u2u_sim.pkl',
        'cold_start_recall': 'cold_start_user_items_dict.pkl'
    }
    
    print("=" * 80)
    print("加载各路召回结果...")
    print("=" * 80)
    
    for method, filename in recall_files.items():
        filepath = os.path.join(save_path, filename)
        if os.path.exists(filepath):
            try:
                recall_dict = pickle.load(open(filepath, 'rb'))
                # 确保格式正确：{user_id: [(item_id, score), ...]}
                if isinstance(recall_dict, dict):
                    # 检查第一个值是否为列表或字典
                    if len(recall_dict) > 0:
                        first_user = list(recall_dict.keys())[0]
                        first_value = recall_dict[first_user]
                        
                        # 如果是字典格式，转换为列表格式
                        if isinstance(first_value, dict):
                            recall_dict = {
                                user: sorted(items.items(), key=lambda x: x[1], reverse=True)
                                for user, items in recall_dict.items()
                            }
                    
                    user_multi_recall_dict[method] = recall_dict
                    print(f"  ✓ {method}: {len(recall_dict)} 用户")
                else:
                    print(f"  ✗ {method}: 数据格式不正确（不是字典）")
            except Exception as e:
                print(f"  ✗ {method}: 加载失败 - {e}")
        else:
            print(f"  ✗ {method}: 文件不存在 - {filepath}")
    
    return user_multi_recall_dict


def load_transformer_recall(use_sample=False):
    """
    加载 Transformer 召回结果
    
    Args:
        use_sample: 是否使用采样模式的结果
    
    Returns:
        transformer_recall_dict: Transformer 召回字典，格式为 {user_id: [(item_id, score), ...]}
    """
    print("\n" + "=" * 80)
    print("加载 Transformer 召回结果...")
    print("=" * 80)
    
    # 根据模式选择文件
    if use_sample:
        recall_file = os.path.join(save_path, 'transformer_recall_dict_test_sample.pkl')
    else:
        recall_file = os.path.join(save_path, 'transformer_recall_dict_test.pkl')
    
    if not os.path.exists(recall_file):
        print(f"  ✗ Transformer 召回文件不存在: {recall_file}")
        return None
    
    try:
        recall_dict = pickle.load(open(recall_file, 'rb'))
        
        # 确保格式正确：{user_id: [(item_id, score), ...]}
        if isinstance(recall_dict, dict):
            if len(recall_dict) > 0:
                first_user = list(recall_dict.keys())[0]
                first_value = recall_dict[first_user]
                
                # 如果是字典格式，转换为列表格式
                if isinstance(first_value, dict):
                    recall_dict = {
                        user: sorted(items.items(), key=lambda x: x[1], reverse=True)
                        for user, items in recall_dict.items()
                    }
                # 如果已经是列表格式，确保是排序的
                elif isinstance(first_value, list):
                    # 检查是否已排序
                    for user in recall_dict:
                        recall_dict[user] = sorted(recall_dict[user], key=lambda x: x[1], reverse=True)
            
            print(f"  ✓ Transformer 召回: {len(recall_dict)} 用户")
            print(f"  ✓ 文件路径: {recall_file}")
            
            # 统计信息
            recall_counts = [len(items) for items in recall_dict.values()]
            if recall_counts:
                print(f"  ✓ 平均召回数/用户: {np.mean(recall_counts):.2f}")
                print(f"  ✓ 最大召回数/用户: {np.max(recall_counts)}")
                print(f"  ✓ 最小召回数/用户: {np.min(recall_counts)}")
            
            return recall_dict
        else:
            print(f"  ✗ Transformer 召回数据格式不正确（不是字典）")
            return None
    except Exception as e:
        print(f"  ✗ Transformer 召回加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def merge_transformer_recall(use_sample=False, transformer_weight=1.5, topk=150, evaluate=True):
    """
    合并 Transformer 召回结果到多路召回中
    
    Args:
        use_sample: 是否使用采样模式的结果
        transformer_weight: Transformer 召回的权重（默认 1.5，因为 Transformer 效果较好）
        topk: 最终合并后每个用户召回的物品数量
        evaluate: 是否评估合并后的召回效果
    
    Returns:
        final_recall_items_dict_rank: 合并后的召回结果
    """
    print("\n" + "=" * 80)
    print("合并 Transformer 召回结果")
    print("=" * 80)
    
    # 1. 加载现有的各路召回结果
    user_multi_recall_dict = load_recall_results(use_sample=use_sample)
    
    # 2. 加载 Transformer 召回结果
    transformer_recall_dict = load_transformer_recall(use_sample=use_sample)
    
    if transformer_recall_dict is None:
        print("\n  ✗ 无法加载 Transformer 召回结果，退出")
        return None
    
    # 3. 将 Transformer 召回结果添加到多路召回字典中
    user_multi_recall_dict['transformer_recall'] = transformer_recall_dict
    print(f"\n  ✓ 已添加 Transformer 召回到多路召回字典中")
    print(f"  ✓ 当前召回方法数: {len(user_multi_recall_dict)}")
    
    # 4. 设置权重字典（根据召回效果调整）
    # 注意：如果某些召回方法不存在，权重字典中可以不包含它们
    weight_dict = {
        'itemcf_sim_itemcf_recall': 0.5,      # 降低权重：命中率低
        'embedding_sim_item_recall': 1.0,     # 标准权重
        'youtubednn_recall': 2.0,              # 提高权重：命中率最高
        'youtubednn_usercf_recall': 0.5,      # 降低权重：命中率低
        'cold_start_recall': 1.0,              # 标准权重
        'transformer_recall': transformer_weight  # Transformer 召回权重（可调整）
    }
    
    # 只保留存在的召回方法的权重
    weight_dict = {k: v for k, v in weight_dict.items() if k in user_multi_recall_dict}
    
    print(f"\n权重设置:")
    for method, weight in weight_dict.items():
        print(f"  {method}: {weight}")
    
    # 5. 合并召回结果
    print(f"\n开始合并召回结果（topk={topk}）...")
    final_recall_items_dict_rank = combine_recall_results(
        user_multi_recall_dict, 
        weight_dict=weight_dict, 
        topk=topk
    )
    
    print(f"\n  ✓ 合并完成，共 {len(final_recall_items_dict_rank)} 个用户")
    
    # 6. 手动添加训练集中每个用户的最后一次点击到召回结果中
    print("\n" + "=" * 80)
    print("手动添加训练集的最后一次点击到召回结果...")
    print("=" * 80)
    
    trn_last_click_df = None
    try:
        all_click_df = pd.read_csv(os.path.join(data_path, 'train_click_log.csv'))
        trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
        
        # ⚠️ 诊断：检查用户集合的一致性
        recall_users = set(final_recall_items_dict_rank.keys())
        last_click_users = set(trn_last_click_df['user_id'].unique())
        hist_click_users = set(trn_hist_click_df['user_id'].unique())
        
        print(f"\n[诊断] 用户集合分析：")
        print(f"  召回结果中的用户数: {len(recall_users)}")
        print(f"  最后一次点击中的用户数: {len(last_click_users)}")
        print(f"  历史点击中的用户数: {len(hist_click_users)}")
        print(f"  只在召回结果中的用户数: {len(recall_users - last_click_users)}")
        print(f"  只在最后一次点击中的用户数: {len(last_click_users - recall_users)}")
        print(f"  只在历史点击中的用户数: {len(hist_click_users - recall_users)}")
        
        # 检查：如果用户不在召回结果中，但在最后一次点击中，这些用户会被添加
        users_to_add = last_click_users - recall_users
        if len(users_to_add) > 0:
            print(f"\n  ⚠️  警告：{len(users_to_add)} 个用户不在召回结果中，但会在最后一次点击中被添加")
            print(f"     这些用户可能只有一次点击，或者不在原始召回结果中")
            if len(users_to_add) <= 20:
                print(f"     示例用户: {list(users_to_add)[:20]}")
        
        # 检查：如果用户不在历史点击中，但在最后一次点击中，这些用户可能有问题
        users_only_in_last = last_click_users - hist_click_users
        if len(users_only_in_last) > 0:
            print(f"\n  ⚠️  警告：{len(users_only_in_last)} 个用户只在最后一次点击中，不在历史点击中")
            print(f"     这些用户可能只有一次点击（根据 get_hist_and_last_click 的逻辑，应该不会出现这种情况）")
            if len(users_only_in_last) <= 20:
                print(f"     示例用户: {list(users_only_in_last)[:20]}")
        
        # 将最后一次点击添加到召回结果中
        added_count = 0
        updated_count = 0
        users_added_to_recall = set()  # 记录被添加到召回结果中的新用户
        
        for _, row in tqdm(trn_last_click_df.iterrows(), total=len(trn_last_click_df), desc="添加最后一次点击"):
            user_id = row['user_id']
            last_item = row['click_article_id']
            
            # 如果用户不在召回结果中，创建一个新的召回列表
            if user_id not in final_recall_items_dict_rank:
                final_recall_items_dict_rank[user_id] = []
                users_added_to_recall.add(user_id)
            
            # 检查最后一次点击是否已经在召回列表中
            recall_items = [item for item, score in final_recall_items_dict_rank[user_id]]
            
            if last_item not in recall_items:
                # 如果不在，添加到列表的最前面（最高分），确保它在 topk 中
                # 使用一个很高的分数，确保它排在前面
                max_score = max([score for _, score in final_recall_items_dict_rank[user_id]], default=0.0)
                final_recall_items_dict_rank[user_id].insert(0, (last_item, max_score + 1.0))
                added_count += 1
            else:
                # 如果已经在列表中，将其移到最前面并提高分数
                recall_list = final_recall_items_dict_rank[user_id]
                # 找到该物品的位置
                for i, (item, score) in enumerate(recall_list):
                    if item == last_item:
                        # 移除旧的位置
                        recall_list.pop(i)
                        # 添加到最前面并提高分数
                        max_score = max([s for _, s in recall_list], default=0.0)
                        recall_list.insert(0, (item, max_score + 1.0))
                        updated_count += 1
                        break
        
        print(f"\n[诊断] 添加最后一次点击后的统计：")
        print(f"  ✓ 已为 {added_count} 个用户添加了最后一次点击（之前不在召回列表中）")
        print(f"  ✓ 已更新 {updated_count} 个用户的最后一次点击（已在召回列表中，已提升优先级）")
        print(f"  ✓ 新增了 {len(users_added_to_recall)} 个用户到召回结果中（之前不在召回结果中）")
        
        # 检查新增的用户是否在历史点击中
        if len(users_added_to_recall) > 0:
            users_added_in_hist = users_added_to_recall & hist_click_users
            users_added_not_in_hist = users_added_to_recall - hist_click_users
            print(f"    其中 {len(users_added_in_hist)} 个用户在历史点击中（这些用户会被 news4.py 处理）")
            print(f"    其中 {len(users_added_not_in_hist)} 个用户不在历史点击中（这些用户可能不会被 news4.py 处理）")
            if len(users_added_not_in_hist) > 0 and len(users_added_not_in_hist) <= 20:
                print(f"    不在历史点击中的用户示例: {list(users_added_not_in_hist)[:20]}")
        
        # 重新截取 topk（因为添加了新的物品）
        for user_id in final_recall_items_dict_rank:
            final_recall_items_dict_rank[user_id] = final_recall_items_dict_rank[user_id][:topk]
        
        # 填充少于5个物品的用户（使用热门物品）
        print("\n" + "=" * 80)
        print("填充少于5个物品的用户（使用热门物品）...")
        print("=" * 80)
        
        # 获取热门物品（从训练数据中统计）
        try:
            all_click_df = pd.read_csv(os.path.join(data_path, 'train_click_log.csv'))
            hot_items = all_click_df['click_article_id'].value_counts().index.tolist()
            print(f"  ✓ 已获取 {len(hot_items)} 个热门物品")
        except Exception as e:
            print(f"  ✗ 获取热门物品失败: {e}")
            hot_items = []
        
        if len(hot_items) > 0:
            min_items_per_user = 5
            users_need_fill = []
            fill_count = 0
            
            for user_id, recall_list in final_recall_items_dict_rank.items():
                if len(recall_list) < min_items_per_user:
                    users_need_fill.append(user_id)
                    # 获取用户已有的物品
                    existing_items = set([item for item, _ in recall_list])
                    
                    # 从热门物品中选择未包含的物品进行填充
                    needed_count = min_items_per_user - len(recall_list)
                    filled_items = []
                    min_score = min([score for _, score in recall_list], default=0.0)
                    
                    for hot_item in hot_items:
                        if hot_item not in existing_items:
                            filled_items.append((hot_item, min_score - 0.1))  # 使用较低的分数
                            if len(filled_items) >= needed_count:
                                break
                    
                    # 添加到召回列表
                    final_recall_items_dict_rank[user_id].extend(filled_items)
                    # 重新排序
                    final_recall_items_dict_rank[user_id] = sorted(
                        final_recall_items_dict_rank[user_id],
                        key=lambda x: x[1],
                        reverse=True
                    )[:topk]
                    fill_count += len(filled_items)
            
            if len(users_need_fill) > 0:
                print(f"  ✓ 已为 {len(users_need_fill)} 个用户填充了 {fill_count} 个热门物品")
                print(f"  ✓ 填充后，所有用户至少有 {min_items_per_user} 个物品")
                if len(users_need_fill) <= 20:
                    print(f"     需要填充的用户示例: {users_need_fill[:20]}")
            else:
                print(f"  ✓ 所有用户都有至少 {min_items_per_user} 个物品，无需填充")
        
        # 重新保存合并后的结果（包含最后一次点击和填充）
        final_recall_items_dict = {}
        for user_id, recall_list in final_recall_items_dict_rank.items():
            final_recall_items_dict[user_id] = {item: score for item, score in recall_list}
        
        pickle.dump(
            final_recall_items_dict, 
            open(os.path.join(save_path, 'final_recall_items_dict.pkl'), 'wb')
        )
        print(f"\n  ✓ 已更新保存的召回结果（包含最后一次点击）")
        print(f"  ✓ 最终召回结果包含 {len(final_recall_items_dict)} 个用户")
        
    except Exception as e:
        print(f"  ✗ 添加最后一次点击失败: {e}")
        import traceback
        traceback.print_exc()
        print("  （继续执行，但召回结果可能不包含最后一次点击）")
    
    # 7. 评估合并后的召回效果（可选）
    if evaluate:
        print("\n" + "=" * 80)
        print("评估合并后的召回效果...")
        print("=" * 80)
        
        # 如果之前已经加载了训练数据，直接使用；否则重新加载
        if trn_last_click_df is None:
            try:
                all_click_df = pd.read_csv(os.path.join(data_path, 'train_click_log.csv'))
                trn_hist_click_df, trn_last_click_df = get_hist_and_last_click(all_click_df)
            except Exception as e:
                print(f"  ✗ 加载训练数据失败: {e}")
                trn_last_click_df = None
        
        if trn_last_click_df is not None:
            print(f"\n评估数据: {len(trn_last_click_df)} 个用户")
            metrics_recall(final_recall_items_dict_rank, trn_last_click_df, topk=50)
        else:
            print("  ✗ 无法加载训练数据，跳过评估")
    
    return final_recall_items_dict_rank


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='合并 Transformer 召回结果到多路召回中')
    parser.add_argument('--use_sample', action='store_true', help='使用采样模式的结果')
    parser.add_argument('--transformer_weight', type=float, default=2, help='Transformer 召回的权重（默认 1.5）')
    parser.add_argument('--topk', type=int, default=150, help='最终合并后每个用户召回的物品数量（默认 150）')
    parser.add_argument('--no_evaluate', action='store_true', help='不评估合并后的召回效果')
    
    args = parser.parse_args()
    
    # 执行合并
    final_recall = merge_transformer_recall(
        use_sample=args.use_sample,
        transformer_weight=args.transformer_weight,
        topk=args.topk,
        evaluate=not args.no_evaluate
    )
    
    if final_recall is not None:
        print("\n" + "=" * 80)
        print("合并完成！")
        print("=" * 80)
        print(f"最终召回结果已保存到: {save_path}final_recall_items_dict.pkl")
        print(f"合并后的召回字典包含 {len(final_recall)} 个用户")
    else:
        print("\n合并失败，请检查错误信息")


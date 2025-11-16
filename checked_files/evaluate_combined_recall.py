"""
评估多路召回合并后的召回率
检查最后一次点击是否在召回结果中
"""
import pickle
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

save_path = '../temp_results/'
data_path = '../data_raw/'

print("=" * 80)
print("多路召回合并后的召回率评估")
print("=" * 80)

# 1. 加载最后一次点击数据
print("\n1. 加载测试数据...")
try:
    all_click_df = pd.read_csv(data_path + 'train_click_log.csv')
    print(f"  训练数据: {len(all_click_df)} 条记录")
    
    # 获取最后一次点击
    def get_hist_and_last_click(all_click):
        all_click = all_click.sort_values(['user_id', 'click_timestamp'])
        click_list = []
        for user_id, hist_click in tqdm(all_click.groupby('user_id'), desc="处理用户"):
            click_list.append({
                'user_id': user_id,
                'click_article_id': hist_click['click_article_id'].tolist(),
                'click_timestamp': hist_click['click_timestamp'].tolist()
            })
        
        last_click_list = []
        for click in click_list:
            if len(click['click_article_id']) > 1:
                last_click_list.append({
                    'user_id': click['user_id'],
                    'click_article_id': click['click_article_id'][-1],
                    'click_timestamp': click['click_timestamp'][-1]
                })
        
        return pd.DataFrame(last_click_list)
    
    trn_last_click_df = get_hist_and_last_click(all_click_df)
    print(f"  最后一次点击数据: {len(trn_last_click_df)} 用户")
    
except Exception as e:
    print(f"  ✗ 加载数据失败: {e}")
    exit(1)

# 2. 加载各路召回结果
print("\n2. 加载各路召回结果...")
user_multi_recall_dict = {}
recall_files = {
    'itemcf_sim_itemcf_recall': 'itemcf_u2i_dict.pkl',
    'embedding_sim_item_recall': 'embedding_u2i_dict.pkl',
    'youtubednn_recall': 'youtube_u2i_dict.pkl',
    'youtubednn_usercf_recall': 'youtube_u2u_sim.pkl',
    'cold_start_recall': 'cold_start_user_items_dict.pkl'
}

for method, filename in recall_files.items():
    filepath = os.path.join(save_path, filename)
    if os.path.exists(filepath):
        try:
            user_multi_recall_dict[method] = pickle.load(open(filepath, 'rb'))
            print(f"  ✓ {method}: {len(user_multi_recall_dict[method])} 用户")
        except Exception as e:
            print(f"  ✗ {method}: 加载失败 - {e}")
    else:
        print(f"  ✗ {method}: 文件不存在")

# 3. 评估各路召回的召回率
print("\n3. 各路召回的召回率评估:")
def metrics_recall(user_recall_items_dict, trn_last_click_df, topk_list=[10, 20, 50, 100, 150]):
    """评估召回率"""
    last_click_item_dict = dict(zip(trn_last_click_df['user_id'], trn_last_click_df['click_article_id']))
    user_num = len(user_recall_items_dict)
    
    results = {}
    for k in topk_list:
        hit_num = 0
        valid_user_num = 0
        for user, item_list in user_recall_items_dict.items():
            if user not in last_click_item_dict:
                continue
            valid_user_num += 1
            # 获取前k个召回的结果
            if isinstance(item_list, list):
                tmp_recall_items = [x[0] if isinstance(x, tuple) else x for x in item_list[:k]]
            elif isinstance(item_list, dict):
                tmp_recall_items = list(item_list.keys())[:k]
            else:
                continue
            
            if last_click_item_dict[user] in tmp_recall_items:
                hit_num += 1
        
        if valid_user_num > 0:
            hit_rate = hit_num / valid_user_num
            results[k] = {'hit_num': hit_num, 'hit_rate': hit_rate, 'valid_user_num': valid_user_num}
        else:
            results[k] = {'hit_num': 0, 'hit_rate': 0.0, 'valid_user_num': 0}
    
    return results

# 评估各路召回
single_recall_results = {}
for method, recall_dict in user_multi_recall_dict.items():
    if len(recall_dict) == 0:
        continue
    print(f"\n  {method}:")
    results = metrics_recall(recall_dict, trn_last_click_df)
    single_recall_results[method] = results
    for k, res in results.items():
        print(f"    Top-{k:3d}: hit_num={res['hit_num']:5d}, hit_rate={res['hit_rate']:.5f}, valid_users={res['valid_user_num']}")

# 4. 加载并评估合并后的召回结果
print("\n4. 合并后的召回率评估:")
final_recall_file = os.path.join(save_path, 'final_recall_items_dict.pkl')
if os.path.exists(final_recall_file):
    try:
        # 尝试加载最终召回结果（可能是字典或列表格式）
        final_recall_data = pickle.load(open(final_recall_file, 'rb'))
        
        # 检查数据格式
        if isinstance(final_recall_data, dict):
            # 检查第一个值是什么格式
            first_user = list(final_recall_data.keys())[0]
            first_value = final_recall_data[first_user]
            
            if isinstance(first_value, list):
                # 列表格式：[(item, score), ...]
                final_recall_dict = final_recall_data
            elif isinstance(first_value, dict):
                # 字典格式：{item: score, ...}
                # 转换为列表格式
                final_recall_dict = {}
                for user, item_dict in final_recall_data.items():
                    final_recall_dict[user] = sorted(item_dict.items(), key=lambda x: x[1], reverse=True)
            else:
                print(f"  ✗ 未知的数据格式: {type(first_value)}")
                final_recall_dict = {}
        else:
            print(f"  ✗ 未知的数据格式: {type(final_recall_data)}")
            final_recall_dict = {}
        
        if len(final_recall_dict) > 0:
            print(f"  ✓ 加载成功: {len(final_recall_dict)} 用户")
            combined_results = metrics_recall(final_recall_dict, trn_last_click_df)
            print("\n  合并后的召回率:")
            for k, res in combined_results.items():
                print(f"    Top-{k:3d}: hit_num={res['hit_num']:5d}, hit_rate={res['hit_rate']:.5f}, valid_users={res['valid_user_num']}")
            
            # 对比单路召回和合并后的效果
            print("\n5. 效果对比 (Top-50):")
            print(f"  {'召回方法':<30} {'命中数':<10} {'召回率':<10}")
            print("  " + "-" * 50)
            for method, results in single_recall_results.items():
                if 50 in results:
                    res = results[50]
                    print(f"  {method:<30} {res['hit_num']:<10} {res['hit_rate']:.5f}")
            if 50 in combined_results:
                res = combined_results[50]
                print(f"  {'合并后':<30} {res['hit_num']:<10} {res['hit_rate']:.5f}")
        else:
            print("  ✗ 数据格式不正确")
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"  ✗ 文件不存在: {final_recall_file}")
    print("  提示: 需要先运行 news3.py 生成合并后的召回结果")

print("\n" + "=" * 80)

